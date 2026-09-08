"""
sbdd_pipeline.py — генерация лигандов DiffSBDD + докинг через Vina,
с resume по JSONL и state.json.

Запускать в env molgen (тут же живут RDKit/Meeko/vina.exe вызовы из
dock_existing_candidates.py). Сама генерация DiffSBDD требует
несовместимого стека (torch 2.0.1+cu118, numpy<2, rdkit 2022.03.2 —
см. DiffSBDD/environment.yaml) и поэтому запускается отдельным
процессом в env diffsbdd через DiffSBDD/generate_smiles_seeded.py.

Архитектура резюмируемости (проверить в шаге 6 промта, ДО реальных
замеров — обязательно прогнать: запустить, убить на середине,
запустить снова, проверить что не начинает с нуля):

  1. Генерация идемпотентна на уровне сида: DiffSBDD/generate_smiles_seeded.py
     сам пропускает сид, если файл смайлов для него уже существует
     и непустой. Стадия генерации сама по себе не пишет частичных
     файлов — либо файла нет, либо он дописан целиком (json.dump в
     конце, после того как все molecules сгенерированы). Генерация
     одного сида ограничена таймаутом (--gen_timeout_sec) и не роняет
     весь прогон при зависании/падении — сид помечается failed в
     state.json и пропускается, остальные сиды продолжаются.

  2. Полный список (seed, index, smiles) собирается ЗАНОВО на каждом
     запуске из уже сгенерированных per-seed файлов. Resume докинга —
     по СОДЕРЖИМОМУ, не по позиции: из уже записанных строк JSONL
     строится множество завершённых (seed, seed_local_index), и
     докуются только лиганды, которых там ещё нет (load_completed_keys).
     Это специально НЕ позиционный resume (не "первые N строк") — если
     бы один сид упал в первом прогоне, а на resume сгенерировался
     успешно, его лиганды вставились бы в середину списка и сдвинули
     позиции всех следующих сидов, ломая позиционный подсчёт.

  3. Каждая строка JSONL пишется атомарно одним write() + flush()
     + os.fsync() — обрыв процесса может оставить только целые
     строки, не потерять и не испортить частичную. Каждая строка несёт
     exhaustiveness/dock_timeout, с которыми она задокирована — если
     эти параметры поменяются между запусками (например, в замерах
     параллелизма из шага 7.3), это видно по данным, а не теряется.

  4. state.json обновляется после каждого сида (генерация) и
     периодически во время докинга — для быстрой человекочитаемой
     проверки прогресса, но НЕ является источником истины для resume
     докинга (источник истины — содержимое JSONL, см. п.2); если
     state.json и JSONL разойдутся (обрыв между их обновлениями),
     авторитетен JSONL.

  5. Манифест (--seeds + --n_per_seed) фиксируется в state.json при
     первом запуске в данной run_dir и сверяется при каждом
     последующем — расхождение останавливает прогон с явной ошибкой
     ДО того, как resume молча привяжет скоры не к тем SMILES
     (check_or_record_manifest).

  6. Перед генерацией — контрольный докинг известного со-
     кристаллизованного ингибитора (run_control_docking), как в
     module_generative/iterative_finetune_loop.py и по той же причине
     (история PIK3CA/9CMK): если бокс/рецептор сломан, это видно СРАЗУ,
     а не после часов впустую потраченной генерации+докинга.

Использование (dev/smoke-прогон на маленьких числах):
  python sbdd_pipeline.py --gene PIK3CA --seeds 0 1 --n_per_seed 5
      --run_dir runs/smoke

Боевой прогон (числа НЕ финальные — их считает шаг 12 промта):
  python sbdd_pipeline.py --gene PIK3CA --seeds 0 1 2 3 4 5 6 7
      --n_per_seed 1000 --run_dir runs/full
"""

import argparse
import json
import os
import subprocess
import sys
import time

from rdkit import Chem

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from dock_existing_candidates import (  # noqa: E402
    DockingError, prepare_receptor, dock_smiles_isolated,
)
from gene_target_utils import GeneTargetError, resolve_gene_to_docking_target  # noqa: E402

DIFFSBDD_DIR = os.path.join(BASE_DIR, "DiffSBDD")
DIFFSBDD_CHECKPOINT = os.path.join(DIFFSBDD_DIR, "checkpoints", "crossdocked_fullatom_cond.ckpt")
DIFFSBDD_GEN_SCRIPT = os.path.join(DIFFSBDD_DIR, "generate_smiles_seeded.py")

# diffsbdd — env-сосед molgen (оба под ...\miniconda3\envs\<name>), путь
# вычисляется относительно текущего python, а не хардкодится под
# конкретного пользователя/диск.
_ENVS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(sys.executable)))
DIFFSBDD_PYTHON = os.path.join(_ENVS_DIR, "diffsbdd", "python.exe")

# 4JPS: цепь A, номер остатка 1102 — со-кристаллизованный алпелисиб
# (1LT), см. structures/4JPS.pdb HETATM записи. Используется НЕОЧИЩЕННЫЙ
# PDB (не 4JPS_clean.pdb — там лиганд уже вырезан strip_ligands_and_waters
# ради Vina-рецептора, а DiffSBDD нужен исходный файл с лигандом, чтобы
# --ref_ligand нашёл carman по HETATM).
REF_LIGAND_BY_PDB = {
    "4JPS": "A:1102",
}


def state_path(run_dir):
    return os.path.join(run_dir, "state.json")


def jsonl_path(run_dir):
    return os.path.join(run_dir, "results.jsonl")


def control_path(run_dir):
    return os.path.join(run_dir, "control.json")


class PipelineAborted(Exception):
    """Остановка по control.json — не ошибка, а сознательное вмешательство
    (перенято из CONTROL.json оркестратора пилота: способ мягко
    остановить многочасовой прогон, не убивая процесс силой и не теряя
    уже записанный прогресс)."""


def check_control(run_dir):
    """Проверяется перед каждым сидом генерации и перед каждым лигандом
    докинга. Отсутствие файла — норм (ничего не просили). abort=true
    останавливает прогон ЧИСТО: текущее состояние уже на диске (JSONL
    построчно, state.json после каждого шага), просто выходим, ничего
    досрочно не обрываем на середине записи."""
    p = control_path(run_dir)
    if not os.path.exists(p):
        return
    try:
        with open(p, "r", encoding="utf-8") as f:
            control = json.load(f)
    except (OSError, json.JSONDecodeError):
        return  # битый control.json — не роняем из-за него прогон, просто игнорируем
    if control.get("abort"):
        note = control.get("note", "")
        raise PipelineAborted(f"Остановлено через control.json (abort=true){': ' + note if note else ''}")


def load_state(run_dir):
    p = state_path(run_dir)
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"seeds": {}, "docking": {"n_docked": 0}, "started_at": time.time()}


def save_state(run_dir, state):
    state["updated_at"] = time.time()
    tmp = state_path(run_dir) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    os.replace(tmp, state_path(run_dir))  # атомарная замена — не оставит битый state.json


def append_jsonl_atomic(path, record):
    line = json.dumps(record, ensure_ascii=False) + "\n"
    with open(path, "a", encoding="utf-8") as f:
        f.write(line)
        f.flush()
        os.fsync(f.fileno())


def load_completed_keys(path):
    """Источник истины для resume докинга — МНОЖЕСТВО уже задокированных
    (seed, seed_local_index), а не число строк. Раньше resume был
    позиционным (по числу строк) и предполагал, что all_ligands
    пересобирается КАЖДЫЙ раз в идентичном порядке — но если один сид
    падает в первом прогоне, а на resume генерируется успешно, его
    лиганды вставляются в середину списка и сдвигают позиции всех
    следующих сидов, ломая позиционный resume (скоры привязались бы не
    к тем SMILES). Ключ по (seed, local_index) устойчив к такому сдвигу.

    Игнорирует потенциальную незакрытую последнюю строку (на случай
    обрыва в середине fsync — подстраховка сверх атомарной записи, не
    единственная линия защиты)."""
    completed = set()
    if not os.path.exists(path):
        return completed
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                print(f"[resume] последняя строка JSONL повреждена/не дописана, дальше не читаю")
                break
            completed.add((rec["seed"], rec["seed_local_index"]))
    return completed


class SeedGenerationFailed(Exception):
    """Генерация для одного сида не удалась (упала или зависла) — не
    должна ронять весь прогон остальных сидов, но должна быть громко
    видна (state.json + печать), а не тихо проигнорирована."""


def generate_seed(seed, n_per_seed, pdbfile, ref_ligand, run_dir, timesteps=None, gen_timeout_sec=7200, batch_size=None):
    """Поднимает SeedGenerationFailed при падении/таймауте — вызывающий
    код решает, пропустить ли сид и продолжить остальные.

    batch_size=None -> дефолт generate_smiles_seeded.py (сейчас 4, выбран
    консервативно под 4 ГБ VRAM). Больший batch_size ускоряет генерацию,
    но может дать OOM — теперь это не роняет весь прогон (SeedGenerationFailed
    ловится вызывающим кодом, сид просто помечается failed и пропускается),
    но конкретный безопасный максимум для этой карты не определён
    экспериментально (шаг 8 исходного промта) — можно смело пробовать
    большие значения и уменьшать при падении по памяти."""
    seed_dir = os.path.join(run_dir, "generation")
    os.makedirs(seed_dir, exist_ok=True)
    outfile = os.path.join(seed_dir, f"seed_{seed}.json")
    sdf_outfile = os.path.join(seed_dir, f"seed_{seed}.sdf")

    if os.path.exists(outfile) and os.path.getsize(outfile) > 0:
        print(f"[seed {seed}] генерация уже есть, пропуск: {outfile}")
    else:
        if not os.path.exists(DIFFSBDD_PYTHON):
            raise RuntimeError(
                f"Не найден python окружения diffsbdd: {DIFFSBDD_PYTHON}. "
                f"conda env diffsbdd должен быть создан заранее (conda env create -f DiffSBDD/environment.yaml)."
            )
        cmd = [
            DIFFSBDD_PYTHON, DIFFSBDD_GEN_SCRIPT,
            DIFFSBDD_CHECKPOINT, pdbfile, ref_ligand,
            "--seed", str(seed),
            "--n_samples", str(n_per_seed),
            "--outfile", outfile,
            "--sdf_outfile", sdf_outfile,
        ]
        if timesteps is not None:
            cmd += ["--timesteps", str(timesteps)]
        if batch_size is not None:
            cmd += ["--batch_size", str(batch_size)]
        print(f"[seed {seed}] генерация {n_per_seed} молекул (batch_size={batch_size or 'дефолт'}, таймаут {gen_timeout_sec}с)...")

        # ВАЖНО (см. отдельную изоляцию докинга в _dock_worker.py/
        # dock_smiles_isolated по той же причине): генерация DiffSBDD —
        # это тоже CUDA/torch код, который в принципе может зависнуть
        # (драйвер, фрагментация VRAM и т.д.). Без таймаута зависший сид
        # останавливает весь недельный прогон молча, без вывода.
        # generate_smiles_seeded.py — один процесс без дочерних
        # (в отличие от докинга, где vina.exe/meeko — отдельные
        # бинарники-внуки), поэтому одного kill() достаточно, отдельный
        # taskkill по внукам не нужен.
        t0 = time.time()
        log_path = os.path.join(seed_dir, f"seed_{seed}.log")
        # PYTHONUTF8/PYTHONIOENCODING: без них дочерний python.exe (env
        # diffsbdd) пишет print() в перенаправленный файл через кодировку
        # локали Windows (cp1251 на этой машине), а не UTF-8 — сам файл
        # открыт здесь как encoding="utf-8" для ЧТЕНИЯ ПОСЛЕ, и несовпадение
        # даёт кракозябры при перепечати лога в консоль ниже. Данные
        # (JSON/JSONL) это не портит — там всегда явный encoding="utf-8" на
        # запись и чтение — но человекочитаемый лог выглядит битым.
        child_env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
        with open(log_path, "w", encoding="utf-8") as log_fh:
            proc = subprocess.Popen(cmd, cwd=DIFFSBDD_DIR, stdout=log_fh, stderr=subprocess.STDOUT, env=child_env)
            try:
                proc.wait(timeout=gen_timeout_sec)
                timed_out = False
            except subprocess.TimeoutExpired:
                proc.kill()
                try:
                    proc.wait(timeout=10)
                except Exception:
                    pass
                timed_out = True
        elapsed = time.time() - t0

        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            log_output = f.read()
        print(log_output[-3000:])

        if timed_out:
            raise SeedGenerationFailed(
                f"[seed {seed}] генерация превысила таймаут {gen_timeout_sec}с и была прибита "
                f"(прошло {elapsed:.1f}с). Возможное зависание CUDA/torch."
            )
        if proc.returncode != 0 or not os.path.exists(outfile):
            raise SeedGenerationFailed(
                f"[seed {seed}] генерация упала (returncode={proc.returncode}, {elapsed:.1f}с).\n"
                f"Хвост лога:\n{log_output[-3000:]}"
            )

    with open(outfile, "r", encoding="utf-8") as f:
        return json.load(f)


def check_or_record_manifest(state, seeds, n_per_seed):
    """Resume докинга устойчив к перестановке уже завершённых сидов
    (см. load_completed_keys), но НЕ к смене состава/числа сидов между
    запусками — другой --n_per_seed или другой набор --seeds означает
    другой список кандидатов на генерацию, а не то же самое продолжение.
    Первый запуск фиксирует манифест в state.json, а все последующие
    сверяются с ним и ОТКАЗЫВАЮТСЯ продолжать при расхождении."""
    manifest = {"seeds": list(seeds), "n_per_seed": n_per_seed}
    if "manifest" not in state:
        state["manifest"] = manifest
        return
    old = state["manifest"]
    if old["seeds"] != manifest["seeds"] or old["n_per_seed"] != manifest["n_per_seed"]:
        raise RuntimeError(
            f"Параметры прогона не совпадают с тем, с чего начинали resume в этой run_dir:\n"
            f"  было:  seeds={old['seeds']}, n_per_seed={old['n_per_seed']}\n"
            f"  сейчас: seeds={manifest['seeds']}, n_per_seed={manifest['n_per_seed']}\n"
            f"Resume по числу строк JSONL при таком расхождении привяжет скоры к "
            f"неправильным SMILES. Используй те же --seeds/--n_per_seed, что и в "
            f"первом запуске, или новую --run_dir для другого набора параметров."
        )


def run_control_docking(target, receptor_pdbqt, exhaustiveness, dock_timeout, max_score=-5.0):
    """Докует известный со-кристаллизованный ингибитор (не сгенерированную
    молекулу) ПЕРЕД тем, как тратить время на генерацию+докинг целой
    партии — та же защита, что run_control_docking в
    module_generative/iterative_finetune_loop.py, и по той же причине
    (см. историю PIK3CA/9CMK в gene_target_utils.py): бокс/рецептор
    может быть тихо сломан, и без этой проверки мы узнаем об этом
    только после того, как уже сгенерировали и задокировали сотни
    молекул впустую."""
    ligand_chembl_id = target.get("matched_known_ligand_chembl_id")
    ligand_name = target.get("matched_known_ligand")
    if not ligand_chembl_id:
        raise RuntimeError(
            "Нет ChEMBL ID известного лиганда структуры (matched_known_ligand_chembl_id) — "
            "не могу выполнить контрольный докинг перед прогоном."
        )

    from gene_target_utils import get_chembl_new_client

    try:
        new_client = get_chembl_new_client()
        rec = new_client.molecule.get(ligand_chembl_id)
        smiles = (rec.get("molecule_structures") or {}).get("canonical_smiles")
    except Exception as e:
        raise RuntimeError(f"Не удалось получить SMILES контрольного лиганда {ligand_chembl_id}: {str(e)[:200]}")
    if not smiles:
        raise RuntimeError(f"У контрольного лиганда {ligand_chembl_id} нет SMILES в ChEMBL.")

    print(f"\n--- Контрольный докинг перед прогоном: {ligand_name} ({ligand_chembl_id}) в {target['pdb_id']} ---")
    workdir = os.path.join(os.path.dirname(receptor_pdbqt), "_control_dock_tmp")
    os.makedirs(workdir, exist_ok=True)
    score = dock_smiles_isolated(
        smiles, receptor_pdbqt, target["box_center"], target["box_size"],
        workdir, tag="control", exhaustiveness=exhaustiveness, timeout=max(dock_timeout, 120),
    )
    print(f"  Docking score контроля: {score}")

    if score is None:
        raise RuntimeError(
            f"Контрольный докинг {ligand_name} не удался (вернул None) — "
            f"настройка рецептора/бокса сломана, прогон остановлен ДО генерации."
        )
    if score > max_score:
        raise RuntimeError(
            f"Контрольный докинг {ligand_name} дал аномальный скор {score} ккал/моль "
            f"(порог {max_score}) — настройка докинга, вероятно, сломана, прогон остановлен "
            f"ДО генерации, чтобы не тратить время впустую."
        )
    print("  Контроль пройден: скор в разумном диапазоне.\n")
    return score


def run_pipeline(gene, seeds, n_per_seed, run_dir, exhaustiveness=8, dock_timeout=120, timesteps=None, gen_timeout_sec=7200, batch_size=None):
    # ОБЯЗАТЕЛЬНО абсолютный путь: generate_seed() запускает генерацию
    # ПОДПРОЦЕССОМ с cwd=DIFFSBDD_DIR (генератор — в env diffsbdd, ему
    # нужно запускаться из своей папки). Если run_dir относительный,
    # --outfile/--sdf_outfile в этом подпроцессе резолвятся ОТНОСИТЕЛЬНО
    # DIFFSBDD_DIR, а не относительно текущей директории, откуда запущен
    # sbdd_pipeline.py — файл реально пишется (генерация не падает), просто
    # не туда, куда потом смотрит эта же функция. Итог без этой правки:
    # generate_seed() считает сид проваленным (outfile "не найден"), хотя
    # генерация прошла успешно — проверено вживую (DiffSBDD\runs\...
    # вместо runs\...).
    run_dir = os.path.abspath(run_dir)
    os.makedirs(run_dir, exist_ok=True)
    state = load_state(run_dir)
    results_path = jsonl_path(run_dir)
    check_or_record_manifest(state, seeds, n_per_seed)

    try:
        target = resolve_gene_to_docking_target(gene)
    except GeneTargetError as e:
        raise RuntimeError(f"Не удалось определить структуру-мишень для {gene}: {e}")
    if target is None:
        raise RuntimeError(f"Структура-мишень для {gene} не найдена.")

    pdb_id = target["pdb_id"]
    if pdb_id not in REF_LIGAND_BY_PDB:
        raise RuntimeError(
            f"Для {pdb_id} не задан ref_ligand (chain:resi со-кристаллизованного лиганда) в "
            f"REF_LIGAND_BY_PDB — нужно проверить structures/{pdb_id}.pdb вручную и добавить."
        )
    ref_ligand = REF_LIGAND_BY_PDB[pdb_id]
    pdbfile = target["pdb_path"]  # НЕ *_clean.pdb — см. комментарий у REF_LIGAND_BY_PDB

    receptor_basename = os.path.join(
        os.path.dirname(target["pdb_path"]), f"{pdb_id}_receptor"
    )
    receptor_pdbqt = prepare_receptor(
        target["pdb_path"], target["box_center"], target["box_size"], receptor_basename
    )
    print(f"Рецептор для докинга готов: {receptor_pdbqt}")

    state["gene"] = gene
    state["pdb_id"] = pdb_id
    state["receptor_pdbqt"] = receptor_pdbqt
    save_state(run_dir, state)

    # Контроль ДО генерации — см. docstring run_control_docking. Пропускается
    # только если уже пройден в этой run_dir (не дублировать на resume).
    if not state.get("control_docking_passed"):
        run_control_docking(target, receptor_pdbqt, exhaustiveness, dock_timeout)
        state["control_docking_passed"] = True
        save_state(run_dir, state)
    else:
        print("Контрольный докинг уже пройден в этой run_dir ранее (resume) — не повторяю.")

    # --- фаза генерации (идемпотентна по сидам, см. docstring) ---
    all_ligands = []  # [(seed, local_index, smiles), ...] — детерминированный порядок
    for seed in seeds:
        check_control(run_dir)
        try:
            gen_result = generate_seed(
                seed, n_per_seed, pdbfile, ref_ligand, run_dir,
                timesteps=timesteps, gen_timeout_sec=gen_timeout_sec, batch_size=batch_size,
            )
        except SeedGenerationFailed as e:
            print(f"ПРЕДУПРЕЖДЕНИЕ: {e}\nСид {seed} пропущен, продолжаю с остальными.")
            state["seeds"][str(seed)] = {"status": "failed", "error": str(e)}
            save_state(run_dir, state)
            continue
        state["seeds"][str(seed)] = {
            "status": "done",
            "n_valid_smiles": gen_result["n_valid_smiles"],
            "generation_sec": gen_result["generation_sec"],
            "checkpoint": gen_result.get("checkpoint"),
            "timesteps": gen_result.get("timesteps"),
        }
        save_state(run_dir, state)
        for i, smi in enumerate(gen_result["smiles"]):
            all_ligands.append((seed, i, smi))

    print(f"\nВсего сгенерировано валидных SMILES по всем сидам: {len(all_ligands)}")

    # --- фаза докинга (resume по (seed, local_index), см. load_completed_keys) ---
    completed = load_completed_keys(results_path)
    if completed:
        print(f"[resume] {len(completed)} лигандов уже задокированы (по {results_path}), продолжаю с этого места")

    todo = [(seed, local_i, smi) for (seed, local_i, smi) in all_ligands if (seed, local_i) not in completed]

    workdir = os.path.join(run_dir, "dock_tmp")
    os.makedirs(workdir, exist_ok=True)

    n_ok = 0
    n_fail = 0
    n_done_total = len(completed)
    for pos, (seed, local_i, smi) in enumerate(todo):
        check_control(run_dir)
        t0 = time.time()
        score = dock_smiles_isolated(
            smi, receptor_pdbqt, target["box_center"], target["box_size"],
            workdir, tag=f"lig{seed}_{local_i}", exhaustiveness=exhaustiveness, timeout=dock_timeout,
        )
        dock_sec = time.time() - t0

        # Heavy-atom count + Ligand Efficiency (-score/heavy_atoms) — та же
        # size-нормализация, что использует test_b_generation.py пилота при
        # сравнении с baseline, и прямой ответ на то, что сами авторы
        # DiffSBDD пишут в статье: "докинг-скор сильно коррелирует с
        # размером молекулы". Считаем здесь, а не в analysis-скрипте —
        # чтобы это было в самих данных, а не пересчитывалось задним числом.
        # Формальный заряд ДО Meeko — не фильтр и не блокирует ничего (см.
        # Meeko issue #63, github.com/forlilab/Meeko/issues/63: заряд может
        # меняться при подготовке лиганда — открытый вопрос даже для самих
        # мейнтейнеров Meeko, баг это или ожидаемое присвоение состояния
        # протонирования). Просто провенанс: если позже понадобится
        # диагностировать аномальные скоры, будет с чем сравнить, а не
        # пересчитывать задним числом по одному SMILES за раз.
        heavy_atoms = None
        ligand_efficiency = None
        formal_charge = None
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            heavy_atoms = mol.GetNumHeavyAtoms()
            formal_charge = Chem.GetFormalCharge(mol)
            if score is not None and heavy_atoms > 0:
                ligand_efficiency = round(-score / heavy_atoms, 4)

        record = {
            "seed": seed,
            "seed_local_index": local_i,
            "smiles": smi,
            "pdb_id": pdb_id,
            "docking_score_kcal_mol": score,
            "heavy_atom_count": heavy_atoms,
            "ligand_efficiency": ligand_efficiency,
            "formal_charge_pre_meeko": formal_charge,
            "dock_sec": round(dock_sec, 2),
            "exhaustiveness": exhaustiveness,
            "dock_timeout": dock_timeout,
            "timestamp": time.time(),
        }
        append_jsonl_atomic(results_path, record)
        n_done_total += 1

        if score is None:
            n_fail += 1
            print(f"  [{n_done_total}/{len(all_ligands)}] докинг не удался (seed {seed})")
        else:
            n_ok += 1
            print(f"  [{n_done_total}/{len(all_ligands)}] score={score:.2f} ккал/моль (seed {seed}, {dock_sec:.1f}с)")

        state["docking"] = {"n_docked": n_done_total, "n_ok": n_ok, "n_fail": n_fail, "total": len(all_ligands)}
        save_state(run_dir, state)

    print(
        f"\n=== ГОТОВО: {run_dir} ===\n"
        f"Всего лигандов: {len(all_ligands)}, задокировано успешно: {n_ok}, не удалось: {n_fail}\n"
        f"Результаты: {results_path}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gene", default="PIK3CA")
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--n_per_seed", type=int, required=True)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--exhaustiveness", type=int, default=8)
    parser.add_argument("--dock_timeout", type=int, default=120)
    parser.add_argument("--timesteps", type=int, default=None, help="Число шагов диффузии (шаг 10 промта: полное/половина/четверть)")
    parser.add_argument("--gen_timeout_sec", type=int, default=7200, help="Таймаут генерации ОДНОГО сида (защита от зависания CUDA/torch)")
    parser.add_argument("--batch_size", type=int, default=None, help="Батч генерации DiffSBDD (дефолт generate_smiles_seeded.py: 4, консервативно под 4 ГБ VRAM). Больше — быстрее, но риск OOM; сид с OOM теперь просто пропускается, не роняет весь прогон")
    args = parser.parse_args()

    try:
        run_pipeline(
            args.gene, args.seeds, args.n_per_seed, args.run_dir,
            exhaustiveness=args.exhaustiveness, dock_timeout=args.dock_timeout,
            timesteps=args.timesteps, gen_timeout_sec=args.gen_timeout_sec,
            batch_size=args.batch_size,
        )
    except PipelineAborted as e:
        print(f"ОСТАНОВЛЕНО: {e}\nПрогресс сохранён (JSONL/state.json), можно продолжить обычным перезапуском.")
        sys.exit(0)
    except (RuntimeError, DockingError, GeneTargetError) as e:
        print(f"ОШИБКА: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
