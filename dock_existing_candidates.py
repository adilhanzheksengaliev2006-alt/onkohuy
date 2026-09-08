"""
dock_existing_candidates.py — докинг через AutoDock Vina
=========================================================================
ПЕРЕДЕЛАНО под динамический поиск структуры по GENE_NAME (было
захардкожено под PIK3CA/4JPS). Структура рецептора теперь ищется
автоматически через gene_target_utils.resolve_gene_to_docking_target().

Стек:
  - тот же AutoDock Vina, но т.к. python-биндинги `pip install vina`
    не собираются на Windows без Boost/SWIG, используется официальный
    бинарник vina.exe (tools/vina.exe, AutoDock Vina 1.2.7) через
    subprocess — тот же движок, тот же алгоритм, тот же результат.
  - meeko (mk_prepare_receptor.exe / mk_prepare_ligand.exe) для
    подготовки PDBQT из PDB/mmCIF рецептора и SMILES-лиганда
  - RDKit для генерации 3D-конформера лиганда из SMILES

Функции prepare_receptor / dock_smiles переиспользуются в
module_generative/iterative_finetune_loop.py.
"""

import os
import re
import subprocess
import sys
import tempfile
import time

import pandas as pd

from gene_target_utils import GeneTargetError, resolve_gene_to_docking_target, get_chembl_new_client

GENE_NAME = "PIK3CA"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VINA_EXE = os.path.join(BASE_DIR, "tools", "vina.exe")
MK_PREPARE_RECEPTOR = os.path.join(
    os.path.dirname(sys.executable), "Scripts", "mk_prepare_receptor.exe"
)
MK_PREPARE_LIGAND = os.path.join(
    os.path.dirname(sys.executable), "Scripts", "mk_prepare_ligand.exe"
)


class DockingError(Exception):
    """Понятная ошибка на любом шаге подготовки/запуска докинга."""


def _run(cmd, timeout=120):
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout
        )
    except FileNotFoundError as e:
        raise DockingError(f"Исполняемый файл не найден: {cmd[0]} ({e})")
    except subprocess.TimeoutExpired:
        raise DockingError(f"Команда превысила таймаут {timeout}с: {' '.join(cmd)}")
    return result


def convert_to_legacy_pdb(structure_path: str) -> str:
    """meeko's --read_pdb требует классический .pdb. Если скачан
    mmCIF (частый случай для новых структур), конвертирует через gemmi."""
    if structure_path.lower().endswith(".pdb"):
        return structure_path

    try:
        import gemmi
    except Exception as e:
        raise DockingError(f"Не удалось импортировать gemmi для конвертации CIF->PDB: {e}")

    out_path = structure_path.rsplit(".", 1)[0] + "_converted.pdb"
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return out_path
    try:
        st = gemmi.read_structure(structure_path)
        st.setup_entities()
        st.write_pdb(out_path)
    except Exception as e:
        raise DockingError(f"Ошибка конвертации {structure_path} в legacy PDB: {e}")
    return out_path


def strip_ligands_and_waters(pdb_path: str) -> str:
    """Убирает со-кристаллизованный лиганд/воду/ионы кристаллизации из PDB
    ПЕРЕД подготовкой рецептора.

    ВАЖНО (найдено при разборе аномально слабого контрольного докинга
    алпелисиба на 4JPS — скор стабильно ~-6 ккал/моль вместо ожидаемых
    -7...-10, RMSD к кристаллической позе 2.1 A): mk_prepare_receptor
    вызывается с --read_pdb (классический ридер), который, в отличие от
    ProDy-ридера (-i), НЕ фильтрует гетероатомы — старый комментарий в
    prepare_receptor() про автоматическое исключение лиганда был неверен.
    В результате "жёсткий" рецептор физически содержал исходный
    со-кристаллизованный лиганд (в т.ч. на 4JPS — саму молекулу
    алпелисиба, resname 1LT) плюс воду/ионы, и докинг НОВОЙ копии той же
    молекулы шёл в карман, где старая копия уже физически сидела.
    Переключиться на -i (ProDy) нельзя: в этом окружении ProDy сломан
    несовместимостью с текущим numpy (`cannot import name 'alltrue'`).
    Поэтому чистим сами через gemmi (уже используется в проекте для
    CIF->PDB) — Structure.remove_ligands_and_waters() убирает и
    со-кристаллизованный лиганд, и воду, и мелкие ионы/добавки
    кристаллизации, оставляя только полимерные цепи."""
    import gemmi

    out_path = pdb_path.rsplit(".", 1)[0] + "_clean.pdb"
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return out_path
    try:
        st = gemmi.read_structure(pdb_path)
        st.setup_entities()
        st.remove_ligands_and_waters()
        st.remove_empty_chains()
        # Обнаружено на HDAC2 (4LXZ): mk_prepare_receptor падает намертво
        # ("Requested altlocs not found for: ['B:40']"), если у остатка
        # есть альтернативные конформации (altloc), но конкретно у ЭТОГО
        # остатка нет варианта 'A' (--default_altloc A, жёстко заданный
        # ниже, не помогает). Это обычный артефакт кристаллографии в
        # структурах повыше разрешением, не специфично для HDAC2 -
        # убираем alt-конформации на уровне структуры (оставляя основную/
        # первую), а не патчим altloc-флаг под каждую мишень отдельно.
        st.remove_alternative_conformations()
        st.write_pdb(out_path)
    except Exception as e:
        raise DockingError(f"Ошибка очистки {pdb_path} от лиганда/воды через gemmi: {e}")
    return out_path


def prepare_receptor(structure_path: str, box_center, box_size, out_basename: str) -> str:
    """Готовит receptor PDBQT из PDB/mmCIF файла рецептора. Со-кристаллизованный
    лиганд и вода явно вычищаются заранее (см. strip_ligands_and_waters) —
    mk_prepare_receptor с --read_pdb сам этого не делает."""
    pdb_path = convert_to_legacy_pdb(structure_path)
    pdb_path = strip_ligands_and_waters(pdb_path)
    receptor_pdbqt = out_basename + ".pdbqt"

    if os.path.exists(receptor_pdbqt) and os.path.getsize(receptor_pdbqt) > 0:
        return receptor_pdbqt

    cmd = [
        MK_PREPARE_RECEPTOR,
        "--read_pdb", pdb_path,
        "-o", out_basename,
        "-p",
        "-a",
        "--default_altloc", "A",
        "--box_center", str(box_center[0]), str(box_center[1]), str(box_center[2]),
        "--box_size", str(box_size[0]), str(box_size[1]), str(box_size[2]),
    ]
    result = _run(cmd, timeout=180)
    if not os.path.exists(receptor_pdbqt):
        raise DockingError(
            f"mk_prepare_receptor не создал {receptor_pdbqt}.\n"
            f"stdout: {result.stdout[-1500:]}\nstderr: {result.stderr[-1500:]}"
        )
    return receptor_pdbqt


def prepare_ligand_pdbqt(smiles: str, out_path: str) -> bool:
    """SMILES -> 3D-конформер (RDKit) -> PDBQT (meeko). Возвращает
    False (не бросает исключение), если конкретная молекула не
    embed-ится в 3D или не парсится meeko — это ожидаемо для части
    генеративных SMILES и не должно останавливать весь цикл."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    mol = Chem.AddHs(mol)
    try:
        embed_ok = AllChem.EmbedMolecule(mol, randomSeed=42, useRandomCoords=True)
        if embed_ok != 0:
            return False
        AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
    except Exception:
        return False

    sdf_path = out_path.rsplit(".", 1)[0] + ".sdf"
    try:
        writer = Chem.SDWriter(sdf_path)
        writer.write(mol)
        writer.close()
    except Exception:
        return False

    cmd = [MK_PREPARE_LIGAND, "-i", sdf_path, "-o", out_path]
    # _run() бросает DockingError на таймауте/ненайденном экзешнике - в
    # нарушение контракта, задокументированного выше ("не бросает
    # исключение"). Раньше это не проявлялось на одиночном последовательном
    # докинге (phase_dock оборачивает КАЖДЫЙ лиганд в отдельный процесс
    # через multiprocessing.Pool, где падение воркера по-другому
    # прокидывается наверх), но в controls/test09-10/test08 (тоже
    # Pool.imap_unordered) необработанное исключение здесь убивало ВЕСЬ
    # пул из-за ОДНОЙ зависшей на mk_prepare_ligand молекулы - поймано
    # в бою на реальном прогоне. Ловим здесь, у источника, а не в каждом
    # вызывающем месте по отдельности.
    try:
        _run(cmd, timeout=60)
    except DockingError:
        return False
    return os.path.exists(out_path) and os.path.getsize(out_path) > 0


def run_vina(receptor_pdbqt, ligand_pdbqt, box_center, box_size, out_pdbqt, exhaustiveness=8, timeout=300, cpu=None):
    """Запускает vina.exe, возвращает лучшую (наиболее отрицательную)
    аффинность в ккал/моль, либо None, если докинг не удался (включая
    превышение timeout — некоторые сгенерированные молекулы дают
    вырожденные/сложные лиганды, на которых поиск идёт аномально
    долго; таймаут просто исключает такую молекулу из цикла, а не
    останавливает весь пайплайн).

    cpu: без --cpu Vina сам забирает ВСЕ доступные потоки CPU на один
    процесс. Это нормально при строго последовательном докинге (как
    сейчас в основном пайплайне), но фатально при параллельном докинге
    через несколько процессов (benchmark_parallel_docking.py, будущий
    параллельный Тест A/B) — N процессов, каждый пытающийся забрать
    ВСЕ потоки, дают oversubscription в разы больше числа ядер.
    Обнаружено на реальном прогоне: эффективность параллелизма падала
    с 65% (2 воркера) до 27% (6 воркеров), и на 4/6 воркерах стали
    появляться докинги, упирающиеся ровно в 120с таймаут - оба
    признака классического oversubscription, не проблемы самого
    параллелизма. При параллельном запуске сюда нужно передавать
    cpu = (доступные_потоки - резерв) // число_воркеров."""
    cmd = [
        VINA_EXE,
        "--receptor", receptor_pdbqt,
        "--ligand", ligand_pdbqt,
        "--center_x", str(box_center[0]),
        "--center_y", str(box_center[1]),
        "--center_z", str(box_center[2]),
        "--size_x", str(box_size[0]),
        "--size_y", str(box_size[1]),
        "--size_z", str(box_size[2]),
        "--out", out_pdbqt,
        "--exhaustiveness", str(exhaustiveness),
    ]
    if cpu is not None:
        cmd += ["--cpu", str(cpu)]
    try:
        result = _run(cmd, timeout=timeout)
    except DockingError:
        return None
    combined_output = result.stdout + result.stderr
    matches = re.findall(r"^\s*\d+\s+(-?\d+\.\d+)", combined_output, re.MULTILINE)
    if not matches:
        return None
    return float(matches[0])


def dock_smiles(smiles: str, receptor_pdbqt: str, box_center, box_size, workdir: str, tag: str = "lig", exhaustiveness: int = 8, timeout: int = 300, cpu=None):
    """Полный докинг одной молекулы. Возвращает лучшую аффинность
    (float, ккал/моль) или None, если молекулу не удалось
    подготовить/задокировать (не считается фатальной ошибкой)."""
    ligand_pdbqt = os.path.join(workdir, f"{tag}.pdbqt")
    out_pdbqt = os.path.join(workdir, f"{tag}_out.pdbqt")
    try:
        ok = prepare_ligand_pdbqt(smiles, ligand_pdbqt)
        if not ok:
            return None
        return run_vina(receptor_pdbqt, ligand_pdbqt, box_center, box_size, out_pdbqt, exhaustiveness=exhaustiveness, timeout=timeout, cpu=cpu)
    except DockingError:
        return None
    finally:
        for f in (ligand_pdbqt, out_pdbqt, ligand_pdbqt.replace(".pdbqt", ".sdf")):
            try:
                os.remove(f)
            except OSError:
                pass


_DOCK_WORKER = os.path.join(BASE_DIR, "_dock_worker.py")
GNINA_WRAPPER_WSL = "$HOME/gnina/gnina_run.sh"


def _win_path_to_wsl(path):
    drive, rest = os.path.splitdrive(os.path.abspath(path))
    return f"/mnt/{drive[0].lower()}{rest.replace(chr(92), '/')}"


GNINA_FAIL_LOG = os.path.join(BASE_DIR, "gnina_failures.log")


def _log_gnina_failure(msg: str):
    """Печать из воркеров multiprocessing.Pool не всегда долетает до
    терминала родителя (обнаружено: на реальном прогоне PIK3CA - 1331
    успешных Vina-докингов, но только 5 успешных gnina-рескорингов,
    и НИ ОДНОГО отладочного сообщения не было видно, хотя код его
    печатал). Пишем в файл, чтобы при следующем сбое было что разбирать."""
    try:
        with open(GNINA_FAIL_LOG, "a", encoding="utf-8") as f:
            f.write(msg + "\n")
    except OSError:
        pass


def gnina_rescore_pose(receptor_pdbqt: str, docked_ligand_pdbqt: str, timeout: int = 60, attempts: int = 2):
    """CNN-рескоринг УЖЕ ЗАДОКИРОВАННОЙ позы через gnina (работает только
    в WSL - см. установку CUDA-библиотек в ~/cuda_libs). Возвращает dict
    {cnn_score, cnn_affinity, cnn_variance, vina_affinity_gnina} или None
    при любой ошибке/недоступности gnina - рескоринг вспомогательный,
    его отказ не должен ронять основной Vina-докинг.

    При реальном прогоне на 1581 лигандах (6 параллельных воркеров) из
    1331 успешных Vina-докингов gnina сработала только 5 раз - на малых
    тестах (12, 31 лиганд, тоже 6 воркеров) была стабильно 100%. Похоже
    на деградацию под устойчивой длительной нагрузкой (вероятный
    подозреваемый: .wslconfig ограничивал WSL 3ГБ памяти - CUDA-библиотеки
    gnina тяжёлые, cudnn_engines_precompiled.so один ~576МБ, 6 параллельных
    CUDA-контекстов могли не влезать - увеличено до 4ГБ), а не разовый баг
    конкурентности. retry + логирование в файл - защита на случай, если
    первопричина не только в памяти."""
    # docked_ligand_pdbqt - это ПОЛНЫЙ вывод Vina с несколькими MODEL
    # (несколько поз). Три отдельные проблемы обнаружены методом проб
    # (каждая давала одну и ту же невыразительную ошибку gnina "Parse
    # error ... Unknown or inappropriate tag", разбирались по очереди):
    #  1. Нужна только ПЕРВАЯ (лучшая по скору) поза, не все сразу.
    #  2. Этот конкретный vina.exe (Windows-сборка 1.2.7) дописывает
    #     ~55 нулевых байт в конец КАЖДОЙ позы (похоже на баг
    #     буферизации при записи) - meeko это молча проглатывает
    #     (см. run_redock.py), а строгий парсер gnina - нет.
    #  3. gnina в режиме --score_only для -l ожидает "голый" лигандный
    #     PDBQT (как из meeko/mk_prepare_ligand - ROOT...TORSDOF без
    #     обёртки), А НЕ формат результата докинга Vina с MODEL/ENDMDL -
    #     обёртку и вообще любые пустые строки нужно убрать целиком,
    #     не только распаковать одну позу.
    first_model_path = docked_ligand_pdbqt.replace("_out.pdbqt", "_out_first.pdbqt")
    try:
        with open(docked_ligand_pdbqt, encoding="utf-8", errors="ignore") as f:
            content = f.read().replace("\x00", "")
        first_pose_lines = content.split("ENDMDL")[0].splitlines()
        body_lines = [l for l in first_pose_lines if l.strip() and not l.strip().startswith("MODEL")]
        with open(first_model_path, "w", encoding="utf-8") as f:
            f.write("\n".join(body_lines) + "\n")
    except Exception as e:
        _log_gnina_failure(f"pid={os.getpid()} не смог подготовить first_model_path: {type(e).__name__}: {e}")
        return None

    receptor_wsl = _win_path_to_wsl(receptor_pdbqt)
    ligand_wsl = _win_path_to_wsl(first_model_path)
    cmd = ["wsl", "-e", "bash", "-c",
           f"{GNINA_WRAPPER_WSL} -r '{receptor_wsl}' -l '{ligand_wsl}' --score_only"]

    result = None
    for attempt in range(1, attempts + 1):
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        except Exception as e:
            _log_gnina_failure(f"pid={os.getpid()} попытка {attempt}/{attempts} исключение: {type(e).__name__}: {e}")
            result = None
        if result is not None and result.returncode == 0:
            break
        if result is not None:
            _log_gnina_failure(f"pid={os.getpid()} попытка {attempt}/{attempts} rc={result.returncode} "
                                f"stderr_tail={result.stderr[-300:]!r}")
        if attempt < attempts:
            time.sleep(2)

    try:
        os.remove(first_model_path)
    except OSError:
        pass

    if result is None or result.returncode != 0:
        return None
    parsed = {}
    for line in result.stdout.splitlines():
        if line.startswith("Affinity:"):
            parsed["vina_affinity_gnina"] = float(line.split()[1])
        elif line.startswith("CNNscore:"):
            parsed["cnn_score"] = float(line.split()[1])
        elif line.startswith("CNNaffinity:"):
            parsed["cnn_affinity"] = float(line.split()[1])
        elif line.startswith("CNNvariance:"):
            parsed["cnn_variance"] = float(line.split()[1])
    return parsed or None


def dock_smiles_isolated(smiles: str, receptor_pdbqt: str, box_center, box_size, workdir: str, tag: str = "lig", exhaustiveness: int = 8, timeout: int = 60, cpu=None, gnina_rescore: bool = False):
    """Как dock_smiles(), но весь докинг ОДНОЙ молекулы (RDKit-embedding
    + подготовка лиганда + Vina) выполняется в отдельном процессе с
    жёстким таймаутом на уровне ОС. Нужно для генеративного цикла:
    небольшая недообученная MolGPT изредка выдаёт синтаксически
    валидные, но патологические SMILES (сильно сшитые кольцевые
    системы), на которых RDKit-embedding или сам Vina могут зависать
    на много минут — обычный subprocess-таймаут внутри run_vina от
    этого не защищает, т.к. зависание может быть ДО запуска vina.exe.

    gnina_rescore=True: после успешного Vina-докинга, ДО удаления файла
    позы (см. ниже - раньше поза удалялась сразу же, gnina физически не
    успела бы её увидеть), рескорит эту же позу через gnina (CNN-скор).
    Возвращает (vina_score, gnina_dict_или_None) вместо просто float.
    Возвращает лучшую аффинность (float) или None (или (None, None) при
    gnina_rescore=True)."""
    out_pdbqt = os.path.join(workdir, f"{tag}_out.pdbqt")
    cmd = [
        sys.executable, _DOCK_WORKER,
        smiles, receptor_pdbqt,
        str(box_center[0]), str(box_center[1]), str(box_center[2]),
        str(box_size[0]), str(box_size[1]), str(box_size[2]),
        out_pdbqt, str(exhaustiveness),
        str(cpu) if cpu is not None else "0",
    ]

    # ВАЖНО (Windows): subprocess.run/communicate(timeout=...) читает вывод
    # потомка через pipe, а pipe закрывается (EOF) только когда ВСЕ
    # процессы, унаследовавшие его write-хендл, завершились — включая
    # внуков (vina.exe/mk_prepare_ligand.exe), которых порождает воркер.
    # Если воркер завис (недообученная MolGPT изредка выдаёт синтаксически
    # валидные, но патологические SMILES, на которых RDKit-embedding или
    # сам Vina зависают), communicate() будет ждать закрытия pipe
    # НАВСЕГДА — даже proc.kill()/taskkill дочернего процесса это не
    # разблокирует, если внук всё ещё жив. Поэтому вывод воркера пишется
    # в обычный файл (а не pipe), а ожидание идёт через proc.wait(),
    # который завязан на состояние процесса в ОС, а не на pipe, и потому
    # не подвержен этой блокировке.
    def _cleanup_pose_files():
        for f in (out_pdbqt, out_pdbqt.replace("_out.pdbqt", ".pdbqt"), out_pdbqt.replace("_out.pdbqt", ".sdf")):
            try:
                os.remove(f)
            except OSError:
                pass

    fail_result = (None, None) if gnina_rescore else None

    log_path = os.path.join(workdir, f"{tag}_worker.log")
    with open(log_path, "w", encoding="utf-8") as log_fh:
        proc = subprocess.Popen(cmd, stdout=log_fh, stderr=subprocess.STDOUT)
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            # ВАЖНО: убиваем только ДЕРЕВО ПРОЦЕССОВ этого конкретного воркера
            # (по PID, /T добивает и внуков типа vina.exe/mk_prepare_ligand.exe),
            # а НЕ по имени образа глобально. При параллельном докинге
            # (несколько worker-процессов через multiprocessing.Pool) taskkill
            # по /IM убил бы vina.exe ВСЕХ воркеров разом, включая те, что
            # нормально считают чужие молекулы - это раньше было безопасно
            # только потому, что докинг был строго последовательным (один
            # vina.exe в моменте), но перестало быть безопасным с введением
            # параллельного докинга (benchmark_parallel_docking.py и далее
            # параллельный Тест A/B). Убиваем по PID ДО proc.kill(), пока
            # процесс ещё точно жив - иначе есть маленькое окно, где PID
            # успевает освободиться/переиспользоваться между kill() и taskkill.
            try:
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True, timeout=10)
            except Exception:
                pass
            try:
                proc.wait(timeout=10)
            except Exception:
                pass
            _cleanup_pose_files()
            return fail_result
        except Exception:
            _cleanup_pose_files()
            return fail_result

    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            output = f.read()
    except OSError:
        _cleanup_pose_files()
        return fail_result
    finally:
        try:
            os.remove(log_path)
        except OSError:
            pass

    score = None
    for line in output.splitlines():
        if line.startswith("SCORE="):
            try:
                score = float(line[len("SCORE="):])
            except ValueError:
                score = None
            break

    # gnina нужна СЫРАЯ поза (out_pdbqt) - рескорим ДО удаления, раньше
    # этот файл удалялся сразу в finally выше, gnina физически не могла
    # его увидеть (обнаружено при подключении gnina к Тесту A).
    gnina_result = None
    if gnina_rescore and score is not None and os.path.exists(out_pdbqt):
        gnina_result = gnina_rescore_pose(receptor_pdbqt, out_pdbqt)

    _cleanup_pose_files()
    return (score, gnina_result) if gnina_rescore else score


def dock_existing_candidates(gene_name: str) -> pd.DataFrame:
    print(f"=== Докинг известных/одобренных кандидатов для гена {gene_name} ===")

    target = resolve_gene_to_docking_target(gene_name)
    if target is None:
        print(f"Докинг для {gene_name} пропущен — подходящая структура не найдена.")
        return pd.DataFrame()

    from find_repurposing_candidates import find_repurposing_candidates

    candidates_df = find_repurposing_candidates(gene_name)
    if candidates_df.empty:
        print(f"Нет одобренных кандидатов для докинга по гену {gene_name}.")
        return pd.DataFrame()

    new_client = get_chembl_new_client()

    molecule = new_client.molecule

    receptor_basename = os.path.join(
        os.path.dirname(target["pdb_path"]), f"{target['pdb_id']}_receptor"
    )
    receptor_pdbqt = prepare_receptor(
        target["pdb_path"], target["box_center"], target["box_size"], receptor_basename
    )
    print(f"Рецептор подготовлен: {receptor_pdbqt}")

    rows = []
    with tempfile.TemporaryDirectory() as workdir:
        for i, row in candidates_df.iterrows():
            mol_id = row["molecule_chembl_id"]
            try:
                rec = molecule.get(mol_id)
                smiles = rec.get("molecule_structures", {}).get("canonical_smiles")
            except Exception as e:
                print(f"  [{row['pref_name']}] пропущен — ошибка получения SMILES: {e}")
                continue
            if not smiles:
                print(f"  [{row['pref_name']}] пропущен — нет SMILES в ChEMBL")
                continue

            affinity = dock_smiles(
                smiles, receptor_pdbqt, target["box_center"], target["box_size"], workdir, tag=f"cand{i}"
            )
            if affinity is None:
                print(f"  [{row['pref_name']}] докинг не удался")
            else:
                print(f"  [{row['pref_name']}] docking score = {affinity:.2f} ккал/моль")
            rows.append(
                {
                    "molecule_chembl_id": mol_id,
                    "pref_name": row["pref_name"],
                    "smiles": smiles,
                    "docking_score_kcal_mol": affinity,
                    "pdb_id": target["pdb_id"],
                }
            )

    return pd.DataFrame(rows)


def main():
    gene = sys.argv[1] if len(sys.argv) > 1 else GENE_NAME
    try:
        df = dock_existing_candidates(gene)
    except (GeneTargetError, DockingError) as e:
        print(f"ОШИБКА: {e}")
        sys.exit(1)

    if df.empty:
        print("Итог: результатов докинга нет.")
        return

    out_path = f"docking_results_{gene}.csv"
    df.to_csv(out_path, index=False)
    print(f"\nСохранено: {out_path}")


if __name__ == "__main__":
    main()
