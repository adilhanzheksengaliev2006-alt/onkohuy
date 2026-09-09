"""
benchmark_parallel_docking.py — шаг 5 отчёта: реальный коэффициент
ускорения докинга Vina при параллелизации по процессам (1/2/4/6
воркеров) на пуле РЕАЛЬНЫХ сгенерированных лигандов (SMILES взяты из
фактических прошлых прогонов sbdd_pipeline.py: runs/batch128_test,
runs/overnight_check, runs/check1 — а не синтетика, чтобы отражать
настоящее распределение сложности молекул).

box_center/box_size берутся ЛОКАЛЬНО из structures/4JPS.pdb через
find_ligand_center() — без единого сетевого запроса, т.к. ChEMBL
сейчас недоступен, а resolve_gene_to_docking_target() иначе бы упал
на первом же вызове find_chembl_target() ещё до проверки
confirmed_structures.json.

ВАЖНО: это тяжёлый прогон (по умолчанию 4 фазы x 100 лигандов =>
по грубой прикидке от ~10 мин до ~40+ мин суммарно, зависит от
реального ускорения, которое мы и измеряем). Запускать ТОЛЬКО из
консоли с закрытым VS Code — как договорено, VS Code съедает
1-1.5ГБ из 8ГБ RAM и исказит и абсолютные числа, и сам замер
параллелизма (меньше свободных ядер/памяти под воркеры).

Использование:
    python benchmark_parallel_docking.py [N_ЛИГАНДОВ] [W1,W2,W3,...]

    N_ЛИГАНДОВ  - лигандов на фазу (по умолчанию 100)
    W1,W2,...   - число воркеров на фазу через запятую (по умолчанию 1,2,4,6)

Можно прервать между фазами (Ctrl+C или control.json {"abort": true} в
корне проекта) - результаты уже завершённых фаз сохранены в
runs/benchmark_parallel_docking_results.json после каждой фазы.
"""
import json
import multiprocessing
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dock_existing_candidates import dock_smiles_isolated  # noqa: E402
from gene_target_utils import find_ligand_center  # noqa: E402

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RECEPTOR_PDBQT = os.path.join(BASE_DIR, "structures", "4JPS_receptor.pdbqt")
LIGAND_STRUCTURE_PDB = os.path.join(BASE_DIR, "structures", "4JPS.pdb")
RESULTS_JSONL_SOURCES = [
    os.path.join(BASE_DIR, "runs", "batch128_test", "results.jsonl"),
    os.path.join(BASE_DIR, "runs", "overnight_check", "results.jsonl"),
    os.path.join(BASE_DIR, "runs", "check1", "results.jsonl"),
]
WORKDIR = os.path.join(BASE_DIR, "runs", "benchmark_parallel_docking_tmp")
OUT_PATH = os.path.join(BASE_DIR, "runs", "benchmark_parallel_docking_results.json")
CONTROL_PATH = os.path.join(BASE_DIR, "control.json")


def load_ligand_pool(n_target):
    """Реальные SMILES из прошлых успешных прогонов пайплайна, не
    синтетика - чтобы бенчмарк отражал настоящее распределение
    сложности/размера молекул, которое встретится в реальном Тесте A/B."""
    smiles_list = []
    seen = set()
    for path in RESULTS_JSONL_SOURCES:
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                s = rec.get("smiles")
                if s and s not in seen:
                    seen.add(s)
                    smiles_list.append(s)
    if not smiles_list:
        raise RuntimeError(
            "Не нашлось ни одного SMILES в прошлых runs/*/results.jsonl - "
            "нечем бенчмаркать. Нужен хотя бы один успешный прогон sbdd_pipeline.py."
        )
    if len(smiles_list) < n_target:
        print(f"[warn] в пуле только {len(smiles_list)} уникальных SMILES из прошлых прогонов, "
              f"запрошено {n_target} - зацикливаю пул с повтором. Для замера СКОРОСТИ "
              f"повтор SMILES не проблема (та же сложность молекулы -> та же нагрузка на "
              f"Vina), это НЕ Тест A и не требует уникальности.")
        reps = (n_target // len(smiles_list)) + 1
        smiles_list = smiles_list * reps
    return smiles_list[:n_target]


def check_control():
    if os.path.exists(CONTROL_PATH):
        try:
            with open(CONTROL_PATH, encoding="utf-8") as f:
                c = json.load(f)
            if c.get("abort"):
                print("[control] control.json просит остановиться - прерываю бенчмарк между фазами.")
                return True
        except Exception:
            pass
    return False


def auto_n_workers(budget_fraction=0.8, threads_per_worker=2):
    """Число воркеров, подобранное под ЭТУ КОНКРЕТНУЮ машину - не
    захардкожено (было =6, верно только для машины разработки на 16
    логических потоках; на машине с другим числом ядер либо
    недогружало бы CPU, либо превышало бы заданный бюджет ресурсов).

    budget_fraction: не занимать больше этой доли логических ядер -
    оставляет запас, чтобы машиной можно было пользоваться параллельно
    с многодневным фоновым докингом."""
    total_threads = multiprocessing.cpu_count()
    usable = max(1, int(total_threads * budget_fraction))
    n_workers = max(1, usable // threads_per_worker)
    print(f"[auto_n_workers] {total_threads} логических ядер, бюджет {budget_fraction*100:.0f}% "
          f"= {usable} потоков, {threads_per_worker} потока/воркер -> {n_workers} воркеров")
    return n_workers


def cpu_per_worker(n_workers, reserved=2, budget_fraction=None):
    """Vina без --cpu забирает ВСЕ доступные потоки на процесс - при
    n_workers>1 это даёт oversubscription (N процессов x все потоки),
    что на практике и обнаружилось: эффективность падала с 65% (2
    воркера) до 27% (6 воркеров), и часть докингов упиралась ровно в
    таймаут. Для n_workers=1 сознательно возвращаем None (не передаём
    --cpu) - это поведение сегодняшнего последовательного пайплайна,
    и baseline фазы должен ему соответствовать 1:1 для честного
    сравнения, а не быть искусственно замедленным.

    budget_fraction, если задан, ЗАМЕНЯЕТ фиксированный reserved на
    процентный бюджет (напр. 0.8 = не больше 80% ядер суммарно на все
    воркеры) - reserved=2 был откалиброван под машину разработки (16
    логических потоков, 2 из них в резерв = 87.5% используется), на
    машине с другим числом ядер фиксированное число резерва даёт
    другой процент - для переносимости на новое железо нужен процент,
    не абсолютное число."""
    if n_workers <= 1:
        return None
    if budget_fraction is not None:
        usable = max(int(multiprocessing.cpu_count() * budget_fraction), n_workers)
    else:
        usable = max(multiprocessing.cpu_count() - reserved, n_workers)
    return max(1, usable // n_workers)


def _dock_one(args):
    idx, smiles, box_center, box_size, tag, cpu = args
    t0 = time.time()
    score = dock_smiles_isolated(
        smiles, RECEPTOR_PDBQT, box_center, box_size, WORKDIR, tag=tag,
        exhaustiveness=8, timeout=120, cpu=cpu,
    )
    return {"idx": idx, "score": score, "dock_sec": time.time() - t0}


def run_phase(smiles_list, box_center, box_size, n_workers):
    os.makedirs(WORKDIR, exist_ok=True)
    cpu = cpu_per_worker(n_workers)
    tasks = [
        (i, s, box_center, box_size, f"bench_w{n_workers}_{i}", cpu)
        for i, s in enumerate(smiles_list)
    ]
    print(f"  (--cpu на процесс Vina: {cpu if cpu else 'не ограничено (как в текущем пайплайне)'})")
    t_start = time.time()
    results = []
    if n_workers == 1:
        for task in tasks:
            results.append(_dock_one(task))
    else:
        with multiprocessing.Pool(processes=n_workers) as pool:
            for res in pool.imap_unordered(_dock_one, tasks):
                results.append(res)
    wall_sec = time.time() - t_start
    n_ok = sum(1 for r in results if r["score"] is not None)
    return {
        "n_workers": n_workers,
        "n_ligands": len(results),
        "n_ok": n_ok,
        "n_fail": len(results) - n_ok,
        "wall_sec": round(wall_sec, 2),
        "sec_per_ligand": round(wall_sec / len(results), 3) if results else None,
        "per_ligand_dock_sec": [r["dock_sec"] for r in results],
    }


def main():
    n_ligands = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    worker_counts = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [1, 2, 4, 6]

    if not os.path.exists(RECEPTOR_PDBQT):
        print(f"ОШИБКА: нет {RECEPTOR_PDBQT}"); sys.exit(1)

    print(f"[setup] Определяю box_center/box_size из локального {LIGAND_STRUCTURE_PDB} (без сети)...")
    ligand_info = find_ligand_center(LIGAND_STRUCTURE_PDB)
    if ligand_info is None:
        print("ОШИБКА: не удалось определить центр лиганда в 4JPS.pdb")
        sys.exit(1)
    box_center = ligand_info["center"]
    box_size = ligand_info["box_size"]
    print(f"[setup] box_center={box_center}, box_size={box_size} (лиганд {ligand_info['resname']}, "
          f"{ligand_info['n_atoms']} тяжёлых атомов)")

    smiles_pool = load_ligand_pool(n_ligands)
    print(f"[setup] пул лигандов: {len(smiles_pool)} (реальные SMILES из прошлых прогонов)")
    print(f"[setup] фазы по числу воркеров: {worker_counts}, по {len(smiles_pool)} лигандов каждая")
    print(f"[setup] CPU-ядер видно системе: {multiprocessing.cpu_count()}")

    phase_results = []
    for nw in worker_counts:
        if check_control():
            break
        print(f"\n=== ФАЗА: {nw} воркер(ов), {len(smiles_pool)} лигандов ===")
        phase = run_phase(smiles_pool, box_center, box_size, nw)
        phase_results.append(phase)
        print(f"  wall={phase['wall_sec']:.1f}с, ok={phase['n_ok']}/{phase['n_ligands']}, "
              f"{phase['sec_per_ligand']:.2f}с/лиганд (среднее по стенным часам, включая накладные "
              f"расходы планировщика ОС/Pool)")
        os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
        with open(OUT_PATH, "w", encoding="utf-8") as f:
            json.dump(
                {"box_center": box_center, "box_size": box_size,
                 "cpu_count": multiprocessing.cpu_count(), "phases": phase_results},
                f, indent=2,
            )

    print(f"\n\n=== ИТОГ ===")
    baseline = next((p["wall_sec"] for p in phase_results if p["n_workers"] == 1), None)
    if baseline:
        for p in phase_results:
            speedup = baseline / p["wall_sec"] if p["wall_sec"] else None
            efficiency = (speedup / p["n_workers"] * 100) if speedup else None
            print(f"  {p['n_workers']} воркер(ов): {p['wall_sec']:.1f}с суммарно, "
                  f"ускорение x{speedup:.2f}, эффективность {efficiency:.0f}%"
                  if speedup else f"  {p['n_workers']}: н/д")
    else:
        print("  Нет фазы с 1 воркером (базовая линия) - коэффициент ускорения не посчитан, "
              "только абсолютные времена по фазам в результатах.")

    print(f"\nПолные результаты (в т.ч. время на каждый отдельный лиганд, для анализа разброса "
          f"и возможного троттлинга внутри фазы): {OUT_PATH}")


if __name__ == "__main__":
    main()
