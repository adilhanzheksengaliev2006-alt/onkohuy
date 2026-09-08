"""
benchmark_throttling.py — шаг 5 отчёта: тест на троттлинг под
длительной (по умолчанию 30 мин) непрерывной нагрузкой докингом.

Ноутбук под многочасовой нагрузкой (реальный Тест A - несколько часов
докинга подряд) может просесть по частоте CPU из-за нагрева - этот
тест гоняет докинг непрерывно N минут с фиксированным числом воркеров
и смотрит, растёт ли среднее время на лиганд от начала прогона к концу.

Использование:
    python benchmark_throttling.py [МИНУТ] [ЧИСЛО_ВОРКЕРОВ]

    МИНУТ            - длительность теста (по умолчанию 30)
    ЧИСЛО_ВОРКЕРОВ   - сколько параллельных Vina-процессов (по умолчанию 6;
                       имеет смысл сначала прогнать benchmark_parallel_docking.py
                       и подставить сюда то число воркеров, которое там
                       дало лучшую эффективность)

Лиганды берутся из того же пула реальных SMILES, что и в
benchmark_parallel_docking.py (runs/*/results.jsonl), с повтором по
кругу, пока не истечёт время.

ВАЖНО: тяжёлый прогон ровно на заданное число минут (+ несколько минут
на текущую партию до завершения). Запускать ТОЛЬКО из консоли с
закрытым VS Code.
"""
import json
import multiprocessing
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dock_existing_candidates import dock_smiles_isolated  # noqa: E402
from gene_target_utils import find_ligand_center  # noqa: E402
from benchmark_parallel_docking import load_ligand_pool, cpu_per_worker  # noqa: E402

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RECEPTOR_PDBQT = os.path.join(BASE_DIR, "structures", "4JPS_receptor.pdbqt")
LIGAND_STRUCTURE_PDB = os.path.join(BASE_DIR, "structures", "4JPS.pdb")
WORKDIR = os.path.join(BASE_DIR, "runs", "benchmark_throttling_tmp")
OUT_PATH = os.path.join(BASE_DIR, "runs", "benchmark_throttling_results.json")
CONTROL_PATH = os.path.join(BASE_DIR, "control.json")


def check_control():
    if os.path.exists(CONTROL_PATH):
        try:
            with open(CONTROL_PATH, encoding="utf-8") as f:
                c = json.load(f)
            if c.get("abort"):
                return True
        except Exception:
            pass
    return False


def _dock_one(args):
    idx, smiles, box_center, box_size, tag, t_run_start, cpu = args
    t0 = time.time()
    score = dock_smiles_isolated(
        smiles, RECEPTOR_PDBQT, box_center, box_size, WORKDIR, tag=tag,
        exhaustiveness=8, timeout=120, cpu=cpu,
    )
    t1 = time.time()
    return {"idx": idx, "score": score, "dock_sec": t1 - t0, "offset_from_start_sec": t1 - t_run_start}


def main():
    minutes = float(sys.argv[1]) if len(sys.argv) > 1 else 30.0
    n_workers = int(sys.argv[2]) if len(sys.argv) > 2 else 6

    if not os.path.exists(RECEPTOR_PDBQT):
        print(f"ОШИБКА: нет {RECEPTOR_PDBQT}"); sys.exit(1)

    print(f"[setup] Определяю box_center/box_size из локального {LIGAND_STRUCTURE_PDB} (без сети)...")
    ligand_info = find_ligand_center(LIGAND_STRUCTURE_PDB)
    box_center = ligand_info["center"]
    box_size = ligand_info["box_size"]

    pool_smiles = load_ligand_pool(200)  # с запасом, дальше просто идём по кругу
    print(f"[setup] пул лигандов: {len(pool_smiles)}, воркеров: {n_workers}, "
          f"длительность: {minutes:.0f} мин, CPU-ядер видно системе: {multiprocessing.cpu_count()}")
    os.makedirs(WORKDIR, exist_ok=True)

    cpu = cpu_per_worker(n_workers)
    print(f"[setup] --cpu на процесс Vina: {cpu if cpu else 'не ограничено'} "
          f"(при n_workers>1 без этого получаем oversubscription - см. cpu_per_worker())")

    deadline = time.time() + minutes * 60
    t_run_start = time.time()
    all_results = []
    batch_size = n_workers * 4  # чтобы не пересоздавать Pool на каждый лиганд, но и не гнать слишком крупными пачками
    idx = 0

    with multiprocessing.Pool(processes=n_workers) as pool:
        while time.time() < deadline:
            if check_control():
                print("[control] control.json просит остановиться - прерываю.")
                break
            tasks = []
            for _ in range(batch_size):
                s = pool_smiles[idx % len(pool_smiles)]
                tasks.append((idx, s, box_center, box_size, f"throttle_{idx}", t_run_start, cpu))
                idx += 1
            for res in pool.imap_unordered(_dock_one, tasks):
                all_results.append(res)
            elapsed_min = (time.time() - t_run_start) / 60
            recent = [r["dock_sec"] for r in all_results[-batch_size:]]
            print(f"  [{elapsed_min:5.1f} мин] обработано {len(all_results)} лигандов всего, "
                  f"среднее по последней партии: {sum(recent)/len(recent):.2f}с/лиганд")
            with open(OUT_PATH, "w", encoding="utf-8") as f:
                json.dump({"n_workers": n_workers, "minutes_requested": minutes,
                           "results": all_results}, f, indent=2)

    if len(all_results) < 10:
        print("\nСлишком мало лигандов обработано для содержательного вывода о троттлинге.")
        return

    # первая треть по времени vs последняя треть - признак троттлинга,
    # если среднее время на лиганд заметно выросло
    n = len(all_results)
    first_third = all_results[: n // 3]
    last_third = all_results[-(n // 3):]
    avg_first = sum(r["dock_sec"] for r in first_third) / len(first_third)
    avg_last = sum(r["dock_sec"] for r in last_third) / len(last_third)
    drift_pct = (avg_last - avg_first) / avg_first * 100

    print(f"\n=== ИТОГ ===")
    print(f"  всего задокировано: {n} лигандов за {minutes:.0f} мин, {n_workers} воркеров")
    print(f"  среднее время/лиганд, первая треть прогона: {avg_first:.2f}с")
    print(f"  среднее время/лиганд, последняя треть прогона: {avg_last:.2f}с")
    print(f"  дрейф: {drift_pct:+.1f}%"
          + (" - ПОХОЖЕ НА ТРОТТЛИНГ, закладывай эту просадку в долгие прогоны"
             if drift_pct > 15 else " - в пределах шума, троттлинг не выражен"))
    print(f"\nПолные результаты (с offset_from_start_sec на каждый лиганд - можно построить график "
          f"времени докинга от момента запуска): {OUT_PATH}")


if __name__ == "__main__":
    main()
