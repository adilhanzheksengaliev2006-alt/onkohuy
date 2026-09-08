"""
Test 10 (Stage 2) — сравнение exhaustiveness=4 vs 8 (или других значений
из config): если ранжирование лигандов (Spearman) почти не меняется,
можно снизить exhaustiveness в основном пайплайне и сэкономить время
докинга без потери сигнала.

ВАЖНО: докинг идёт через multiprocessing.Pool (docking.n_workers, по
умолчанию 6), по одному воркеру на ЛИГАНД (low+high считаются в одном
воркере, чтобы не терять парность лиганд<->оба скора) - раньше здесь был
строго последовательный цикл (1 vina.exe за раз), из-за чего полный
конфиг (200 лигандов x 2 уровня = 400 вызовов) растягивался на часы.
Обнаружено и исправлено во время тестового прогона на CDK2.

Использование:
    python controls/test10_exhaustiveness.py GENE [--smoke] [--force]
"""
import json
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import protocol  # noqa: E402
from dock_existing_candidates import prepare_ligand_pdbqt, run_vina  # noqa: E402
from gene_target_utils import find_ligand_center  # noqa: E402
from benchmark_parallel_docking import cpu_per_worker  # noqa: E402


def _dock_low_high_task(args):
    """Модуль-уровневая функция для multiprocessing.Pool. Считает low И
    high для ОДНОГО лиганда в своей временной директории - так пара
    (score_low, score_high) остаётся привязана к одному и тому же
    воркеру/лиганду, а не зависит от порядка завершения задач."""
    # Целиком в try/except: одна молекула, упавшая непредвиденно, не должна
    # ронять весь multiprocessing.Pool (тот же класс бага, найденный и
    # исправленный в test09_seed_variance.py - см. комментарий там).
    ligand_idx, smiles, receptor_pdbqt, box_center, box_size, exh_low, exh_high, cpu = args
    try:
        with tempfile.TemporaryDirectory() as workdir:
            ligand_pdbqt = os.path.join(workdir, "lig.pdbqt")
            if not prepare_ligand_pdbqt(smiles, ligand_pdbqt):
                return ligand_idx, None, None, None, None
            t0 = time.time()
            score_low = run_vina(receptor_pdbqt, ligand_pdbqt, box_center, box_size,
                                  os.path.join(workdir, "low.pdbqt"), exhaustiveness=exh_low, timeout=180, cpu=cpu)
            t_low = time.time() - t0
            t0 = time.time()
            score_high = run_vina(receptor_pdbqt, ligand_pdbqt, box_center, box_size,
                                   os.path.join(workdir, "high.pdbqt"), exhaustiveness=exh_high, timeout=300, cpu=cpu)
            t_high = time.time() - t0
            return ligand_idx, score_low, score_high, t_low, t_high
    except Exception:
        return ligand_idx, None, None, None, None


def load_receptor_and_box(gene):
    with open(os.path.join(protocol.BASE_DIR, "confirmed_structures.json"), encoding="utf-8") as f:
        confirmed = json.load(f)[gene]
    pdb_id = confirmed["pdb_id"]
    struct_dir = os.path.join(protocol.BASE_DIR, "structures")
    receptor_pdbqt = os.path.join(struct_dir, f"{pdb_id}_receptor.pdbqt")
    original_pdb = os.path.join(struct_dir, f"{pdb_id}.pdb")
    info = find_ligand_center(original_pdb)
    return receptor_pdbqt, info["center"], info["box_size"]


def load_sample_smiles(gene, n):
    path = os.path.join(protocol.runs_dir(gene), "ligands.json")
    if not os.path.exists(path):
        print(f"ОШИБКА: нет {path}"); sys.exit(1)
    with open(path, encoding="utf-8") as f:
        ligs = json.load(f)["ligands"]
    return [r["smiles"] for r in ligs[:n]]


def run(gene, force=False, smoke=False):
    if protocol.test_already_done(gene, 2, "test10_exhaustiveness", force):
        print(f"[test10] {gene}: уже посчитано, пропускаю. --force для пересчёта")
        return protocol.load_stage(gene, 2)["test10_exhaustiveness"]

    protocol.print_banner("test10", ["test10_exhaustiveness.exhaustiveness_low",
                                      "test10_exhaustiveness.exhaustiveness_high", "docking.n_workers"])
    cfg = protocol.cfg_get("test10_exhaustiveness")
    n_ligands = cfg["n_ligands_smoke_test"] if smoke else cfg["n_ligands"]
    exh_low, exh_high = cfg["exhaustiveness_low"], cfg["exhaustiveness_high"]
    switch_thresh = cfg["spearman_switch_threshold"]
    n_workers = protocol.cfg_get("docking", "n_workers", default=6)

    receptor_pdbqt, box_center, box_size = load_receptor_and_box(gene)
    smiles_list = load_sample_smiles(gene, n_ligands)
    cpu = cpu_per_worker(n_workers)
    print(f"[test10] {gene}: {len(smiles_list)} лигандов, exhaustiveness {exh_low} vs {exh_high}, "
          f"{n_workers} воркеров, --cpu={cpu} на воркер ({'SMOKE TEST' if smoke else 'полный прогон'})")

    import multiprocessing
    tasks = [(i, smi, receptor_pdbqt, box_center, box_size, exh_low, exh_high, cpu)
             for i, smi in enumerate(smiles_list)]
    rows = []
    t0 = time.time()
    n_done = 0
    with multiprocessing.Pool(processes=n_workers) as pool:
        for ligand_idx, score_low, score_high, t_low, t_high in pool.imap_unordered(_dock_low_high_task, tasks):
            n_done += 1
            if score_low is not None and score_high is not None:
                rows.append({"ligand_idx": ligand_idx, "score_low": score_low, "score_high": score_high,
                             "time_low_sec": t_low, "time_high_sec": t_high})
            if n_done % 10 == 0 or n_done == len(tasks):
                print(f"  [{n_done}/{len(tasks)}] {(time.time()-t0)/60:.1f} мин прошло")

    from scipy.stats import spearmanr
    import numpy as np
    scores_low = [r["score_low"] for r in rows]
    scores_high = [r["score_high"] for r in rows]
    corr, p_value = spearmanr(scores_low, scores_high) if len(rows) >= 3 else (None, None)
    mean_time_low = float(np.mean([r["time_low_sec"] for r in rows])) if rows else None
    mean_time_high = float(np.mean([r["time_high_sec"] for r in rows])) if rows else None

    result = {
        "n_ligands": len(rows), "exhaustiveness_low": exh_low, "exhaustiveness_high": exh_high,
        "n_workers": n_workers, "smoke_test": smoke, "per_ligand": rows,
        "spearman_corr": float(corr) if corr is not None else None,
        "spearman_p": float(p_value) if p_value is not None else None,
        "mean_time_low_sec": mean_time_low, "mean_time_high_sec": mean_time_high,
        "switch_threshold": switch_thresh,
        "recommend_switch_to_low": bool(corr is not None and corr > switch_thresh),
    }
    protocol.save_stage(gene, 2, {"test10_exhaustiveness": result})

    print(f"\n=== Test 10 ({gene}): exhaustiveness {exh_low} vs {exh_high} ===")
    if corr is not None:
        print(f"  Spearman corr={corr:.4f} (p={p_value:.2e}), порог переключения={switch_thresh}")
        print(f"  среднее время: low={mean_time_low:.1f}s high={mean_time_high:.1f}s")
        if result["recommend_switch_to_low"]:
            print(f"  [!] корреляция выше порога - можно переключить основной пайплайн на exhaustiveness={exh_low}")
    else:
        print(f"  недостаточно данных для Spearman (n={len(rows)})")
    return result


def main():
    gene = sys.argv[1] if len(sys.argv) > 1 else "PIK3CA"
    smoke = "--smoke" in sys.argv
    force = "--force" in sys.argv
    run(gene, force, smoke)


if __name__ == "__main__":
    main()
