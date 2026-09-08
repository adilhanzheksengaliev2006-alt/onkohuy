"""
Test 9 (Stage 2) — собственная дисперсия Vina от --seed: если разброс
скора одной и той же молекулы между случайными сидами сопоставим с
разницей между активными и неактивными, то различия в BEDROC/ранжировании
могут быть просто шумом движка, а не сигналом связывания.

Обобщено из vina_variance.py (был захардкожен под PIK3CA/4JPS,
N=10 лигандов): теперь работает по любому GENE, читает бокс из
confirmed_structures.json, число лигандов/сидов - из
config/protocol.yaml (test09_seed_variance), сравнивает std с медианным
разрывом active-vs-inactive скора (если есть runs/test_a_<GENE>/results.jsonl).

ВАЖНО: докинг всех (лиганд x сид) комбинаций идёт через multiprocessing.Pool
(docking.n_workers, по умолчанию 6) - раньше здесь был строго
последовательный цикл (1 vina.exe за раз), из-за чего полный конфиг
(50 лигандов x 6 сидов = 300 вызовов) растягивался на часы вместо
заявленных в аудите "30-60 минут на подвыборку". Обнаружено и исправлено
во время тестового прогона на CDK2.

Использование:
    python controls/test09_seed_variance.py GENE [--smoke] [--force]
    --smoke: n_ligands_smoke_test вместо полного n_ligands (быстрая проверка)
"""
import json
import os
import re
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import protocol  # noqa: E402
from dock_existing_candidates import prepare_ligand_pdbqt, VINA_EXE  # noqa: E402
from gene_target_utils import find_ligand_center  # noqa: E402
from benchmark_parallel_docking import cpu_per_worker  # noqa: E402


def run_vina_with_seed(receptor_pdbqt, ligand_pdbqt, box_center, box_size, out_pdbqt, seed, exhaustiveness, timeout=None, cpu=None):
    cmd = [
        VINA_EXE, "--receptor", receptor_pdbqt, "--ligand", ligand_pdbqt,
        "--center_x", str(box_center[0]), "--center_y", str(box_center[1]), "--center_z", str(box_center[2]),
        "--size_x", str(box_size[0]), "--size_y", str(box_size[1]), "--size_z", str(box_size[2]),
        "--out", out_pdbqt, "--exhaustiveness", str(exhaustiveness), "--seed", str(seed),
    ]
    if cpu is not None:
        cmd += ["--cpu", str(cpu)]
    # было захардкожено 180с, не из конфига - несогласовано с остальным
    # пайплайном (300с везде, после болезненного прошлого урока про
    # слишком короткие таймауты). Дефолт теперь читается из
    # docking.timeout_sec, а не захардкожен здесь отдельно.
    effective_timeout = timeout if timeout is not None else protocol.cfg_get("docking", "timeout_sec", default=300)
    # subprocess.TimeoutExpired здесь раньше не ловился - в
    # multiprocessing.Pool.imap_unordered одно зависшее на таймауте vina.exe
    # роняло ВЕСЬ пул вместо того, чтобы просто засчитать эту (лиганд,сид)
    # комбинацию как неудачную (тот же класс бага, что был у
    # prepare_ligand_pdbqt - см. фикс в dock_existing_candidates.py).
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=effective_timeout)
    except subprocess.TimeoutExpired:
        return None
    combined = result.stdout + result.stderr
    matches = re.findall(r"^\s*\d+\s+(-?\d+\.\d+)", combined, re.MULTILINE)
    return float(matches[0]) if matches else None


def _dock_one_seed_task(args):
    """Модуль-уровневая функция для multiprocessing.Pool (нужна picklable
    функция, не замыкание). Готовит лиганд и докует в СВОЕЙ временной
    директории, чтобы параллельные воркеры не затирали файлы друг друга.

    Обёрнуто целиком в try/except Exception (не только известные точки
    отказа выше) - ОДНА молекула, упавшая по любой непредвиденной причине
    (диск, кодировка, что угодно), не должна ронять весь
    multiprocessing.Pool и вместе с ним многочасовой прогон. Найдено и
    исправлено в бою: mk_prepare_ligand.exe завис на 60с на реальной
    молекуле и уронил весь Stage 2."""
    ligand_idx, smiles, seed, receptor_pdbqt, box_center, box_size, exhaustiveness, cpu = args
    try:
        with tempfile.TemporaryDirectory() as workdir:
            ligand_pdbqt = os.path.join(workdir, "lig.pdbqt")
            if not prepare_ligand_pdbqt(smiles, ligand_pdbqt):
                return ligand_idx, seed, None
            out_pdbqt = os.path.join(workdir, "out.pdbqt")
            score = run_vina_with_seed(receptor_pdbqt, ligand_pdbqt, box_center, box_size,
                                        out_pdbqt, seed, exhaustiveness, cpu=cpu)
            return ligand_idx, seed, score
    except Exception:
        return ligand_idx, seed, None


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


def median_active_inactive_gap(gene):
    """Медианный |score_active - score_inactive| по уже посчитанному Vina-докингу,
    если есть - для сравнения с seed-шумом. None, если данных нет."""
    import numpy as np
    path = os.path.join(protocol.runs_dir(gene), "results.jsonl")
    if not os.path.exists(path):
        return None
    actives, decoys = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            s = r.get("docking_score_kcal_mol")
            if s is None:
                continue
            (actives if r["label"] == 1 else decoys).append(s)
    if not actives or not decoys:
        return None
    return abs(float(np.median(actives)) - float(np.median(decoys)))


def run(gene, force=False, smoke=False):
    if protocol.test_already_done(gene, 2, "test09_seed_variance", force):
        print(f"[test09] {gene}: уже посчитано, пропускаю. --force для пересчёта")
        return protocol.load_stage(gene, 2)["test09_seed_variance"]

    protocol.print_banner("test09", ["test09_seed_variance.n_ligands", "test09_seed_variance.seeds",
                                      "docking.n_workers"])
    cfg = protocol.cfg_get("test09_seed_variance")
    n_ligands = cfg["n_ligands_smoke_test"] if smoke else cfg["n_ligands"]
    seeds = cfg["seeds"]
    exhaustiveness = cfg["exhaustiveness"]
    n_workers = protocol.cfg_get("docking", "n_workers", default=6)

    receptor_pdbqt, box_center, box_size = load_receptor_and_box(gene)
    smiles_list = load_sample_smiles(gene, n_ligands)
    n_tasks = len(smiles_list) * len(seeds)
    cpu = cpu_per_worker(n_workers)
    print(f"[test09] {gene}: {len(smiles_list)} лигандов x {len(seeds)} сидов = {n_tasks} докингов, "
          f"{n_workers} воркеров, --cpu={cpu} на воркер ({'SMOKE TEST' if smoke else 'полный прогон'})")

    import multiprocessing
    import numpy as np
    import time

    tasks = [
        (i, smi, seed, receptor_pdbqt, box_center, box_size, exhaustiveness, cpu)
        for i, smi in enumerate(smiles_list) for seed in seeds
    ]
    scores_by_ligand = {i: [] for i in range(len(smiles_list))}
    t0 = time.time()
    n_done = 0
    with multiprocessing.Pool(processes=n_workers) as pool:
        for ligand_idx, seed, score in pool.imap_unordered(_dock_one_seed_task, tasks):
            if score is not None:
                scores_by_ligand[ligand_idx].append(score)
            n_done += 1
            if n_done % 10 == 0 or n_done == n_tasks:
                elapsed = time.time() - t0
                print(f"  [{n_done}/{n_tasks}] {elapsed/60:.1f} мин прошло")

    rows = []
    for i, scores in scores_by_ligand.items():
        if scores:
            rows.append({"ligand_idx": i, "scores": scores, "mean": float(np.mean(scores)), "std": float(np.std(scores))})

    all_stds = [r["std"] for r in rows]
    mean_std = float(np.mean(all_stds)) if all_stds else None
    max_std = float(np.max(all_stds)) if all_stds else None
    gap = median_active_inactive_gap(gene)

    result = {
        "n_ligands": len(rows), "n_seeds": len(seeds), "exhaustiveness": exhaustiveness,
        "n_workers": n_workers, "smoke_test": smoke, "per_ligand": rows,
        "mean_std_across_ligands": mean_std, "max_std_across_ligands": max_std,
        "median_active_inactive_score_gap": gap,
        "seed_noise_vs_signal_ratio": (mean_std / gap) if (mean_std and gap) else None,
    }
    protocol.save_stage(gene, 2, {"test09_seed_variance": result})

    print(f"\n=== Test 9 ({gene}): seed variance ===")
    print(f"  mean std={mean_std}, max std={max_std} (ккал/моль, {len(seeds)} сидов, {len(rows)} лигандов)")
    if gap is not None:
        print(f"  медианный разрыв active-vs-inactive: {gap:.4f} ккал/моль")
        print(f"  отношение шум/сигнал: {result['seed_noise_vs_signal_ratio']:.3f}")
    else:
        print(f"  [warn] нет results.jsonl для {gene} - не с чем сравнить (докинг ещё не запускался)")
    return result


def main():
    gene = sys.argv[1] if len(sys.argv) > 1 else "PIK3CA"
    smoke = "--smoke" in sys.argv
    force = "--force" in sys.argv
    run(gene, force, smoke)


if __name__ == "__main__":
    main()
