"""
Test 3 (Stage 0) — AVE bias (Asymmetric Validation Embedding,
Wallach & Heifets, JCIM 2018): одно число на мишень, отражающее
искусственную "разделимость" actives/decoys по чистому химическому
сходству (ECFP4/Tanimoto), без какого-либо докинга.

AVE = <NN sim AA> - <NN sim AI> + <NN sim II> - <NN sim IA>
(средние ближайшие Tanimoto-сходства внутри/между классами - здесь
считаем упрощённо по всему датасету, без train/test сплита, т.к. цель -
диагностика самого датасета, а не валидация модели).

Источник: runs/test_a_<GENE>/ligands.json.

Использование:
    python controls/test03_ave_bias.py GENE [--force]
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import protocol  # noqa: E402


def load_ligands(gene):
    path = os.path.join(protocol.runs_dir(gene), "ligands.json")
    if not os.path.exists(path):
        print(f"ОШИБКА: нет {path}"); sys.exit(1)
    with open(path, encoding="utf-8") as f:
        return json.load(f)["ligands"]


def compute_fingerprints(smiles_list):
    from rdkit import Chem
    from rdkit.Chem import AllChem
    fps = []
    valid_idx = []
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048))
        valid_idx.append(i)
    return fps, valid_idx


def nearest_neighbor_mean_sim(fps_query, fps_pool, exclude_self=False):
    """Для каждого fp в fps_query - максимальное Tanimoto-сходство к fps_pool,
    затем среднее по всем query. exclude_self=True когда query и pool - один
    и тот же список (не считаем сходство молекулы с собой)."""
    import numpy as np
    from rdkit import DataStructs
    sims = []
    for i, fp in enumerate(fps_query):
        pool = fps_pool if not exclude_self else [f for j, f in enumerate(fps_pool) if j != i]
        if not pool:
            continue
        best = max(DataStructs.BulkTanimotoSimilarity(fp, pool))
        sims.append(best)
    return float(np.mean(sims)) if sims else None


def run(gene, force=False, max_per_class=1500):
    if protocol.test_already_done(gene, 0, "test03_ave_bias", force):
        print(f"[test03] {gene}: уже посчитано, пропускаю. --force для пересчёта")
        return protocol.load_stage(gene, 0)["test03_ave_bias"]

    protocol.print_banner("test03", ["flags.ave_bias_warn"])
    warn_thresh = protocol.cfg_get("flags", "ave_bias_warn")

    ligands = load_ligands(gene)
    actives_smiles = [r["smiles"] for r in ligands if r["label"] == 1]
    decoys_smiles = [r["smiles"] for r in ligands if r["label"] == 0]

    # ВАЖНО: nearest-neighbour similarity механически растёт с размером пула
    # (в большом пуле проще найти близкого соседа даже без реального
    # сходства) - без выравнивания размеров классов decoys (обычно их на
    # порядок больше) искусственно взвинчивают nn_sim_II и ломают AVE.
    # Подрезаем оба класса до min(len(actives), len(decoys), max_per_class)
    # с фиксированным seed, чтобы результат был воспроизводим.
    import random
    rng = random.Random(protocol.cfg_get("metrics", "seed", default=0))
    n_target = min(len(actives_smiles), len(decoys_smiles), max_per_class)
    if len(decoys_smiles) > n_target:
        decoys_smiles = rng.sample(decoys_smiles, n_target)
    if len(actives_smiles) > n_target:
        actives_smiles = rng.sample(actives_smiles, n_target)

    fps_a, _ = compute_fingerprints(actives_smiles)
    fps_d, _ = compute_fingerprints(decoys_smiles)
    print(f"[test03] {gene}: actives={len(fps_a)}, decoys={len(fps_d)} "
          f"(классы выровнены по размеру до {n_target}, seed зафиксирован)")

    if len(fps_a) < 2 or len(fps_d) < 2:
        result = {"error": "недостаточно валидных молекул для AVE"}
        protocol.save_stage(gene, 0, {"test03_ave_bias": result})
        print(f"[test03] [warn] {result['error']}")
        return result

    nn_aa = nearest_neighbor_mean_sim(fps_a, fps_a, exclude_self=True)
    nn_ai = nearest_neighbor_mean_sim(fps_a, fps_d, exclude_self=False)
    nn_ii = nearest_neighbor_mean_sim(fps_d, fps_d, exclude_self=True)
    nn_ia = nearest_neighbor_mean_sim(fps_d, fps_a, exclude_self=False)

    ave = (nn_aa - nn_ai) + (nn_ii - nn_ia)

    result = {
        "n_actives": len(fps_a), "n_decoys": len(fps_d), "n_class_matched": n_target,
        "nn_sim_AA": nn_aa, "nn_sim_AI": nn_ai, "nn_sim_II": nn_ii, "nn_sim_IA": nn_ia,
        "ave_bias": float(ave), "warn_threshold": warn_thresh,
        "warn_high_ave_bias": bool(abs(ave) > warn_thresh),
        "citation": "Wallach & Heifets, JCIM 2018 - Asymmetric Validation Embedding",
    }
    protocol.save_stage(gene, 0, {"test03_ave_bias": result})

    print(f"\n=== Test 3 ({gene}): AVE bias ===")
    print(f"  <sim AA>={nn_aa:.3f}  <sim AI>={nn_ai:.3f}  <sim II>={nn_ii:.3f}  <sim IA>={nn_ia:.3f}")
    print(f"  AVE = {ave:+.3f} (порог предупреждения: {warn_thresh})")
    if result["warn_high_ave_bias"]:
        print(f"  [!] высокий AVE bias - датасет может быть искусственно 'разделим' по чистой химии")
    return result


def main():
    gene = sys.argv[1] if len(sys.argv) > 1 else "PIK3CA"
    force = "--force" in sys.argv
    run(gene, force)


if __name__ == "__main__":
    main()
