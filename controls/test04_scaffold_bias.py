"""
Test 4 (Stage 0) — analog/scaffold bias среди ACTIVES: если все известные
активные вещества - это по сути один и тот же скэлд с косметическими
заменами заместителей, любая модель (или докинг) может "угадывать"
активность по скэлду, а не по реальному связыванию.

Считаем: доля крупнейшего Murcko-скэлд-кластера, число уникальных
скэлдов, отношение скэлдов к молекулам.

Источник: runs/test_a_<GENE>/ligands.json (только label==1).

Использование:
    python controls/test04_scaffold_bias.py GENE [--force]
"""
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import protocol  # noqa: E402


def load_actives(gene):
    path = os.path.join(protocol.runs_dir(gene), "ligands.json")
    if not os.path.exists(path):
        print(f"ОШИБКА: нет {path}"); sys.exit(1)
    with open(path, encoding="utf-8") as f:
        ligs = json.load(f)["ligands"]
    return [r for r in ligs if r["label"] == 1]


def murcko_scaffold(smiles):
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
    except Exception:
        return None


def run(gene, force=False):
    if protocol.test_already_done(gene, 0, "test04_scaffold_bias", force):
        print(f"[test04] {gene}: уже посчитано, пропускаю. --force для пересчёта")
        return protocol.load_stage(gene, 0)["test04_scaffold_bias"]

    protocol.print_banner("test04", ["flags.scaffold_dominance_warn"])
    warn_thresh = protocol.cfg_get("flags", "scaffold_dominance_warn")

    actives = load_actives(gene)
    scaffolds = []
    for r in actives:
        s = murcko_scaffold(r["smiles"])
        if s is not None:
            scaffolds.append(s if s else "NO_RING_SCAFFOLD")

    n_mols = len(scaffolds)
    counts = Counter(scaffolds)
    n_unique = len(counts)
    largest_scaffold, largest_count = counts.most_common(1)[0] if counts else (None, 0)
    largest_fraction = largest_count / n_mols if n_mols else 0.0
    ratio = n_unique / n_mols if n_mols else 0.0

    result = {
        "n_actives_total": len(actives), "n_actives_parsed": n_mols,
        "n_unique_scaffolds": n_unique,
        "largest_cluster_scaffold_smiles": largest_scaffold,
        "largest_cluster_fraction": float(largest_fraction),
        "scaffold_to_molecule_ratio": float(ratio),
        "warn_threshold": warn_thresh,
        "warn_scaffold_dominance": bool(largest_fraction > warn_thresh),
        "top5_scaffolds": [{"scaffold": s, "count": c} for s, c in counts.most_common(5)],
    }
    protocol.save_stage(gene, 0, {"test04_scaffold_bias": result})

    print(f"\n=== Test 4 ({gene}): analog/scaffold bias (actives only) ===")
    print(f"  actives: {n_mols} распарсено из {len(actives)}")
    print(f"  уникальных Murcko-скэлдов: {n_unique}  (скэлд/молекула = {ratio:.3f})")
    print(f"  крупнейший кластер: {largest_count}/{n_mols} = {largest_fraction:.3f} "
          f"(порог предупреждения: {warn_thresh})")
    if result["warn_scaffold_dominance"]:
        print(f"  [!] активные сильно доминированы одним скэлдом - риск, что модель учит скэлд, а не связывание")
    return result


def main():
    gene = sys.argv[1] if len(sys.argv) > 1 else "PIK3CA"
    force = "--force" in sys.argv
    run(gene, force)


if __name__ == "__main__":
    main()
