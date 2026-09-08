"""
Test 1 (Stage 0) — bias по размеру/физхимии между actives и decoys
в самом датасете, ДО какого-либо докинга.

Для heavy_atoms/MW/logP/TPSA (и до кучи HBD/HBA/RotB/rings): Mann-Whitney U
+ Cliff's delta (эффект-сайз, не зависит от распределения) + медианы обеих
групп. Плюс перекрывающиеся гистограммы в PNG.

Источник: runs/test_a_<GENE>/ligands.json (SMILES+label, до докинга).
Ничего не исключает, только считает числа - см. flags.size_effect_warn
в config/protocol.yaml.

Использование:
    python controls/test01_dataset_bias.py GENE [--force]
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import protocol  # noqa: E402

BASE_DIR = protocol.BASE_DIR

DESCRIPTORS = ["heavy_atoms", "mw", "logp", "tpsa", "hbd", "hba", "rotb", "rings", "qed", "fraction_csp3"]


def compute_descriptors(smiles):
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Lipinski, QED, rdMolDescriptors
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return {
        "heavy_atoms": mol.GetNumHeavyAtoms(),
        "mw": Descriptors.MolWt(mol),
        "logp": Descriptors.MolLogP(mol),
        "tpsa": Descriptors.TPSA(mol),
        "hbd": Lipinski.NumHDonors(mol),
        "hba": Lipinski.NumHAcceptors(mol),
        "rotb": Descriptors.NumRotatableBonds(mol),
        "rings": rdMolDescriptors.CalcNumRings(mol),
        "qed": QED.qed(mol),
        "fraction_csp3": rdMolDescriptors.CalcFractionCSP3(mol),
    }


def cliffs_delta(x, y):
    """Cliff's delta: доля пар (xi,yj), где x>y, минус доля где x<y. [-1,1]."""
    import numpy as np
    x = np.asarray(x); y = np.asarray(y)
    gt = 0; lt = 0
    for xi in x:
        gt += (xi > y).sum()
        lt += (xi < y).sum()
    n = len(x) * len(y)
    return (gt - lt) / n if n else 0.0


def load_ligands(gene):
    path = os.path.join(protocol.runs_dir(gene), "ligands.json")
    if not os.path.exists(path):
        print(f"ОШИБКА: нет {path} - сначала нужен build (phase_build/_dude) для {gene}")
        sys.exit(1)
    with open(path, encoding="utf-8") as f:
        return json.load(f)["ligands"]


def run(gene, force=False):
    if protocol.test_already_done(gene, 0, "test01_dataset_bias", force):
        print(f"[test01] {gene}: уже посчитано (stage_0.json), пропускаю. --force для пересчёта")
        return protocol.load_stage(gene, 0)["test01_dataset_bias"]

    protocol.print_banner("test01", ["flags.size_effect_warn"])
    warn_thresh = protocol.cfg_get("flags", "size_effect_warn")

    ligands = load_ligands(gene)
    actives_desc = []
    decoys_desc = []
    n_failed_parse = 0
    for r in ligands:
        d = compute_descriptors(r["smiles"])
        if d is None:
            n_failed_parse += 1
            continue
        (actives_desc if r["label"] == 1 else decoys_desc).append(d)

    print(f"[test01] {gene}: actives={len(actives_desc)}, decoys={len(decoys_desc)}, "
          f"не распарсились={n_failed_parse}")

    from scipy.stats import mannwhitneyu
    import numpy as np

    per_descriptor = {}
    for key in DESCRIPTORS:
        a = [d[key] for d in actives_desc]
        b = [d[key] for d in decoys_desc]
        try:
            u_stat, p_value = mannwhitneyu(a, b, alternative="two-sided")
        except ValueError as e:
            per_descriptor[key] = {"error": str(e)}
            continue
        delta = cliffs_delta(a, b)
        per_descriptor[key] = {
            "median_actives": float(np.median(a)),
            "median_decoys": float(np.median(b)),
            "mannwhitney_u": float(u_stat),
            "mannwhitney_p": float(p_value),
            "cliffs_delta": float(delta),
            "warn_large_effect": bool(abs(delta) > warn_thresh),
        }

    # перекрывающиеся гистограммы для ключевых дескрипторов
    plot_path = None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        key_descs = ["heavy_atoms", "mw", "logp", "tpsa"]
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        for ax, key in zip(axes.flat, key_descs):
            a = [d[key] for d in actives_desc]
            b = [d[key] for d in decoys_desc]
            ax.hist(a, bins=30, alpha=0.5, label="actives", density=True, color="tab:orange")
            ax.hist(b, bins=30, alpha=0.5, label="decoys", density=True, color="tab:blue")
            ax.set_title(key)
            ax.legend()
        fig.suptitle(f"Test 1: dataset size/physchem bias - {gene}")
        fig.tight_layout()
        plot_path = os.path.join(protocol.results_dir(gene), "test01_histograms.png")
        fig.savefig(plot_path, dpi=110)
        plt.close(fig)
    except Exception as e:
        print(f"[test01] [warn] не удалось построить гистограммы: {e}")

    result = {
        "n_actives": len(actives_desc), "n_decoys": len(decoys_desc),
        "n_failed_parse": n_failed_parse,
        "per_descriptor": per_descriptor,
        "plot": plot_path,
        "warn_threshold_cliffs_delta": warn_thresh,
    }
    protocol.save_stage(gene, 0, {"test01_dataset_bias": result})

    print(f"\n=== Test 1 ({gene}): dataset bias ===")
    for key in DESCRIPTORS:
        r = per_descriptor.get(key, {})
        if "error" in r:
            print(f"  {key}: {r['error']}")
            continue
        flag = " [!]" if r["warn_large_effect"] else ""
        print(f"  {key:15s} median act={r['median_actives']:.2f} dec={r['median_decoys']:.2f} "
              f"Cliff's delta={r['cliffs_delta']:+.3f} p={r['mannwhitney_p']:.2e}{flag}")
    if plot_path:
        print(f"  гистограммы: {plot_path}")
    return result


def main():
    gene = sys.argv[1] if len(sys.argv) > 1 else "PIK3CA"
    force = "--force" in sys.argv
    run(gene, force)


if __name__ == "__main__":
    main()
