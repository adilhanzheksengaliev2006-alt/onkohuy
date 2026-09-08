"""
Test 12 (Stage 3) — ranking power: среди ТОЛЬКО активных, коррелирует ли
докинг-скор с реальной потентностью (pChEMBL)? Spearman + Kendall,
отдельно Vina/gnina. Терминология "screening power" vs "ranking power" -
как в CASF-2016.

pChEMBL = 9 - log10(IC50_нМ) считается из уже сохранённого ic50_nm в
ligands.json - рефетч из ChEMBL не нужен.

Источник: runs/test_a_<GENE>/{ligands.json,results.jsonl}.

Использование:
    python controls/test12_ranking_power.py GENE [--force]
"""
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import protocol  # noqa: E402

SCORE_COLUMNS = [
    ("docking_score_kcal_mol", "Vina"),
    ("gnina_cnn_affinity", "gnina CNNaffinity"),
]


def load_actives_with_pchembl(gene):
    lig_path = os.path.join(protocol.runs_dir(gene), "ligands.json")
    res_path = os.path.join(protocol.runs_dir(gene), "results.jsonl")
    if not os.path.exists(lig_path) or not os.path.exists(res_path):
        print(f"ОШИБКА: нужны и {lig_path}, и {res_path}"); sys.exit(1)

    with open(lig_path, encoding="utf-8") as f:
        ligs = {r["chembl_id"]: r for r in json.load(f)["ligands"] if r["label"] == 1 and r.get("ic50_nm")}

    scores_by_id = {}
    with open(res_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r["label"] == 1 and r["chembl_id"] in ligs:
                scores_by_id[r["chembl_id"]] = r

    rows = []
    for cid, lig in ligs.items():
        r = scores_by_id.get(cid)
        if not r:
            continue
        ic50_nm = lig["ic50_nm"]
        if ic50_nm is None or ic50_nm <= 0:
            continue
        pchembl = 9 - math.log10(ic50_nm)
        row = {"chembl_id": cid, "pchembl": pchembl}
        for score_col, _ in SCORE_COLUMNS:
            row[score_col] = r.get(score_col)
        rows.append(row)
    return rows


def run(gene, force=False):
    if protocol.test_already_done(gene, 3, "test12_ranking_power", force):
        print(f"[test12] {gene}: уже посчитано, пропускаю. --force для пересчёта")
        return protocol.load_stage(gene, 3)["test12_ranking_power"]

    protocol.print_banner("test12")
    rows = load_actives_with_pchembl(gene)
    print(f"[test12] {gene}: активных с ic50_nm и докинг-скором: {len(rows)}")

    if len(rows) < 5:
        result = {"error": f"недостаточно активных с pChEMBL (n={len(rows)}, нужно >=5)", "n": len(rows)}
        protocol.save_stage(gene, 3, {"test12_ranking_power": result})
        print(f"[test12] [warn] {result['error']}")
        return result

    import pandas as pd
    from scipy.stats import spearmanr, kendalltau
    df = pd.DataFrame(rows)

    by_score = {}
    plot_paths = {}
    for score_col, label_name in SCORE_COLUMNS:
        sub = df[[score_col, "pchembl"]].dropna()
        if len(sub) < 5:
            by_score[score_col] = {"error": f"n={len(sub)} < 5 после dropna"}
            continue
        sp_corr, sp_p = spearmanr(sub[score_col], sub["pchembl"])
        kt_corr, kt_p = kendalltau(sub[score_col], sub["pchembl"])
        by_score[score_col] = {
            "label": label_name, "n": int(len(sub)),
            "spearman": float(sp_corr), "spearman_p": float(sp_p),
            "kendall": float(kt_corr), "kendall_p": float(kt_p),
        }
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.scatter(sub[score_col], sub["pchembl"], alpha=0.6)
            ax.set_xlabel(label_name); ax.set_ylabel("pChEMBL (from ic50_nm)")
            ax.set_title(f"Test 12 ranking power - {gene}\nSpearman={sp_corr:.3f} (p={sp_p:.2e})")
            path = os.path.join(protocol.results_dir(gene), f"test12_scatter_{score_col}.png")
            fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)
            plot_paths[score_col] = path
        except Exception as e:
            print(f"[test12] [warn] не удалось построить scatter для {score_col}: {e}")

    result = {"n_actives_with_pchembl": len(rows), "by_score": by_score, "plots": plot_paths}
    protocol.save_stage(gene, 3, {"test12_ranking_power": result})

    print(f"\n=== Test 12 ({gene}): ranking power (actives only, vs pChEMBL) ===")
    for score_col, r in by_score.items():
        if "error" in r:
            print(f"  {score_col}: {r['error']}"); continue
        print(f"  {r['label']:18s} n={r['n']:4d}  Spearman={r['spearman']:+.3f} (p={r['spearman_p']:.3f})  "
              f"Kendall={r['kendall']:+.3f} (p={r['kendall_p']:.3f})")
    return result


def main():
    gene = sys.argv[1] if len(sys.argv) > 1 else "PIK3CA"
    force = "--force" in sys.argv
    run(gene, force)


if __name__ == "__main__":
    main()
