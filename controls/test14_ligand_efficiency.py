"""
Test 14 (Stage 3) — ligand efficiency: LE = score / heavy_atoms^n, для
n из metrics.le_exponents (1.0 - классический LE, 2/3 и 1/2 - варианты,
меньше штрафующие крупные молекулы). Пересчитываем BEDROC/EF1% на LE
вместо сырого скора - если LE ранжирует лучше сырого скора, это
намекает, что сырой скор был засорён размерным сигналом.

Источник: runs/test_a_<GENE>/results.jsonl.

Использование:
    python controls/test14_ligand_efficiency.py GENE [--force]
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import protocol  # noqa: E402


def load_results_df(gene):
    import pandas as pd
    path = os.path.join(protocol.runs_dir(gene), "results.jsonl")
    if not os.path.exists(path):
        print(f"ОШИБКА: нет {path}"); sys.exit(1)
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def run(gene, force=False):
    if protocol.test_already_done(gene, 3, "test14_ligand_efficiency", force):
        print(f"[test14] {gene}: уже посчитано, пропускаю. --force для пересчёта")
        return protocol.load_stage(gene, 3)["test14_ligand_efficiency"]

    protocol.print_banner("test14", ["metrics.le_exponents", "metrics.bedroc_alpha"])
    exponents = protocol.cfg_get("metrics", "le_exponents")
    alpha = protocol.cfg_get("metrics", "bedroc_alpha")

    df = load_results_df(gene)
    sub = df[["docking_score_kcal_mol", "heavy_atom_count", "label"]].dropna()
    print(f"[test14] {gene}: n={len(sub)} с скором и heavy_atom_count")

    from rdkit.ML.Scoring import Scoring
    from sklearn.metrics import roc_auc_score

    def bedroc_ef_auc(score_series, ascending):
        s = sub.copy()
        s["_score"] = score_series
        s = s.sort_values("_score", ascending=ascending)
        matrix = s[["_score", "label"]].values.tolist()
        bedroc = Scoring.CalcBEDROC(matrix, 1, alpha)
        ef1 = Scoring.CalcEnrichment(matrix, 1, [0.01])[0]
        # AUC ожидает "выше=позитивнее" - при ascending=True (меньше=лучше) инвертируем для sklearn
        y_score = -s["_score"] if ascending else s["_score"]
        auc = roc_auc_score(s["label"], y_score)
        return float(bedroc), float(ef1), float(auc)

    raw_bedroc, raw_ef1, raw_auc = bedroc_ef_auc(sub["docking_score_kcal_mol"], ascending=True)

    by_exponent = {"raw_score": {"bedroc": raw_bedroc, "ef1pct": raw_ef1, "auc": raw_auc}}
    for n in exponents:
        le = sub["docking_score_kcal_mol"] / (sub["heavy_atom_count"] ** n)
        # Vina score отрицательный (лучше=более отрицательный), heavy_atoms>0,
        # поэтому LE сохраняет тот же знак-порядок: меньше (более отрицательный)=лучше
        bedroc, ef1, auc = bedroc_ef_auc(le, ascending=True)
        by_exponent[f"n={n}"] = {"bedroc": bedroc, "ef1pct": ef1, "auc": auc}

    result = {"n": len(sub), "exponents": exponents, "by_exponent": by_exponent}
    protocol.save_stage(gene, 3, {"test14_ligand_efficiency": result})

    print(f"\n=== Test 14 ({gene}): ligand efficiency at multiple exponents ===")
    for key, r in by_exponent.items():
        print(f"  {key:12s} BEDROC={r['bedroc']:.3f}  EF1%={r['ef1pct']:.2f}  AUC={r['auc']:.3f}")
    return result


def main():
    gene = sys.argv[1] if len(sys.argv) > 1 else "PIK3CA"
    force = "--force" in sys.argv
    run(gene, force)


if __name__ == "__main__":
    main()
