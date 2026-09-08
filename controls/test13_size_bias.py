"""
Test 13 (Stage 3) — Test A': ГЛАВНЫЙ РЕЗУЛЬТАТ ВСЕЙ РАБОТЫ. Насколько
докинг-скор объясняется просто числом тяжёлых атомов, а не реальным
связыванием.

Регрессия score ~ heavy_atoms: R^2/slope/p-value наклона - отдельно для
Vina и gnina, и отдельно на (a) только actives, (b) только decoys/inactives,
(c) всех вместе. Плюс partial correlation label~score, контролируя
heavy_atoms (через residualization обеих переменных на heavy_atoms и
корреляцию остатков) - показывает, остаётся ли связь score-label после
учёта размера.

R^2 > metrics.test_a_prime_r2_fail (0.5 по умолчанию) = провал контроля
(скор в основном отражает размер).

Источник: runs/test_a_<GENE>/results.jsonl.

Использование:
    python controls/test13_size_bias.py GENE [--force]
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import protocol  # noqa: E402

SCORE_COLUMNS = [("docking_score_kcal_mol", "Vina"), ("gnina_cnn_affinity", "gnina CNNaffinity")]


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


def regression_with_pvalue(x, y):
    import numpy as np
    import statsmodels.api as sm
    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit()
    slope = model.params[1]
    intercept = model.params[0]
    slope_p = model.pvalues[1]
    r2 = model.rsquared
    return {"n": int(len(x)), "slope": float(slope), "intercept": float(intercept),
            "r2": float(r2), "slope_p_value": float(slope_p)}


def partial_correlation_label_score_given_size(df, score_col, size_col="heavy_atom_count"):
    """Partial correlation через residualization: регрессируем label и score
    каждый отдельно на size, коррелируем остатки. Убирает линейный эффект
    размера из обеих переменных перед оценкой их связи."""
    import numpy as np
    import statsmodels.api as sm
    from scipy.stats import pearsonr

    sub = df[[score_col, size_col, "label"]].dropna()
    X = sm.add_constant(sub[size_col].values)
    resid_score = sm.OLS(sub[score_col].values, X).fit().resid
    resid_label = sm.OLS(sub["label"].values.astype(float), X).fit().resid
    corr, p_value = pearsonr(resid_score, resid_label)
    return {"n": int(len(sub)), "partial_corr_label_score_given_size": float(corr), "p_value": float(p_value)}


def run(gene, force=False):
    if protocol.test_already_done(gene, 3, "test13_size_bias", force):
        print(f"[test13] {gene}: уже посчитано, пропускаю. --force для пересчёта")
        return protocol.load_stage(gene, 3)["test13_size_bias"]

    protocol.print_banner("test13", ["metrics.test_a_prime_r2_fail"])
    r2_fail = protocol.cfg_get("metrics", "test_a_prime_r2_fail")

    df = load_results_df(gene)
    size_col = "heavy_atom_count"

    by_score = {}
    plot_paths = {}
    for score_col, label_name in SCORE_COLUMNS:
        if score_col not in df.columns or df[score_col].notna().sum() < 10:
            by_score[score_col] = {"error": "нет данных"}
            continue
        splits = {}
        subsets = {
            "all_combined": df,
            "actives_only": df[df["label"] == 1],
            "inactives_only": df[df["label"] == 0],
        }
        for split_name, sub_df in subsets.items():
            sub = sub_df[[score_col, size_col]].dropna()
            if len(sub) < 5:
                splits[split_name] = {"error": f"n={len(sub)} < 5"}
                continue
            reg = regression_with_pvalue(sub[size_col].values, sub[score_col].values)
            reg["size_bias_fail"] = bool(reg["r2"] > r2_fail)
            splits[split_name] = reg

        partial = partial_correlation_label_score_given_size(df, score_col, size_col)

        by_score[score_col] = {"label": label_name, "splits": splits, "partial_correlation": partial}

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import numpy as np
            sub_all = df[[score_col, size_col, "label"]].dropna()
            fig, ax = plt.subplots(figsize=(7, 6))
            colors = sub_all["label"].map({1: "tab:orange", 0: "tab:blue"})
            ax.scatter(sub_all[size_col], sub_all[score_col], c=colors, alpha=0.4, s=12)
            x = sub_all[size_col].values
            reg_all = splits["all_combined"]
            if "slope" in reg_all:
                xs = np.linspace(x.min(), x.max(), 50)
                ax.plot(xs, reg_all["slope"] * xs + reg_all["intercept"], color="black", linewidth=2,
                        label=f"R^2={reg_all['r2']:.3f}")
            ax.set_xlabel("heavy_atom_count"); ax.set_ylabel(label_name)
            ax.set_title(f"Test 13 (A') size-bias - {gene}\norange=active, blue=decoy")
            ax.legend()
            path = os.path.join(protocol.results_dir(gene), f"test13_scatter_{score_col}.png")
            fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)
            plot_paths[score_col] = path
        except Exception as e:
            print(f"[test13] [warn] не удалось построить scatter для {score_col}: {e}")

    result = {"r2_fail_threshold": r2_fail, "by_score": by_score, "plots": plot_paths}
    protocol.save_stage(gene, 3, {"test13_size_bias": result})

    print(f"\n=== Test 13 (A') ({gene}): SIZE BIAS - ГЛАВНЫЙ РЕЗУЛЬТАТ ===")
    for score_col, r in by_score.items():
        if "error" in r:
            print(f"  {score_col}: {r['error']}"); continue
        print(f"  --- {r['label']} ---")
        for split_name, reg in r["splits"].items():
            if "error" in reg:
                print(f"    {split_name}: {reg['error']}"); continue
            flag = " [FAIL >0.5]" if reg["size_bias_fail"] else ""
            print(f"    {split_name:15s} n={reg['n']:5d} R^2={reg['r2']:.3f} slope={reg['slope']:+.4f} "
                  f"(p={reg['slope_p_value']:.2e}){flag}")
        p = r["partial_correlation"]
        print(f"    partial corr(label,score | size)={p['partial_corr_label_score_given_size']:+.3f} "
              f"(p={p['p_value']:.2e}, n={p['n']})")
    return result


def main():
    gene = sys.argv[1] if len(sys.argv) > 1 else "PIK3CA"
    force = "--force" in sys.argv
    run(gene, force)


if __name__ == "__main__":
    main()
