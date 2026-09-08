"""
Test 11 (Stage 3) — screening power: BEDROC(alpha)/EF1%/EF5%/AUC-ROC,
отдельно для Vina и gnina (если есть), с bootstrap 95% CI.

ВАЖНО про знак: Vina "меньше=лучше" (сортировка ascending), у gnina
CNNscore/CNNaffinity "больше=лучше" (сортировка descending) - здесь это
явно параметризовано (ascending=True/False на каждый score_col), а не
предполагается неявно.

Источник: runs/test_a_<GENE>/results.jsonl.

Использование:
    python controls/test11_screening_power.py GENE [--force]
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import protocol  # noqa: E402
import bedroc_calibration as bc  # noqa: E402

# (колонка, ascending) - ascending=True значит "меньше значение = лучше связывание"
SCORE_COLUMNS = [
    ("docking_score_kcal_mol", True, "Vina"),
    ("gnina_cnn_score", False, "gnina CNNscore"),
    ("gnina_cnn_affinity", False, "gnina CNNaffinity"),
]


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
    if protocol.test_already_done(gene, 3, "test11_screening_power", force):
        print(f"[test11] {gene}: уже посчитано, пропускаю. --force для пересчёта")
        return protocol.load_stage(gene, 3)["test11_screening_power"]

    protocol.print_banner("test11", ["metrics.bedroc_alpha", "metrics.bootstrap_n"])
    alpha = protocol.cfg_get("metrics", "bedroc_alpha")
    n_bootstrap = protocol.cfg_get("metrics", "bootstrap_n")
    seed = protocol.cfg_get("metrics", "seed", default=0)

    df = load_results_df(gene)
    by_score = {}
    for score_col, ascending, label_name in SCORE_COLUMNS:
        if score_col not in df.columns or df[score_col].notna().sum() < 10:
            by_score[score_col] = {"error": "нет данных (< 10 значений или колонка отсутствует)"}
            continue
        sub = df[[score_col, "label"]].dropna().copy()
        work_col = score_col if ascending else f"_neg_{score_col}"
        if not ascending:
            sub[work_col] = -sub[score_col]
        else:
            work_col = score_col

        report = bc.run_test_a(sub, work_col, "label", alpha=alpha, n_bootstrap=n_bootstrap, n_permutations=100)
        by_score[score_col] = {
            "label": label_name, "sign_convention": "ascending (raw)" if ascending else "descending (negated for calc)",
            "n": int(len(sub)),
            "bedroc": report["bedroc_observed"],
            "bedroc_ci95_lo": report["bedroc_bootstrap_ci_95_lo"], "bedroc_ci95_hi": report["bedroc_bootstrap_ci_95_hi"],
            "auc_roc": report["auc_roc"],
            "EF_1%": report.get("EF_1%"), "EF_5%": report.get("EF_5%"), "EF_10%": report.get("EF_10%"),
        }

    result = {"n_total": len(df), "by_score": by_score, "bedroc_alpha": alpha}
    protocol.save_stage(gene, 3, {"test11_screening_power": result})

    print(f"\n=== Test 11 ({gene}): screening power ===")
    for score_col, r in by_score.items():
        if "error" in r:
            print(f"  {score_col}: {r['error']}"); continue
        print(f"  {r['label']:18s} n={r['n']:5d}  BEDROC={r['bedroc']:.3f} "
              f"[{r['bedroc_ci95_lo']:.3f},{r['bedroc_ci95_hi']:.3f}]  AUC={r['auc_roc']:.3f}  "
              f"EF1%={r['EF_1%']:.2f} EF5%={r['EF_5%']:.2f} EF10%={r['EF_10%']:.2f}  "
              f"[{r['sign_convention']}]")
    return result


def main():
    gene = sys.argv[1] if len(sys.argv) > 1 else "PIK3CA"
    force = "--force" in sys.argv
    run(gene, force)


if __name__ == "__main__":
    main()
