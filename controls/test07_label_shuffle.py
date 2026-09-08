"""
Test 7 (Stage 3) — label-shuffle null-модель: перемешиваем МЕТКИ (не
скоры) N раз, считаем BEDROC/EF1%/AUC на каждом перемешивании -> если
наблюдаемый BEDROC не сильно отличается от этого случайного
распределения, реальный сигнал под вопросом.

Переиспользует bedroc_calibration.empirical_random_baseline (уже
делает ровно перемешивание меток). Здесь добавляем: percentile 95/99,
гистограмму с вертикальной линией на наблюдаемом значении, явное
сравнение с референсом (в лит-ре при соотношении 1:31 и alpha=20
случайный BEDROC ~0.068).

Источник: runs/test_a_<GENE>/results.jsonl (нужен уже посчитанный докинг).

Использование:
    python controls/test07_label_shuffle.py GENE [--force]
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import protocol  # noqa: E402
import bedroc_calibration as bc  # noqa: E402

REFERENCE_RATIO = 31  # 1 active : 31 decoys
REFERENCE_BEDROC_ALPHA20 = 0.068


def load_results_df(gene):
    import pandas as pd
    path = os.path.join(protocol.runs_dir(gene), "results.jsonl")
    if not os.path.exists(path):
        print(f"ОШИБКА: нет {path} - нужен докинг ('python run_test_a.py dock {gene} ...')")
        sys.exit(1)
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def run(gene, force=False):
    if protocol.test_already_done(gene, 3, "test07_label_shuffle", force):
        print(f"[test07] {gene}: уже посчитано, пропускаю. --force для пересчёта")
        return protocol.load_stage(gene, 3)["test07_label_shuffle"]

    protocol.print_banner("test07", ["metrics.bedroc_alpha", "metrics.permutation_n"])
    alpha = protocol.cfg_get("metrics", "bedroc_alpha")
    n_perm = protocol.cfg_get("metrics", "permutation_n")
    seed = protocol.cfg_get("metrics", "seed", default=0)

    df = load_results_df(gene)
    df = df.dropna(subset=["docking_score_kcal_mol"])
    n_active = int((df["label"] == 1).sum())
    n_decoy = int((df["label"] == 0).sum())
    ratio = n_decoy / n_active if n_active else None

    import numpy as np
    from rdkit.ML.Scoring import Scoring

    def bedroc_for(score_col, ascending):
        sub = df[[score_col, "label"]].dropna().copy()
        sub = sub.sort_values(score_col, ascending=ascending)
        matrix = sub.values.tolist()
        return Scoring.CalcBEDROC(matrix, 1, alpha)

    results_by_score = {}
    for score_col, ascending in [("docking_score_kcal_mol", True)]:
        observed = bedroc_for(score_col, ascending)
        null_dist = bc.empirical_random_baseline(df, score_col, "label", alpha, n_perm, seed=seed)
        p_value = bc.permutation_p_value(observed, null_dist)
        pct95 = float(np.percentile(null_dist, 95))
        pct99 = float(np.percentile(null_dist, 99))

        plot_path = None
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.hist(null_dist, bins=40, alpha=0.7, color="tab:blue", label=f"{n_perm} label-shuffles")
            ax.axvline(observed, color="tab:red", linewidth=2, label=f"наблюдаемый BEDROC={observed:.3f}")
            ax.axvline(REFERENCE_BEDROC_ALPHA20, color="gray", linestyle="--",
                       label=f"референс 1:{REFERENCE_RATIO}, alpha=20 -> {REFERENCE_BEDROC_ALPHA20}")
            ax.set_xlabel("BEDROC"); ax.set_ylabel("частота"); ax.legend()
            ax.set_title(f"Test 7 label-shuffle null - {gene} ({score_col})")
            plot_path = os.path.join(protocol.results_dir(gene), f"test07_shuffle_{score_col}.png")
            fig.tight_layout(); fig.savefig(plot_path, dpi=110); plt.close(fig)
        except Exception as e:
            print(f"[test07] [warn] не удалось построить гистограмму: {e}")

        results_by_score[score_col] = {
            "bedroc_observed": float(observed),
            "bedroc_shuffled_mean": float(null_dist.mean()), "bedroc_shuffled_std": float(null_dist.std()),
            "bedroc_shuffled_p95": pct95, "bedroc_shuffled_p99": pct99,
            "permutation_p_value": float(p_value), "n_permutations": n_perm,
            "plot": plot_path,
        }

    result = {
        "n_active": n_active, "n_decoy": n_decoy, "ratio_decoy_per_active": ratio,
        "reference_ratio": REFERENCE_RATIO, "reference_bedroc_alpha20": REFERENCE_BEDROC_ALPHA20,
        "by_score": results_by_score,
    }
    protocol.save_stage(gene, 3, {"test07_label_shuffle": result})

    print(f"\n=== Test 7 ({gene}): label-shuffle null-модель ===")
    print(f"  actives={n_active}, decoys={n_decoy}, ratio 1:{ratio:.1f}" if ratio else f"  actives={n_active}, decoys={n_decoy}")
    for score_col, r in results_by_score.items():
        print(f"  [{score_col}] наблюдаемый BEDROC={r['bedroc_observed']:.3f}, "
              f"случайный: mean={r['bedroc_shuffled_mean']:.3f} p95={r['bedroc_shuffled_p95']:.3f} p99={r['bedroc_shuffled_p99']:.3f}, "
              f"perm p-value={r['permutation_p_value']:.4f}")
    return result


def main():
    gene = sys.argv[1] if len(sys.argv) > 1 else "PIK3CA"
    force = "--force" in sys.argv
    run(gene, force)


if __name__ == "__main__":
    main()
