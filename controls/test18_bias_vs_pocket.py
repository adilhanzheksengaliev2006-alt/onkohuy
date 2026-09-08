"""
Test 18 (Stage 7) — ГЛАВНЫЙ ВЫВОД ВСЕЙ РАБОТЫ: регрессия величины
размерного смещения (R^2 из Test 13, all_combined, Vina) на геометрию
кармана (Test 17) + ковариаты (AVE bias из Test 3, ligand-only AUC из
Test 2, Cliff's delta по heavy_atoms из Test 1).

Собирает по ВСЕМ мишеням, у которых есть и stage_3.test13_size_bias, и
stage_7.test17_pocket_descriptors, строку в results/summary.csv, затем
считает корреляции/множественную регрессию.

ЧЕСТНО про размер выборки: при малом числе мишеней (n<5-8) полноценная
множественная регрессия ненадёжна - тогда считаем только простые
попарные корреляции по каждому дескриптору кармана отдельно, и явно
печатаем n мишеней, чтобы не создавать иллюзию статистической мощности.

Использование:
    python controls/test18_bias_vs_pocket.py [--force]
"""
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import protocol  # noqa: E402

POCKET_DESCRIPTORS = ["volume", "drug_score", "hydrophobicity_score", "polarity_score",
                       "flex", "mean_asph_radius", "as_density", "apolar_residue_count", "polar_residue_count"]


def discover_genes():
    """Список генов берём из confirmed_structures.json (авторитетный реестр
    мишеней пайплайна), а НЕ сканированием подпапок results/ - там же лежат
    посторонние директории (results/metrics/ из vina_variance.py, и т.п.),
    которые не являются мишенями и ломали бы build_summary_row без этой
    развязки."""
    confirmed_path = os.path.join(protocol.BASE_DIR, "confirmed_structures.json")
    with open(confirmed_path, encoding="utf-8") as f:
        return sorted(json.load(f).keys())


def build_summary_row(gene):
    s3 = protocol.load_stage(gene, 3) or {}
    s7 = protocol.load_stage(gene, 7) or {}
    s0 = protocol.load_stage(gene, 0) or {}

    test13 = s3.get("test13_size_bias")
    test17 = s7.get("test17_pocket_descriptors")
    if not test13 or not test17:
        return None
    if "error" in test17 or "by_score" not in test13:
        return None

    vina_split = test13["by_score"].get("docking_score_kcal_mol", {}).get("splits", {}).get("all_combined")
    if not vina_split or "r2" not in vina_split:
        return None

    row = {"gene": gene, "bias_r2_vina_all_combined": vina_split["r2"]}
    for key in POCKET_DESCRIPTORS:
        row[f"pocket_{key}"] = test17.get(key)

    test03 = s0.get("test03_ave_bias")
    row["ave_bias"] = test03.get("ave_bias") if test03 and "ave_bias" in test03 else None
    test02 = s0.get("test02_ligand_only_baseline")
    row["ligand_only_auc"] = test02.get("auc_scaffold_cv") if test02 and "auc_scaffold_cv" in test02 else None
    test01 = s0.get("test01_dataset_bias")
    row["cliffs_delta_heavy_atoms"] = (test01.get("per_descriptor", {}).get("heavy_atoms", {}).get("cliffs_delta")
                                        if test01 else None)
    return row


def write_summary_csv(rows):
    path = os.path.join(protocol.BASE_DIR, "results", "summary.csv")
    if not rows:
        return path
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def run(force=False):
    protocol.print_banner("test18", ["test18.min_targets", "test18.primary_descriptor"])
    cfg = protocol.cfg_get("test18") or {}
    min_targets = cfg.get("min_targets", 8)
    primary = cfg.get("primary_descriptor", "volume")
    secondary = cfg.get("secondary_descriptor")
    exploratory = cfg.get("exploratory", [k for k in POCKET_DESCRIPTORS if k not in (primary, secondary)])
    expected_sign = cfg.get("expected_sign")

    genes = discover_genes()
    rows = [r for r in (build_summary_row(g) for g in genes) if r is not None]
    print(f"[test18] мишеней с полными данными (Test 13 + Test 17): {len(rows)} из {len(genes)} в confirmed_structures.json")

    summary_path = write_summary_csv(rows)
    print(f"[test18] results/summary.csv записан: {summary_path}")

    if len(rows) < min_targets:
        result = {
            "n_targets": len(rows), "targets": [r["gene"] for r in rows],
            "status": "smoke_test", "do_not_report": True,
            "error": f"n={len(rows)} мишеней < test18.min_targets={min_targets} - результат НЕ репрезентативен, "
                     f"НЕ переносить ни в одну сводную таблицу/отчёт (см. status/do_not_report). "
                     f"Числа сохранены только для диагностики кода.",
        }
        print(f"[test18] {result['error']}")
    else:
        import pandas as pd
        from scipy.stats import pearsonr
        df = pd.DataFrame(rows)

        def _corr(key):
            col = f"pocket_{key}"
            sub = df[[col, "bias_r2_vina_all_combined"]].dropna()
            if len(sub) < 3:
                return {"error": f"n={len(sub)} < 3"}
            corr, p_value = pearsonr(sub[col], sub["bias_r2_vina_all_combined"])
            return {"n": int(len(sub)), "pearson_r": float(corr), "p_value": float(p_value)}

        primary_result = _corr(primary)
        if expected_sign and "pearson_r" in primary_result:
            observed_sign = "positive" if primary_result["pearson_r"] > 0 else "negative"
            primary_result["matches_expected_sign"] = (observed_sign == expected_sign)
        secondary_result = _corr(secondary) if secondary else None

        exploratory_raw = {key: _corr(key) for key in exploratory}
        # Holm-коррекция ТОЛЬКО по exploratory - primary/secondary зафиксированы
        # заранее (protocol.yaml) и не участвуют в множественном сравнении.
        method = cfg.get("multiple_testing", "holm")
        valid_keys = [k for k, v in exploratory_raw.items() if "p_value" in v]
        if valid_keys:
            from statsmodels.stats.multitest import multipletests
            pvals = [exploratory_raw[k]["p_value"] for k in valid_keys]
            reject, p_adj, _, _ = multipletests(pvals, method=method)
            for k, p_corrected, rej in zip(valid_keys, p_adj, reject):
                exploratory_raw[k]["p_value_adjusted"] = float(p_corrected)
                exploratory_raw[k]["significant_after_correction"] = bool(rej)

        multi_regression = None
        if len(rows) >= 8:
            import statsmodels.api as sm
            X = sm.add_constant(df[[f"pocket_{k}" for k in POCKET_DESCRIPTORS]].dropna())
            y = df.loc[X.index, "bias_r2_vina_all_combined"]
            model = sm.OLS(y, X).fit()
            multi_regression = {"n": int(len(X)), "r2": float(model.rsquared),
                                 "coefficients": {k: float(v) for k, v in model.params.items()},
                                 "p_values": {k: float(v) for k, v in model.pvalues.items()}}

        result = {"n_targets": len(rows), "targets": [r["gene"] for r in rows], "status": "final",
                  "primary_descriptor": primary, "primary_result": primary_result,
                  "secondary_descriptor": secondary, "secondary_result": secondary_result,
                  "exploratory_correction_method": method, "exploratory": exploratory_raw,
                  "multiple_regression": multi_regression}

    # НЕ через protocol.save_stage(gene, ...) - это создало бы results/<fake_gene>/
    # директорию, которую discover_genes() (или что-то ещё, сканирующее
    # results/) могло бы принять за настоящую мишень. Это сводный
    # межмишеневый результат - ему место рядом с summary.csv, а не внутри
    # per-gene структуры.
    summary_json_path = os.path.join(protocol.BASE_DIR, "results", "test18_bias_vs_pocket_summary.json")
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"[test18] сводный результат: {summary_json_path}")

    print(f"\n=== Test 18: bias magnitude ~ pocket geometry (n={result['n_targets']} мишеней, status={result['status']}) ===")
    if result["status"] == "final":
        pr = result["primary_result"]
        if "error" not in pr:
            sign_note = ""
            if "matches_expected_sign" in pr:
                sign_note = " [ЗНАК ПО ПЛАНУ]" if pr["matches_expected_sign"] else " [!] знак противоположен ожидаемому"
            print(f"  PRIMARY  {result['primary_descriptor']:25s} r={pr['pearson_r']:+.3f} (p={pr['p_value']:.3f}, n={pr['n']}){sign_note}")
        else:
            print(f"  PRIMARY  {result['primary_descriptor']}: {pr['error']}")
        if result["secondary_result"]:
            sr = result["secondary_result"]
            if "error" not in sr:
                print(f"  SECONDARY {result['secondary_descriptor']:24s} r={sr['pearson_r']:+.3f} (p={sr['p_value']:.3f}, n={sr['n']})")
        print(f"  --- exploratory (скорректировано методом {result['exploratory_correction_method']}) ---")
        for key, r in result["exploratory"].items():
            if "error" in r:
                print(f"  {key}: {r['error']}"); continue
            sig = " *" if r.get("significant_after_correction") else ""
            print(f"  {key:25s} r={r['pearson_r']:+.3f} (p={r['p_value']:.3f} -> p_adj={r.get('p_value_adjusted'):.3f}, n={r['n']}){sig}")
    return result


def main():
    force = "--force" in sys.argv
    run(force)


if __name__ == "__main__":
    main()
