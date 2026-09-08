"""
Post-hoc sensitivity analysis (НЕ часть исходного протокола - запускается
ПОСЛЕ того, как Test 1 показал перекос по logP в декоях, конкретно для
BRAF): строит 1:N property-matched (MW±25, logP±0.5) подвыборку декоев
ИЗ УЖЕ ЗАДОКОВАННЫХ данных (без нового докинга) и пересчитывает тесты
2/3/11/13 на ней, рядом с полным набором - разница между full и balanced
сама по себе результат: сколько "обогащения" создал перекос по logP.

Использование:
    python controls/balanced_sensitivity.py GENE [--n-per-active 6] [--force]
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import protocol  # noqa: E402

MW_TOL = 25
LOGP_TOL = 0.5


def load_scored_ligands(gene):
    from rdkit import Chem
    from rdkit.Chem import Descriptors

    lig_path = os.path.join(protocol.runs_dir(gene), "ligands.json")
    res_path = os.path.join(protocol.runs_dir(gene), "results.jsonl")
    with open(lig_path, encoding="utf-8") as f:
        ligs = {r["chembl_id"]: r for r in json.load(f)["ligands"]}
    rows = []
    with open(res_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("docking_score_kcal_mol") is None:
                continue
            lig = ligs.get(r["chembl_id"])
            if lig is None:
                continue
            mol = Chem.MolFromSmiles(r["smiles"])
            if mol is None:
                continue
            r["mw"] = Descriptors.MolWt(mol)
            r["logp"] = Descriptors.MolLogP(mol)
            r["gnina_cnn_score"] = r.get("gnina_cnn_score")
            rows.append(r)
    return rows


def build_balanced_subsample(actives, decoys, n_per_active, seed):
    """Жадный 1:N матчинг: для каждого активного (в случайном порядке,
    фикс. seed) - до n_per_active ЕЩЁ НЕ использованных декоев в пределах
    MW+-25/logP+-0.5. Не идеальная оптимизация (не гарантирует глобальный
    максимум N), но простая, воспроизводимая, без повторного
    использования декоя между активными."""
    import random
    rng = random.Random(seed)
    order = list(range(len(actives)))
    rng.shuffle(order)

    used = set()
    matched_decoys = []
    n_fully_matched = 0
    for idx in order:
        a = actives[idx]
        eligible = [d for d in decoys if d["chembl_id"] not in used
                    and abs(d["mw"] - a["mw"]) <= MW_TOL
                    and abs(d["logp"] - a["logp"]) <= LOGP_TOL]
        rng.shuffle(eligible)
        picked = eligible[:n_per_active]
        if len(picked) == n_per_active:
            n_fully_matched += 1
        for d in picked:
            used.add(d["chembl_id"])
            matched_decoys.append(d)
    return matched_decoys, n_fully_matched


def run(gene, n_per_active=6, force=False):
    out_path = os.path.join(protocol.results_dir(gene), "balanced_sensitivity.json")
    if os.path.exists(out_path) and not force:
        print(f"[balanced] {gene}: уже посчитано ({out_path}), пропускаю. --force для пересчёта")
        with open(out_path, encoding="utf-8") as f:
            return json.load(f)

    seed = protocol.cfg_get("metrics", "seed", default=0)
    rows = load_scored_ligands(gene)
    actives = [r for r in rows if r["label"] == 1]
    decoys = [r for r in rows if r["label"] == 0]
    print(f"[balanced] {gene}: {len(actives)} активных, {len(decoys)} декоев с посчитанными MW/logP")

    matched_decoys, n_fully_matched = build_balanced_subsample(actives, decoys, n_per_active, seed)
    print(f"[balanced] полностью подобрано (1:{n_per_active}): {n_fully_matched}/{len(actives)} активных, "
          f"итого {len(matched_decoys)} декоев в сбалансированной подвыборке")

    balanced_set = actives + matched_decoys
    import numpy as np
    import pandas as pd
    from scipy.stats import mannwhitneyu

    def cliffs_delta(x, y):
        x = np.asarray(x); y = np.asarray(y)
        gt = sum((xi > y).sum() for xi in x)
        lt = sum((xi < y).sum() for xi in x)
        n = len(x) * len(y)
        return (gt - lt) / n if n else 0.0

    a_mw = [a["mw"] for a in actives]; d_mw = [d["mw"] for d in matched_decoys]
    a_lp = [a["logp"] for a in actives]; d_lp = [d["logp"] for d in matched_decoys]
    mw_delta = cliffs_delta(a_mw, d_mw)
    logp_delta = cliffs_delta(a_lp, d_lp)
    print(f"[balanced] проверка баланса: Cliff's delta MW={mw_delta:+.3f}, logP={logp_delta:+.3f} "
          f"(должно быть близко к 0, было logP=+0.659 до балансировки)")

    from rdkit.ML.Scoring import Scoring

    def bedroc_ef(score_col, ascending, alpha=20.0):
        df = pd.DataFrame(balanced_set).dropna(subset=[score_col])
        if len(df) < 5:
            return None, None, len(df)
        df = df.sort_values(score_col, ascending=ascending)
        matrix = df[[score_col, "label"]].values.tolist()
        bedroc = Scoring.CalcBEDROC(matrix, 1, alpha)
        ef1 = Scoring.CalcEnrichment(matrix, 1, [0.01])[0]
        return bedroc, ef1, len(df)

    vina_bedroc, vina_ef1, vina_n = bedroc_ef("docking_score_kcal_mol", ascending=True)
    gnina_scored = [r for r in balanced_set if r.get("gnina_cnn_score") is not None]
    gnina_bedroc = gnina_ef1 = gnina_n = None
    if len(gnina_scored) >= 10:
        for r in gnina_scored:
            r["_neg_gnina"] = -r["gnina_cnn_score"]
        df_g = pd.DataFrame(gnina_scored).sort_values("_neg_gnina", ascending=True)
        matrix_g = df_g[["_neg_gnina", "label"]].values.tolist()
        gnina_bedroc = Scoring.CalcBEDROC(matrix_g, 1, 20.0)
        gnina_ef1 = Scoring.CalcEnrichment(matrix_g, 1, [0.01])[0]
        gnina_n = len(df_g)

    # Test 13-стиль: R^2(score ~ heavy_atoms) на сбалансированном наборе
    from rdkit import Chem
    for r in balanced_set:
        mol = Chem.MolFromSmiles(r["smiles"])
        r["heavy_atoms"] = mol.GetNumHeavyAtoms() if mol else None
    import statsmodels.api as sm
    df_r2 = pd.DataFrame(balanced_set).dropna(subset=["docking_score_kcal_mol", "heavy_atoms"])
    X = sm.add_constant(df_r2["heavy_atoms"].values.astype(float))
    model = sm.OLS(df_r2["docking_score_kcal_mol"].values.astype(float), X).fit()
    r2_balanced = float(model.rsquared)

    result = {
        "gene": gene, "method": "post_hoc_sensitivity_after_test1_skew",
        "n_per_active_target": n_per_active, "n_actives": len(actives),
        "n_fully_matched_actives": n_fully_matched, "n_decoys_balanced": len(matched_decoys),
        "balance_check": {"mw_cliffs_delta": mw_delta, "logp_cliffs_delta": logp_delta},
        "test11_balanced": {
            "vina": {"bedroc": vina_bedroc, "ef1pct": vina_ef1, "n": vina_n},
            "gnina_cnn_score": {"bedroc": gnina_bedroc, "ef1pct": gnina_ef1, "n": gnina_n},
        },
        "test13_balanced": {"r2_vina_heavy_atoms": r2_balanced, "n": len(df_r2)},
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\n=== Balanced sensitivity ({gene}) ===")
    print(f"  Vina BEDROC: balanced={vina_bedroc}")
    print(f"  gnina BEDROC: balanced={gnina_bedroc}")
    print(f"  Test13 R^2 (balanced): {r2_balanced:.3f}")
    print(f"  Сохранено: {out_path}")
    return result


def main():
    gene = sys.argv[1] if len(sys.argv) > 1 else "BRAF"
    n_per_active = 6
    for i, a in enumerate(sys.argv):
        if a == "--n-per-active" and i + 1 < len(sys.argv):
            n_per_active = int(sys.argv[i + 1])
    force = "--force" in sys.argv
    run(gene, n_per_active, force)


if __name__ == "__main__":
    main()
