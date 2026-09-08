"""
Test 2 (Stage 0) — ligand-only baseline: если RandomForest, обученный
ТОЛЬКО на физхимических дескрипторах лиганда (без рецептора, без докинга),
уже отличает actives от decoys - это hidden bias в самом датасете
(Chen et al., PLOS ONE 2019).

КРИТИЧНО: сплит по Murcko-скэлдам (GroupKFold по scaffold), НЕ случайный
random split - иначе один и тот же скэлд может утечь между train/test и
завысить AUC искусственно.

Источник: runs/test_a_<GENE>/ligands.json.

Использование:
    python controls/test02_ligand_only_baseline.py GENE [--force]
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import protocol  # noqa: E402

FEATURES = ["mw", "logp", "tpsa", "hbd", "hba", "rotb", "heavy_atoms", "rings", "qed", "fraction_csp3"]


def compute_features_and_scaffold(smiles):
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Lipinski, QED, rdMolDescriptors
    from rdkit.Chem.Scaffolds import MurckoScaffold
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None
    feats = {
        "mw": Descriptors.MolWt(mol), "logp": Descriptors.MolLogP(mol),
        "tpsa": Descriptors.TPSA(mol), "hbd": Lipinski.NumHDonors(mol),
        "hba": Lipinski.NumHAcceptors(mol), "rotb": Descriptors.NumRotatableBonds(mol),
        "heavy_atoms": mol.GetNumHeavyAtoms(), "rings": rdMolDescriptors.CalcNumRings(mol),
        "qed": QED.qed(mol), "fraction_csp3": rdMolDescriptors.CalcFractionCSP3(mol),
    }
    try:
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
    except Exception:
        scaffold = None
    return feats, (scaffold or "NO_SCAFFOLD")


def load_ligands(gene):
    path = os.path.join(protocol.runs_dir(gene), "ligands.json")
    if not os.path.exists(path):
        print(f"ОШИБКА: нет {path}"); sys.exit(1)
    with open(path, encoding="utf-8") as f:
        return json.load(f)["ligands"]


def run(gene, force=False, n_folds=5, seed=None):
    if protocol.test_already_done(gene, 0, "test02_ligand_only_baseline", force):
        print(f"[test02] {gene}: уже посчитано, пропускаю. --force для пересчёта")
        return protocol.load_stage(gene, 0)["test02_ligand_only_baseline"]

    protocol.print_banner("test02", ["flags.ligand_only_auc_warn", "metrics.bedroc_alpha"])
    seed = seed if seed is not None else protocol.cfg_get("metrics", "seed", default=0)
    auc_warn = protocol.cfg_get("flags", "ligand_only_auc_warn")
    alpha = protocol.cfg_get("metrics", "bedroc_alpha")

    ligands = load_ligands(gene)
    rows = []
    for r in ligands:
        feats, scaffold = compute_features_and_scaffold(r["smiles"])
        if feats is None:
            continue
        row = dict(feats)
        row["label"] = r["label"]
        row["scaffold"] = scaffold
        rows.append(row)

    import numpy as np
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GroupKFold
    from rdkit.ML.Scoring import Scoring

    df = pd.DataFrame(rows)
    n_scaffolds = df["scaffold"].nunique()
    print(f"[test02] {gene}: {len(df)} лигандов, {n_scaffolds} уникальных Murcko-скэлдов, "
          f"actives={df['label'].sum()}, decoys={(df['label'] == 0).sum()}")

    n_folds_eff = min(n_folds, n_scaffolds)
    if n_folds_eff < 2:
        result = {"error": "недостаточно уникальных скэлдов для scaffold-split CV", "n_scaffolds": n_scaffolds}
        protocol.save_stage(gene, 0, {"test02_ligand_only_baseline": result})
        print(f"[test02] [warn] {result['error']}")
        return result

    gkf = GroupKFold(n_splits=n_folds_eff)
    X = df[FEATURES].values
    y = df["label"].values
    groups = df["scaffold"].values

    oof_scores = np.full(len(df), np.nan)
    n_folds_skipped = 0
    for train_idx, test_idx in gkf.split(X, y, groups):
        if len(np.unique(y[train_idx])) < 2:
            # весь один класс ушёл в train целиком - predict_proba вернёт 1
            # столбец вместо 2, а [:, 1] либо упадёт, либо тихо перепутает
            # классы. На таргетах с малым числом активных (не PIK3CA) это
            # реально может случиться - пропускаем фолд, а не падаем.
            n_folds_skipped += 1
            continue
        clf = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1)
        clf.fit(X[train_idx], y[train_idx])
        oof_scores[test_idx] = clf.predict_proba(X[test_idx])[:, 1]

    if n_folds_skipped:
        print(f"[test02] [warn] {n_folds_skipped}/{n_folds_eff} фолдов пропущено - "
              f"в train целиком ушёл один класс (мало активных на этот scaffold-сплит)")
    valid_mask = ~np.isnan(oof_scores)
    if valid_mask.sum() < len(df):
        df = df[valid_mask].reset_index(drop=True)
        oof_scores = oof_scores[valid_mask]
        y = df["label"].values

    df["_score"] = oof_scores
    # rdkit Scoring ожидает [[score, label], ...], sorted DESC по score (выше=лучше)
    ranked = df.sort_values("_score", ascending=False)[["_score", "label"]].values.tolist()
    bedroc = Scoring.CalcBEDROC(ranked, 1, alpha)
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(y, oof_scores)

    def ef_at(frac):
        return Scoring.CalcEnrichment(ranked, 1, [frac])[0]

    ef1 = ef_at(0.01)

    result = {
        "n": len(df), "n_scaffolds": int(n_scaffolds), "n_folds": n_folds_eff,
        "auc_scaffold_cv": float(auc), "bedroc_scaffold_cv": float(bedroc),
        "ef1pct_scaffold_cv": float(ef1),
        "auc_warn_threshold": auc_warn,
        "warn_ligand_only_predictive": bool(auc > auc_warn),
        "features_used": FEATURES,
        "citation": "Chen et al. PLOS ONE 2019 - hidden bias in DUD-E-style benchmarks",
    }
    protocol.save_stage(gene, 0, {"test02_ligand_only_baseline": result})

    print(f"\n=== Test 2 ({gene}): ligand-only baseline (scaffold-split {n_folds_eff}-fold CV) ===")
    print(f"  AUC-ROC: {auc:.3f} (порог предупреждения: {auc_warn})")
    print(f"  BEDROC(alpha={alpha}): {bedroc:.3f}, EF_1%: {ef1:.2f}")
    if result["warn_ligand_only_predictive"]:
        print(f"  [!] ligand-only baseline УЖЕ предсказывает label лучше случайного - "
              f"часть сигнала может быть hidden bias датасета, а не связыванием")
    return result


def main():
    gene = sys.argv[1] if len(sys.argv) > 1 else "PIK3CA"
    force = "--force" in sys.argv
    run(gene, force)


if __name__ == "__main__":
    main()
