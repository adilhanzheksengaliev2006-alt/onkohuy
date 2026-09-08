"""
run_admet_funnel.py — прогоняет топ-N% (после funnel-доуточнения) через
полную цепочку фильтров: PAINS/Brenk -> SA score -> ADMETlab 2.0 (полный
ADME + токсикология) -> итоговое композитное ранжирование.

Источник кандидатов: runs/test_a_<GENE>/funnel_results.jsonl (уже
передокованные с exhaustiveness=32 + свежий gnina - см. run_test_a.py
funnel). Если funnel не запускался - падает с понятной ошибкой.

PAINS/Brenk и SA score - чистый RDKit, локально, без сети, быстро.
ADMETlab - внешний бесплатный сервис (admetmesh.scbdd.com, версия 2.0), сетевая
фаза, батчами по 25 с паузами (не грузим чужой сервис) - для ~270
кандидатов (20% от 1351 для PIK3CA) это порядка 10-15 минут.

Использование:
    python run_admet_funnel.py [GENE]
    по умолчанию: PIK3CA
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from module5_admet_filters import (  # noqa: E402
    compute_sa_score, fetch_admet_batch, rank_candidates, count_admet_red_flags,
    Admet5Error, SA_REASONABLE_MAX,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def build_toxicity_catalog():
    """PAINS/Brenk через RDKit FilterCatalog - продублировано напрямую из
    module_generative/iterative_finetune_loop.py, чтобы не тащить его
    тяжёлые MolGPT-зависимости (torch, load_molgpt) ради двух функций,
    которым они не нужны."""
    from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
    params = FilterCatalogParams()
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.BRENK)
    return FilterCatalog(params)


def passes_toxicity_filter(smiles, catalog):
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False, []
    matches = catalog.GetMatches(mol)
    return len(matches) == 0, [m.GetDescription() for m in matches]


def run_dir(gene):
    return os.path.join(BASE_DIR, "runs", f"test_a_{gene}")


def load_funnel_candidates(gene):
    funnel_path = os.path.join(run_dir(gene), "funnel_results.jsonl")
    if not os.path.exists(funnel_path):
        print(f"ОШИБКА: нет {funnel_path} - сначала запусти 'python run_test_a.py funnel {gene} ...'")
        sys.exit(1)
    by_id = {}
    with open(funnel_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            cid = rec["chembl_id"]
            prev = by_id.get(cid)
            if prev is None or rec.get("docking_score_kcal_mol") is not None:
                by_id[cid] = rec
    candidates = [r for r in by_id.values() if r.get("docking_score_kcal_mol") is not None]
    return candidates


def main():
    gene = sys.argv[1] if len(sys.argv) > 1 else "PIK3CA"
    candidates = load_funnel_candidates(gene)
    print(f"[admet] кандидатов из funnel ({gene}): {len(candidates)}")

    print("[admet] PAINS/Brenk (локально, RDKit)...")
    catalog = build_toxicity_catalog()
    for r in candidates:
        ok, hits = passes_toxicity_filter(r["smiles"], catalog)
        r["pains_brenk_clean"] = ok
        r["pains_brenk_hits"] = hits
    n_clean = sum(1 for r in candidates if r["pains_brenk_clean"])
    print(f"[admet] чисты от PAINS/Brenk: {n_clean}/{len(candidates)}")

    print("[admet] SA score (локально, RDKit Contrib)...")
    for r in candidates:
        r["SA_score"] = compute_sa_score(r["smiles"])
    n_easy = sum(1 for r in candidates if r.get("SA_score") is not None and r["SA_score"] <= SA_REASONABLE_MAX)
    print(f"[admet] SA score <= {SA_REASONABLE_MAX} (разумно синтезируемые): {n_easy}/{len(candidates)}")

    print(f"[admet] ADMETlab 2.0 (внешний сервис, {len(candidates)} молекул, батчами по 25 - минут 10-15)...")
    try:
        smiles_list = [r["smiles"] for r in candidates]
        admet_df = fetch_admet_batch(smiles_list)
    except Admet5Error as e:
        print(f"[admet] ОШИБКА ADMETlab: {e}")
        print("[admet] продолжаю БЕЗ ADMET-данных (PAINS/Brenk и SA score уже посчитаны и сохранятся)")
        admet_df = None

    if admet_df is not None:
        admet_by_smiles = admet_df.to_dict(orient="index")
        for r in candidates:
            row = admet_by_smiles.get(r["smiles"])
            if row:
                for k, v in row.items():
                    if k not in ("SMILES",):
                        r[k] = v
            else:
                r["ADMET_matched"] = False

    # ВАЖНО: rank_candidates() по умолчанию сортирует ПЕРВЫМ приоритетом
    # по сырому Vina-скору (docking_col) - ровно то, от чего мы уходили
    # через консенсус в phase_funnel (run_test_a.py). Если передать туда
    # docking_score_kcal_mol напрямую, весь смысл консенсуса теряется на
    # финальном шаге (обнаружено: топ-15 по чистому Vina дал всего 2
    # активных из 15, та же болезнь, что и раньше). Пересчитываем тот же
    # консенсус-ранг (сумма рангов Vina+gnina) здесь и используем его как
    # primary sort key вместо голого Vina.
    scored_for_consensus = [r for r in candidates
                             if r.get("docking_score_kcal_mol") is not None and r.get("gnina_cnn_score") is not None]
    for i, r in enumerate(sorted(scored_for_consensus, key=lambda r: r["docking_score_kcal_mol"])):
        r["_vina_rank"] = i
    for i, r in enumerate(sorted(scored_for_consensus, key=lambda r: -r["gnina_cnn_score"])):
        r["_gnina_rank"] = i
    for r in candidates:
        r["_consensus_rank_sum"] = r["_vina_rank"] + r["_gnina_rank"] if "_vina_rank" in r else None

    import pandas as pd
    df = pd.DataFrame(candidates)
    df_ranked = rank_candidates(df, docking_col="_consensus_rank_sum")

    out_path = os.path.join(run_dir(gene), "admet_funnel_report.json")
    df_ranked.to_json(out_path, orient="records", indent=2, force_ascii=False)
    csv_path = os.path.join(run_dir(gene), "admet_funnel_report.csv")
    df_ranked.to_csv(csv_path, index=False)

    print(f"\n=== ИТОГОВЫЙ РАНЖИРОВАННЫЙ СПИСОК ({gene}) ===")
    cols_to_show = ["chembl_id", "label", "docking_score_kcal_mol", "gnina_cnn_score",
                     "_consensus_rank_sum", "pains_brenk_clean", "SA_score"]
    if admet_df is not None:
        cols_to_show.append("ADMET_matched")
    print(df_ranked[[c for c in cols_to_show if c in df_ranked.columns]].head(20).to_string(index=False))

    print(f"\nПолный отчёт: {out_path}")
    print(f"CSV: {csv_path}")


if __name__ == "__main__":
    main()
