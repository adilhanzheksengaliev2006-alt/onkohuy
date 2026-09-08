"""
select_final_shortlist.py — Stage 3 воронки: сначала жёсткий порог по
токсикологии (<=2 красных флага ИЗ hERG/Ames/DILI/H-HT И чист от
PAINS/Brenk), потом топ-20% по консенсусу (Vina+gnina) среди выживших.

Источник: runs/test_a_<GENE>/admet_funnel_report.json (уже посчитан
run_admet_funnel.py). Не требует ни докинга, ни сети - чистая
локальная пересортировка/фильтрация уже готовых данных.

Использование:
    python select_final_shortlist.py [GENE] [MAX_RED_FLAGS] [TOP_FRACTION]
    по умолчанию: PIK3CA, 2, 0.20
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from module5_admet_filters import count_admet_red_flags, ADMET_RED_FLAG_ENDPOINTS  # noqa: E402

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LIVER_ENDPOINTS = {"DILI", "H-HT"}  # для примечания - не для порога


def run_dir(gene):
    return os.path.join(BASE_DIR, "runs", f"test_a_{gene}")


def main():
    gene = sys.argv[1] if len(sys.argv) > 1 else "PIK3CA"
    max_flags = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    top_fraction = float(sys.argv[3]) if len(sys.argv) > 3 else 0.20

    report_path = os.path.join(run_dir(gene), "admet_funnel_report.json")
    if not os.path.exists(report_path):
        print(f"ОШИБКА: нет {report_path} - сначала запусти run_admet_funnel.py"); sys.exit(1)
    with open(report_path, encoding="utf-8") as f:
        candidates = json.load(f)

    matched = [r for r in candidates if r.get("ADMET_matched")]
    print(f"[shortlist] кандидатов с посчитанным ADMET: {len(matched)}")

    survivors = []
    for r in matched:
        rf = count_admet_red_flags(r)
        if rf is None or rf > max_flags:
            continue
        if not r.get("pains_brenk_clean", False):
            continue
        r["_n_red_flags"] = rf
        r["_flagged_endpoints"] = [ep for ep in ADMET_RED_FLAG_ENDPOINTS
                                    if r.get(f"ADMET_{ep}") is not None
                                    and float(r[f"ADMET_{ep}"]) > ADMET_RED_FLAG_ENDPOINTS[ep]]
        r["_liver_only"] = bool(r["_flagged_endpoints"]) and set(r["_flagged_endpoints"]) <= LIVER_ENDPOINTS
        survivors.append(r)
    print(f"[shortlist] прошли жёсткий порог (<={max_flags} флага И PAINS/Brenk чист): {len(survivors)}")

    if not survivors:
        print("[shortlist] Никто не прошёл порог - список пуст. Попробуй увеличить MAX_RED_FLAGS."); return

    # консенсус-ранг (тот же принцип, что и в phase_funnel/run_admet_funnel.py)
    scored = [r for r in survivors
              if r.get("docking_score_kcal_mol") is not None and r.get("gnina_cnn_score") is not None]
    for i, r in enumerate(sorted(scored, key=lambda r: r["docking_score_kcal_mol"])):
        r["_vina_rank"] = i
    for i, r in enumerate(sorted(scored, key=lambda r: -r["gnina_cnn_score"])):
        r["_gnina_rank"] = i
    for r in survivors:
        r["_consensus_rank_sum"] = r["_vina_rank"] + r["_gnina_rank"] if "_vina_rank" in r else None
    survivors_scored = [r for r in survivors if r["_consensus_rank_sum"] is not None]
    survivors_scored.sort(key=lambda r: r["_consensus_rank_sum"])

    n_top = max(1, int(len(survivors_scored) * top_fraction))
    final_shortlist = survivors_scored[:n_top]
    print(f"[shortlist] топ-{top_fraction*100:.0f}% из выживших = {n_top} финалистов")

    out_path = os.path.join(run_dir(gene), "final_shortlist_stage3.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(final_shortlist, f, indent=2, ensure_ascii=False)

    n_active = sum(1 for r in final_shortlist if r["label"] == 1)
    n_liver_only = sum(1 for r in final_shortlist if r["_liver_only"])
    print(f"\n=== ФИНАЛИСТЫ Stage 3 ({gene}) ===")
    header = "{:>3} {:<16} {:>5} {:>7} {:>6} {:>4} {}".format("#", "chembl_id", "label", "vina", "gnina", "флаги", "какие")
    print(header)
    for i, r in enumerate(final_shortlist, 1):
        line = "{:>3} {:<16} {:>5} {:>7.2f} {:>6.3f} {:>5} {}".format(
            i, r["chembl_id"], r["label"], r["docking_score_kcal_mol"], r["gnina_cnn_score"],
            r["_n_red_flags"], r["_flagged_endpoints"])
        print(line)
    print(f"\nактивных (уже известных) среди финалистов: {n_active}/{len(final_shortlist)}")
    print(f"из них флаги только печёночные (DILI/H-HT, одна и та же проблема дважды): {n_liver_only}/{len(final_shortlist)}")
    print(f"\nСохранено: {out_path}")


if __name__ == "__main__":
    main()
