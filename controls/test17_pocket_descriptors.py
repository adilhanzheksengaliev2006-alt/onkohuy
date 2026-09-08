"""
Test 17 (Stage 7) — дескрипторы кармана связывания через fpocket: одна
строка на мишень (карман, ближе всего к известному активному сайту -
уже определяется в run_fpocket.py по расстоянию центров).

Берёт volume/drug_score/hydrophobicity_score/flex/mean_asph_radius
напрямую из fpocket, плюс производные apolar/polar residue counts
(эвристическая классификация 20 аминокислот - не выдумка fpocket, но и
не единственно верная классификация, помечено явно).

Использование:
    python controls/test17_pocket_descriptors.py GENE [--force]
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import protocol  # noqa: E402
import run_fpocket as rf  # noqa: E402

APOLAR_RESIDUES = ["ala", "val", "leu", "ile", "met", "phe", "trp", "pro", "gly"]
POLAR_RESIDUES = ["ser", "thr", "asn", "gln", "cys", "tyr", "his", "asp", "glu", "lys", "arg"]


def run(gene, force=False):
    if protocol.test_already_done(gene, 7, "test17_pocket_descriptors", force):
        print(f"[test17] {gene}: уже посчитано, пропускаю. --force для пересчёта")
        return protocol.load_stage(gene, 7)["test17_pocket_descriptors"]

    protocol.print_banner("test17")

    fpocket_json = os.path.join(protocol.BASE_DIR, "runs", f"fpocket_{gene}.json")
    if not os.path.exists(fpocket_json):
        print(f"[test17] {gene}: нет {fpocket_json}, запускаю fpocket...")
        rf.run_fpocket(gene)

    with open(fpocket_json, encoding="utf-8") as f:
        data = json.load(f)

    matched = [p for p in data["pockets"] if p.get("matches_known_site")]
    if not matched:
        result = {"error": f"ни один карман fpocket не сопоставлен с известным активным сайтом для {gene}"}
        protocol.save_stage(gene, 7, {"test17_pocket_descriptors": result})
        print(f"[test17] [warn] {result['error']}")
        return result

    p = matched[0]
    apolar_count = sum(p.get(r, 0) for r in APOLAR_RESIDUES)
    polar_count = sum(p.get(r, 0) for r in POLAR_RESIDUES)

    result = {
        "gene": gene, "cav_id": p["cav_id"], "dist_to_known_site": p.get("dist_to_known_site"),
        "volume": p["volume"], "drug_score": p["drug_score"],
        "hydrophobicity_score": p["hydrophobicity_score"], "polarity_score": p["polarity_score"],
        "flex": p["flex"], "mean_asph_radius": p["mean_asph_radius"], "as_density": p["as_density"],
        "a0_apol_surface": p["a0_apol"], "a0_pol_surface": p["a0_pol"],
        "af_apol_surface": p["af_apol"], "af_pol_surface": p["af_pol"],
        "apolar_residue_count": apolar_count, "polar_residue_count": polar_count,
        "n_pockets_total": data["n_pockets"],
        "residue_classification_note": "эвристическая (apolar={ala,val,leu,ile,met,phe,trp,pro,gly}, остальное=polar) - не единственно верная",
    }
    protocol.save_stage(gene, 7, {"test17_pocket_descriptors": result})

    print(f"\n=== Test 17 ({gene}): pocket descriptors (fpocket, карман #{p['cav_id']}) ===")
    print(f"  volume={result['volume']:.1f}  drug_score={result['drug_score']:.3f}  "
          f"hydrophobicity={result['hydrophobicity_score']:.2f}  flex={result['flex']:.3f}")
    print(f"  apolar residues={apolar_count}  polar residues={polar_count}")
    return result


def main():
    gene = sys.argv[1] if len(sys.argv) > 1 else "PIK3CA"
    force = "--force" in sys.argv
    run(gene, force)


if __name__ == "__main__":
    main()
