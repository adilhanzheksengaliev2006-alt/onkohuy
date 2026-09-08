"""
Test 8 (Stage 5, ДОРОГОЙ, ТОЛЬКО ПО ЯВНОЙ КОМАНДЕ) — receptor-decoy site:
докуем ВСЕ активные + СЛУЧАЙНУЮ подвыборку декоев (НЕ отобранных по
скору в реальном сайте) в ЗАВЕДОМО НЕПРАВИЛЬНЫЙ карман той же структуры
(2-й по drug_score карман fpocket). Сравниваем BEDROC этого ЖЕ набора в
обоих карманах напрямую.

ВАЖНО - была реальная методологическая дыра, найденная и исправленная:
раньше сюда шёл топ-N% по консенсус-рангу Vina+gnina, т.е. набор,
ОТОБРАННЫЙ по скору в ПРАВИЛЬНОМ кармане - корреляция с тем, что тест
затем измеряет в НЕПРАВИЛЬНОМ. Свойства, дающие хороший скор В ЛЮБОМ
кармане (размер, гибкость), автоматически завышали BEDROC в неправильном
кармане тоже - тест не мог не найти "сигнал", вне зависимости от
специфичности связывания. Теперь набор фиксирован ДО докинга в любой
карман (все активные + n_decoys случайных, config seed) - ни один скор
не участвует в отборе.

НЕ запускается по умолчанию (test08_decoy_site.enabled_by_default: false
в config/protocol.yaml) - требует --confirm И явного списка мишеней в
--genes, как того требует протокол.

Источник кармана: runs/fpocket_<GENE>.json (Test 17).
Источник набора и скоров в реальном сайте: runs/test_a_<GENE>/results.jsonl
(уже посчитанный основной докинг, Stage 3).

Использование:
    python controls/test08_decoy_site.py --confirm --genes PIK3CA,ESR1 [--n-decoys 850] [--force]
"""
import argparse
import json
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import protocol  # noqa: E402
from dock_existing_candidates import prepare_ligand_pdbqt, run_vina  # noqa: E402
from benchmark_parallel_docking import cpu_per_worker  # noqa: E402


def pick_decoy_pocket(gene):
    fpocket_json = os.path.join(protocol.BASE_DIR, "runs", f"fpocket_{gene}.json")
    if not os.path.exists(fpocket_json):
        return None, f"нет {fpocket_json} - сначала запусти controls/test17_pocket_descriptors.py {gene}"
    with open(fpocket_json, encoding="utf-8") as f:
        data = json.load(f)
    known_cav = data.get("pocket_matching_known_site")
    candidates = [p for p in data["pockets"] if int(p["cav_id"]) != known_cav]
    if not candidates:
        return None, "нет других карманов, кроме известного активного сайта"
    candidates.sort(key=lambda p: -p.get("drug_score", 0))
    return candidates[0], None


def load_unbiased_ligand_set(gene, n_decoys, seed):
    """ВСЕ активные + n_decoys СЛУЧАЙНЫХ декоев - набор фиксируется ДО
    докинга в любой карман, ни один скор (ни реального, ни ложного
    сайта) не участвует в отборе. Так BEDROC в обоих карманах сравним
    напрямую: разница объясняется только карманом, не составом набора."""
    results_path = os.path.join(protocol.runs_dir(gene), "results.jsonl")
    if not os.path.exists(results_path):
        return None, f"нет {results_path} - сначала нужен полный докинг (Stage 3) для {gene}"

    rows = []
    with open(results_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    scored = [r for r in rows if r.get("docking_score_kcal_mol") is not None]
    actives = [r for r in scored if r["label"] == 1]
    decoys = [r for r in scored if r["label"] == 0]
    if len(actives) < 5:
        return None, f"только {len(actives)} активных с реальным скором - недостаточно"

    import random
    rng = random.Random(seed)
    n_decoys_eff = min(n_decoys, len(decoys))
    sampled_decoys = rng.sample(decoys, n_decoys_eff)
    ligand_set = actives + sampled_decoys
    print(f"[test08] набор ФИКСИРОВАН до докинга: {len(actives)} активных + {n_decoys_eff} случайных "
          f"декоев (seed={seed}) = {len(ligand_set)}, Ra={len(actives)/len(ligand_set):.3f}")
    return ligand_set, None


def _dock_decoy_site_task(args):
    # Целиком в try/except: одна молекула, упавшая непредвиденно, не должна
    # ронять весь multiprocessing.Pool (тот же класс бага, найденный и
    # исправленный в test09_seed_variance.py - см. комментарий там).
    chembl_id, smiles, label, receptor_pdbqt, center, box_size, cpu = args
    try:
        with tempfile.TemporaryDirectory() as workdir:
            ligand_pdbqt = os.path.join(workdir, "lig.pdbqt")
            if not prepare_ligand_pdbqt(smiles, ligand_pdbqt):
                return chembl_id, label, None
            out_pdbqt = os.path.join(workdir, "out.pdbqt")
            score = run_vina(receptor_pdbqt, ligand_pdbqt, center, box_size, out_pdbqt,
                              exhaustiveness=8, timeout=300, cpu=cpu)
            return chembl_id, label, score
    except Exception:
        return chembl_id, label, None


def _bedroc_ef(rows, score_col, label_col="label", alpha=None):
    import pandas as pd
    from rdkit.ML.Scoring import Scoring
    alpha = alpha or protocol.cfg_get("metrics", "bedroc_alpha")
    df = pd.DataFrame(rows).dropna(subset=[score_col])
    if len(df) < 5:
        return None, None, len(df)
    df = df.sort_values(score_col, ascending=True)
    matrix = df[[score_col, label_col]].values.tolist()
    bedroc = Scoring.CalcBEDROC(matrix, 1, alpha)
    ef1 = Scoring.CalcEnrichment(matrix, 1, [0.01])[0]
    return bedroc, ef1, len(df)


def run_one_gene(gene, n_decoys, force=False):
    if protocol.test_already_done(gene, 5, "test08_decoy_site", force):
        print(f"[test08] {gene}: уже посчитано, пропускаю. --force для пересчёта")
        return protocol.load_stage(gene, 5)["test08_decoy_site"]

    decoy_pocket, err = pick_decoy_pocket(gene)
    if err:
        result = {"error": err}
        protocol.save_stage(gene, 5, {"test08_decoy_site": result})
        print(f"[test08] {gene}: {err}")
        return result

    seed = protocol.cfg_get("metrics", "seed", default=0)
    ligand_set, err = load_unbiased_ligand_set(gene, n_decoys, seed)
    if err:
        result = {"error": err}
        protocol.save_stage(gene, 5, {"test08_decoy_site": result})
        print(f"[test08] {gene}: {err}")
        return result

    # BEDROC в РЕАЛЬНОМ сайте на ЭТОМ ЖЕ фиксированном наборе (Vina-скор
    # уже посчитан в Stage 3) - точка сравнения для BEDROC в ложном сайте.
    real_site_rows = [{"label": r["label"], "score": r["docking_score_kcal_mol"]} for r in ligand_set]
    real_bedroc, real_ef1, real_n = _bedroc_ef(real_site_rows, "score")

    center = decoy_pocket["pocket_center"]
    box_size = (22.5, 22.5, 22.5)  # разумный дефолт-размер бокса вокруг карман-центра
    n_workers = protocol.cfg_get("docking", "n_workers", default=6)

    with open(os.path.join(protocol.BASE_DIR, "confirmed_structures.json"), encoding="utf-8") as f:
        pdb_id = json.load(f)[gene]["pdb_id"]
    receptor_pdbqt = os.path.join(protocol.BASE_DIR, "structures", f"{pdb_id}_receptor.pdbqt")

    cpu = cpu_per_worker(n_workers)
    print(f"[test08] {gene}: докую {len(ligand_set)} лигандов (фиксированный набор) в НЕПРАВИЛЬНЫЙ карман "
          f"#{decoy_pocket['cav_id']} (drug_score={decoy_pocket['drug_score']:.3f}, расстояние до реального "
          f"сайта {decoy_pocket.get('dist_to_known_site', 'н/д')}), {n_workers} воркеров, --cpu={cpu} на воркер")

    import multiprocessing
    tasks = [(r["chembl_id"], r["smiles"], r["label"], receptor_pdbqt, center, box_size, cpu) for r in ligand_set]
    rows = []
    t0 = time.time()
    n_done = 0
    with multiprocessing.Pool(processes=n_workers) as pool:
        for chembl_id, label, score in pool.imap_unordered(_dock_decoy_site_task, tasks):
            n_done += 1
            if score is not None:
                rows.append({"chembl_id": chembl_id, "label": label, "score": score})
            if n_done % 20 == 0 or n_done == len(tasks):
                print(f"  [{n_done}/{len(tasks)}] {(time.time()-t0)/60:.1f} мин прошло")

    decoy_bedroc, decoy_ef1, decoy_n = _bedroc_ef(rows, "score")

    result = {
        "gene": gene, "n_ligands_in_set": len(ligand_set), "seed": seed,
        "decoy_pocket_cav_id": decoy_pocket["cav_id"],
        "decoy_pocket_dist_to_known_site": decoy_pocket.get("dist_to_known_site"),
        "real_site": {"bedroc": real_bedroc, "ef1pct": real_ef1, "n": real_n},
        "decoy_site": {"bedroc": decoy_bedroc, "ef1pct": decoy_ef1, "n": decoy_n, "per_ligand": rows},
        "interpretation": "тот же набор лигандов в двух карманах. Если BEDROC ложного сайта "
                           "сопоставим с реальным - докинг здесь мерит докабельность, не связывание.",
    }
    protocol.save_stage(gene, 5, {"test08_decoy_site": result})

    print(f"\n=== Test 8 ({gene}): receptor-decoy site (набор фиксирован, {len(ligand_set)} лигандов) ===")
    print(f"  РЕАЛЬНЫЙ сайт:    BEDROC={real_bedroc}, EF1%={real_ef1} (n={real_n})")
    print(f"  ЛОЖНЫЙ карман:    BEDROC={decoy_bedroc}, EF1%={decoy_ef1} (n={decoy_n})")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirm", action="store_true", help="обязателен - без него скрипт ничего не делает")
    parser.add_argument("--genes", default=None, help="через запятую, например PIK3CA,ESR1")
    parser.add_argument("--n-decoys", type=int, default=protocol.cfg_get("test08_decoy_site", "n_decoys", default=850))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if not args.confirm or not args.genes:
        print("[test08] Тест ДОРОГОЙ и по протоколу требует --confirm И --genes GENE1,GENE2 явно от пользователя.")
        print("[test08] Ничего не делаю. Пример: python controls/test08_decoy_site.py --confirm --genes PIK3CA")
        return

    for gene in args.genes.split(","):
        run_one_gene(gene.strip(), args.n_decoys, args.force)


if __name__ == "__main__":
    main()
