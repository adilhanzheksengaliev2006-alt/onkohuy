"""
onboard_auto_targets.py - для каждой из 92 автообнаруженных мишеней
(auto_discovered_targets.json): скачивает структуру, готовит рецептор,
добавляет в confirmed_structures.json (помечено как автоматически
найденное - не спутать с вручную подтверждёнными 11), собирает
DUD-E-датасет (~2000 лигандов на Test 18) и прогоняет Stage 1
(redock-гейт) - ЕДИНСТВЕННАЯ реальная проверка, что автонайденная
структура рабочая, а не просто "похоже на то же самое" по UniProt.

Прогресс пишется на диск ПОСЛЕ КАЖДОЙ мишени (урок сегодняшней ночи:
агент дважды терял часы работы, не сохраняя промежуточный результат).

Использование:
    python onboard_auto_targets.py [--limit N]
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gene_target_utils import download_pdb, find_ligand_center  # noqa: E402
from dock_existing_candidates import prepare_receptor  # noqa: E402
import run_test_a as rta  # noqa: E402
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "controls"))
import test05_redock as t05  # noqa: E402

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AUTO_TARGETS_PATH = os.path.join(BASE_DIR, "auto_discovered_targets.json")
CONFIRMED_PATH = os.path.join(BASE_DIR, "confirmed_structures.json")
PROGRESS_PATH = os.path.join(BASE_DIR, "runs", "_onboard_progress.json")

N_DECOYS_TARGET = 1900  # + активные (не капаются) ~= 2000 на мишень, только для Test 18


def load_progress():
    if os.path.exists(PROGRESS_PATH):
        with open(PROGRESS_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_progress(progress):
    os.makedirs(os.path.dirname(PROGRESS_PATH), exist_ok=True)
    with open(PROGRESS_PATH, "w", encoding="utf-8") as f:
        json.dump(progress, f, indent=2, ensure_ascii=False)


def onboard_one(entry):
    gene = entry["gene"]
    pdb_id = entry["pdb_id"]
    dude_code = entry["dude_code"]

    # 1. скачать структуру
    struct_path = download_pdb(pdb_id)

    # 2. найти лиганд/бокс (авто-детект, не просто доверяем ligand_resname из JSON)
    info = find_ligand_center(struct_path)
    if info is None:
        raise RuntimeError(f"find_ligand_center не нашла лиганд в {struct_path}")

    # 3. подготовить рецептор
    struct_dir = os.path.join(BASE_DIR, "structures")
    out_basename = os.path.join(struct_dir, f"{pdb_id}_receptor")
    prepare_receptor(struct_path, info["center"], info["box_size"], out_basename)

    # 4. добавить в confirmed_structures.json (помечено авто)
    with open(CONFIRMED_PATH, encoding="utf-8") as f:
        confirmed = json.load(f)
    confirmed[gene] = {
        "pdb_id": pdb_id,
        "confirmed_by": "automated (RCSB GraphQL + UniProt cross-check, redock-gate pending/passed)",
        "ligand_name": entry.get("ligand_resname"),
        "ligand_chembl_id": entry.get("ligand_chembl_id"),
        "note": f"auto-discovered для Test 18 (100-мишеневая выборка). confidence={entry.get('confidence')}. "
                f"{entry.get('verification_note', '')}",
    }
    with open(CONFIRMED_PATH, "w", encoding="utf-8") as f:
        json.dump(confirmed, f, indent=2, ensure_ascii=False)

    # 5. собрать DUD-E датасет (~2000 на Test 18, не 10000 - это не флагманская мишень)
    rta.phase_build_dude(gene, dude_code, n_actives_cap=None, n_decoys_cap=N_DECOYS_TARGET)

    # 6. Stage 1 - ЕДИНСТВЕННАЯ реальная проверка
    gate_result = t05.run(gene, force=True)
    return gate_result


def main():
    limit = None
    for i, a in enumerate(sys.argv):
        if a == "--limit" and i + 1 < len(sys.argv):
            limit = int(sys.argv[i + 1])

    with open(AUTO_TARGETS_PATH, encoding="utf-8") as f:
        targets = json.load(f)
    if limit:
        targets = targets[:limit]

    progress = load_progress()
    print(f"[onboard] мишеней: {len(targets)}, уже сделано: "
          f"{sum(1 for t in targets if progress.get(t['gene'], {}).get('status') in ('pass','fail'))}")

    for entry in targets:
        gene = entry["gene"]
        if progress.get(gene, {}).get("status") in ("pass", "fail"):
            print(f"[onboard] {gene}: уже сделано ({progress[gene]['status']}), пропускаю")
            continue
        print(f"\n{'='*50}\n[onboard] {gene} ({entry['pdb_id']}, {entry['dude_code']}) старт\n{'='*50}")
        t0 = time.time()
        try:
            gate_result = onboard_one(entry)
            status = gate_result.get("status", "unknown")
            progress[gene] = {"status": status, "rmsd_top1": gate_result.get("rmsd_top1"),
                               "error": gate_result.get("error"),  # раньше терялось для status=error из самого гейта
                               "pdb_id": entry["pdb_id"], "elapsed_sec": time.time() - t0}
            print(f"[onboard] {gene}: гейт status={status}, RMSD={gate_result.get('rmsd_top1')}, "
                  f"{(time.time()-t0)/60:.1f} мин")
        except Exception as e:
            progress[gene] = {"status": "error", "error": str(e), "pdb_id": entry["pdb_id"],
                               "elapsed_sec": time.time() - t0}
            print(f"[onboard] {gene}: ОШИБКА {e}")
        save_progress(progress)  # ПОСЛЕ КАЖДОЙ мишени - не терять прогресс

    n_pass = sum(1 for v in progress.values() if v.get("status") == "pass")
    n_fail = sum(1 for v in progress.values() if v.get("status") == "fail")
    n_error = sum(1 for v in progress.values() if v.get("status") == "error")
    print(f"\n[onboard] ИТОГО: pass={n_pass}, fail(гейт)={n_fail}, error(сбой)={n_error}, всего={len(progress)}")


if __name__ == "__main__":
    main()
