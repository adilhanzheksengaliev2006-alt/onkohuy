"""
check_offline_readiness.py - для каждой мишени в confirmed_structures.json
проверяет, есть ли ВСЕ артефакты, нужные для полностью офлайн-докинга:
структура скачана, рецептор подготовлен, датасет собран, SMILES
референс-лиганда закэширован. НЕ проверяет сам факт прохождения гейта -
это отдельный вопрос (мишень может быть офлайн-готова, но провалить
RMSD, и это нормально).

Использование:
    python check_offline_readiness.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "controls"))
import protocol  # noqa: E402

BASE_DIR = protocol.BASE_DIR


def check_gene(gene, info):
    pdb_id = info["pdb_id"]
    checks = {
        "structure_downloaded": os.path.exists(os.path.join(BASE_DIR, "structures", f"{pdb_id}.pdb")),
        "receptor_prepared": os.path.exists(os.path.join(BASE_DIR, "structures", f"{pdb_id}_receptor.pdbqt")),
        "dataset_built": os.path.exists(os.path.join(protocol.runs_dir(gene), "ligands.json")),
        "smiles_cached": False,
    }
    cache = protocol._load_smiles_cache()
    cache_key = info.get("ligand_chembl_id") or f"resname:{info.get('ligand_name')}"
    checks["smiles_cached"] = cache_key in cache
    checks["fully_offline_ready"] = all(checks.values())
    return checks


def main():
    with open(os.path.join(BASE_DIR, "confirmed_structures.json"), encoding="utf-8") as f:
        confirmed = json.load(f)

    print(f"[offline_check] мишеней в реестре: {len(confirmed)}")
    n_ready = 0
    not_ready = []
    for gene, info in confirmed.items():
        checks = check_gene(gene, info)
        status = "READY" if checks["fully_offline_ready"] else "не готово"
        print(f"  {gene:12s} {status:12s} {checks}")
        if checks["fully_offline_ready"]:
            n_ready += 1
        else:
            not_ready.append(gene)

    print(f"\n[offline_check] готовы к офлайн-работе: {n_ready}/{len(confirmed)}")
    if not_ready:
        print(f"[offline_check] НЕ готовы (нужен интернет хотя бы раз для этих): {not_ready}")


if __name__ == "__main__":
    main()
