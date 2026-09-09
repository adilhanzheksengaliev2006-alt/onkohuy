"""
backfill_smiles_cache.py - заполняет runs/_ligand_smiles_cache.json из
УЖЕ СУЩЕСТВУЮЩИХ локальных runs/redock_<gene>/fresh_ligand.sdf файлов -
SMILES там уже были получены (по сети) во время прошлых прогонов Test 5,
просто не кэшировались до сегодняшнего фикса. Восстановление НЕ требует
сети - читает то, что уже на диске.

Использование:
    python backfill_smiles_cache.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "controls"))
import protocol  # noqa: E402
from rdkit import Chem  # noqa: E402

BASE_DIR = protocol.BASE_DIR


def main():
    with open(os.path.join(BASE_DIR, "confirmed_structures.json"), encoding="utf-8") as f:
        confirmed = json.load(f)

    cache = protocol._load_smiles_cache()
    n_added = 0
    n_missing_sdf = 0
    n_failed_parse = 0

    for gene, info in confirmed.items():
        cache_key = info.get("ligand_chembl_id") or f"resname:{info.get('ligand_name')}"
        if cache_key in cache:
            continue
        sdf_path = os.path.join(BASE_DIR, "runs", f"redock_{gene}", "fresh_ligand.sdf")
        if not os.path.exists(sdf_path):
            n_missing_sdf += 1
            continue
        mol = Chem.MolFromMolFile(sdf_path)
        if mol is None:
            n_failed_parse += 1
            print(f"[backfill] {gene}: не удалось распарсить {sdf_path}")
            continue
        smiles = Chem.MolToSmiles(mol)
        cache[cache_key] = smiles
        n_added += 1
        print(f"[backfill] {gene}: {smiles}")

    protocol._save_smiles_cache(cache)
    print(f"\n[backfill] добавлено: {n_added}, нет fresh_ligand.sdf (гейт не запускался): {n_missing_sdf}, "
          f"не распарсилось: {n_failed_parse}, всего в кэше: {len(cache)}")


if __name__ == "__main__":
    main()
