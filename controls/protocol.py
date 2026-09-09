"""
protocol.py — общая инфраструктура для всех 18 тестов: загрузка
config/protocol.yaml, печать хэша конфига при старте, сохранение
результата стадии в results/<GENE>/stage_N.json с пропуском уже
посчитанного (--force для пересчёта).

Ничего специфичного для одного теста здесь быть не должно.
"""
import hashlib
import json
import os
import sys

import yaml

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(BASE_DIR, "config", "protocol.yaml")

_config_cache = None


def load_config():
    global _config_cache
    if _config_cache is None:
        with open(CONFIG_PATH, encoding="utf-8") as f:
            _config_cache = yaml.safe_load(f)
    return _config_cache


def config_hash():
    with open(CONFIG_PATH, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:12]


def print_banner(test_name, used_keys=None):
    cfg = load_config()
    print(f"[{test_name}] config/protocol.yaml hash={config_hash()}")
    if used_keys:
        for key in used_keys:
            node = cfg
            for part in key.split("."):
                node = node[part]
            print(f"[{test_name}]   {key} = {node}")


def results_dir(gene):
    cfg = load_config()
    d = os.path.join(BASE_DIR, cfg["paths"]["results_dir"], gene)
    os.makedirs(d, exist_ok=True)
    return d


def stage_path(gene, stage_n):
    return os.path.join(results_dir(gene), f"stage_{stage_n}.json")


def load_stage(gene, stage_n):
    path = stage_path(gene, stage_n)
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return None


def save_stage(gene, stage_n, data, merge=True):
    path = stage_path(gene, stage_n)
    existing = {}
    if merge and os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            existing = json.load(f)
    existing.update(data)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)
    return path


def test_already_done(gene, stage_n, test_key, force=False):
    """True, если результат теста уже есть в stage-файле и --force не передан."""
    if force:
        return False
    data = load_stage(gene, stage_n)
    return bool(data and test_key in data)


def cfg_get(*path, default=None):
    node = load_config()
    for part in path:
        if node is None:
            return default
        node = node.get(part) if isinstance(node, dict) else None
    return node if node is not None else default


def runs_dir(gene):
    cfg = load_config()
    return os.path.join(BASE_DIR, cfg["paths"]["runs_dir"], f"test_a_{gene}")


SMILES_CACHE_PATH = os.path.join(BASE_DIR, "runs", "_ligand_smiles_cache.json")


def _load_smiles_cache():
    if os.path.exists(SMILES_CACHE_PATH):
        with open(SMILES_CACHE_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_smiles_cache(cache):
    os.makedirs(os.path.dirname(SMILES_CACHE_PATH), exist_ok=True)
    with open(SMILES_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


def fetch_ligand_smiles(ligand_chembl_id, ligand_resname):
    """SMILES референс-лиганда: ChEMBL, если есть ID, иначе RCSB Chemical
    Component Dictionary по 3-буквенному PDB-коду (ligand_resname) - не
    все автообнаруженные мишени имеют ChEMBL ID у лиганда (сложные
    пептидомиметики, природные субстраты - честно None, не выдумываем).
    Раньше это дублировалось в test05_redock.py и test15_posebusters.py
    по отдельности - тест05 чинили, тест15 - нет, поймали на реальном
    прогоне (ADRB1 падал с JSONDecodeError на nc.molecule.get(None)).

    КЭШИРУЕТСЯ на диск (runs/_ligand_smiles_cache.json) - каждый SMILES
    запрашивается по сети РОВНО ОДИН РАЗ за всё время жизни проекта.
    Смысл: перенос на машину без интернета - весь докинг (phase_dock,
    Stage 2/3/4, fpocket) и так уже офлайн; единственная сетевая
    зависимость - этот вызов при первом Test 5/6 для мишени. Если он
    уже когда-либо успешно отработал (кэш заполнен), --force и
    повторные прогоны Test 5/6/15 для ЭТОЙ мишени больше НИКОГДА не
    трогают сеть."""
    cache_key = ligand_chembl_id or f"resname:{ligand_resname}"
    cache = _load_smiles_cache()
    if cache_key in cache:
        return cache[cache_key]

    smiles = None
    if ligand_chembl_id:
        from gene_target_utils import get_chembl_new_client
        nc = get_chembl_new_client()
        rec = nc.molecule.get(ligand_chembl_id)
        smiles = rec.get("molecule_structures", {}).get("canonical_smiles")

    if not smiles:
        if not ligand_resname:
            raise RuntimeError("нет ни ligand_chembl_id, ни ligand_name (HET-код) - неоткуда взять SMILES")
        import requests
        resp = requests.get(f"https://data.rcsb.org/rest/v1/core/chemcomp/{ligand_resname}", timeout=30)
        resp.raise_for_status()
        descr = resp.json().get("rcsb_chem_comp_descriptor", {})
        smiles = descr.get("SMILES_stereo") or descr.get("SMILES")
        if not smiles:
            raise RuntimeError(f"RCSB CCD не дал SMILES для лиганда {ligand_resname}")
        print(f"[fetch_ligand_smiles] SMILES получен из RCSB CCD (не ChEMBL) для {ligand_resname}")

    cache[cache_key] = smiles
    _save_smiles_cache(cache)
    return smiles
