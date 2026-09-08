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
