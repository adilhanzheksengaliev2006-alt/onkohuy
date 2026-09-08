"""
run_fpocket.py — обёртка над fpocket (установлен в WSL, conda-forge
сборка 4.0.2, т.к. sudo/apt/conda недоступны на этой машине - см.
runs/fpocket_setup_notes в комментариях ниже) для дескрипторов карманов
рецептора. Шаг 4 плана расширения на 10-15 мишеней.

Устройство: fpocket ставит НЕТ Windows-версии, поэтому вызывается через
wsl.exe как subprocess. Копирует PDB в WSL-нативную временную директорию
(НЕ /mnt/c/... - доступ из Linux к NTFS через 9P мост заметно медленнее,
особенно на сотнях мелких файлов, которые генерирует fpocket - по одному
pocketN_atm.pdb/_env_atm.pdb/_vert.pqr на каждый найденный карман).
Парсит табличный вывод (-d, с заголовком в первой строке) в структурированный
JSON и определяет, какой из найденных карманов ближе всего к known
box_center (нашему подтверждённому активному сайту) - помечает его.

Использование:
    python run_fpocket.py [GENE]
    по умолчанию: PIK3CA
"""
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gene_target_utils import find_ligand_center  # noqa: E402

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FPOCKET_BIN_WSL = "~/fpocket_install/extracted/bin/fpocket"
WSL_WORKDIR = "~/fpocket_runs"
CONFIRMED_PATH = os.path.join(BASE_DIR, "confirmed_structures.json")


def _load_confirmed(gene):
    with open(CONFIRMED_PATH, encoding="utf-8") as f:
        data = json.load(f)
    if gene not in data:
        print(f"ОШИБКА: {gene} нет в {CONFIRMED_PATH}")
        sys.exit(1)
    return data[gene]


def struct_pdb_path(gene):
    """_clean.pdb - БЕЗ со-кристаллизованного лиганда (создаётся
    prepare_receptor()/strip_ligands_and_waters() как побочный продукт
    подготовки рецептора) - fpocket ищет карманы по геометрии белка,
    полость с оставленным лигандом не была бы обнаружена как карман."""
    return os.path.join(BASE_DIR, "structures", f"{_load_confirmed(gene)['pdb_id']}_clean.pdb")


def native_pdb_path(gene):
    return os.path.join(BASE_DIR, "structures", f"{_load_confirmed(gene)['pdb_id']}.pdb")


def win_path_to_wsl(path):
    """C:\\a\\b -> /mnt/c/a/b - достаточно для разового copy-in, дальше
    работаем полностью в WSL-нативной директории."""
    drive, rest = os.path.splitdrive(os.path.abspath(path))
    rest = rest.replace("\\", "/")
    return f"/mnt/{drive[0].lower()}{rest}"


def run_wsl(bash_cmd, timeout=180):
    result = subprocess.run(
        ["wsl", "-e", "bash", "-c", bash_cmd],
        capture_output=True, text=True, timeout=timeout,
    )
    return result


def parse_descriptor_table(stdout_text):
    lines = [l for l in stdout_text.strip().splitlines() if l.strip()]
    if not lines:
        return []
    header = lines[0].split()
    rows = []
    for line in lines[1:]:
        parts = line.split()
        if len(parts) != len(header):
            continue  # защитно пропускаем неполные/битые строки, не роняем весь парсинг
        row = {}
        for key, val in zip(header, parts):
            try:
                row[key] = float(val) if "." in val else int(val)
            except ValueError:
                row[key] = val
        rows.append(row)
    return rows


def run_fpocket(gene):
    pdb_path = struct_pdb_path(gene)
    if not pdb_path or not os.path.exists(pdb_path):
        print(f"ОШИБКА: нет подготовленной структуры для {gene} ({pdb_path})")
        sys.exit(1)

    print(f"[fpocket] {gene}: копирую {os.path.basename(pdb_path)} в WSL-нативную директорию...")
    wsl_src = win_path_to_wsl(pdb_path)
    fname = os.path.basename(pdb_path)
    setup = run_wsl(f"mkdir -p {WSL_WORKDIR} && cp '{wsl_src}' {WSL_WORKDIR}/{fname}")
    if setup.returncode != 0:
        print(f"ОШИБКА копирования в WSL: {setup.stderr}"); sys.exit(1)

    print("[fpocket] запускаю fpocket -d ...")
    result = run_wsl(f"cd {WSL_WORKDIR} && rm -rf {fname.rsplit('.',1)[0]}_out && "
                      f"{FPOCKET_BIN_WSL} -f {fname} -d")
    if result.returncode != 0:
        print(f"ОШИБКА fpocket: {result.stderr[-1000:]}"); sys.exit(1)

    pockets = parse_descriptor_table(result.stdout)
    print(f"[fpocket] найдено карманов: {len(pockets)}")
    if not pockets:
        print("ОШИБКА: fpocket не вернул ни одной строки дескрипторов"); sys.exit(1)

    # какой карман ближе всего к нашему подтверждённому активному сайту
    native_pdb = native_pdb_path(gene)
    known_center = None
    if native_pdb and os.path.exists(native_pdb):
        info = find_ligand_center(native_pdb)
        known_center = info["center"]
        print(f"[fpocket] известный активный сайт (центр со-кристаллизованного лиганда): {known_center}")

    out_dir_name = f"{fname.rsplit('.', 1)[0]}_out"
    best_cav = None
    best_dist = None
    if known_center is not None:
        for p in pockets:
            cav_id = int(p["cav_id"])
            pqr_wsl = f"{WSL_WORKDIR}/{out_dir_name}/pocket{cav_id}_vert.pqr"
            cat_result = run_wsl(f"cat {pqr_wsl}", timeout=30)
            if cat_result.returncode != 0:
                continue
            center = pocket_center_from_pqr_text(cat_result.stdout)
            if center is None:
                continue
            p["pocket_center"] = center
            dist = sum((a - b) ** 2 for a, b in zip(center, known_center)) ** 0.5
            p["dist_to_known_site"] = dist
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_cav = cav_id
        if best_cav is not None:
            print(f"[fpocket] карман #{best_cav} ближе всего к известному сайту "
                  f"(расстояние центров {best_dist:.2f}A), drug_score="
                  f"{next(p['drug_score'] for p in pockets if int(p['cav_id']) == best_cav)}")
            for p in pockets:
                p["matches_known_site"] = (int(p["cav_id"]) == best_cav)

    out_path = os.path.join(BASE_DIR, "runs", f"fpocket_{gene}.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"gene": gene, "n_pockets": len(pockets), "known_active_site_center": known_center,
                   "pocket_matching_known_site": best_cav, "pockets": pockets}, f, indent=2)
    print(f"\n[fpocket] Сохранено: {out_path}")

    pockets_by_score = sorted(pockets, key=lambda p: -p.get("drug_score", 0))
    print("\n=== Топ-5 карманов по druggability score ===")
    for p in pockets_by_score[:5]:
        marker = " <-- наш известный сайт" if p.get("matches_known_site") else ""
        print(f"  #{int(p['cav_id'])}: drug_score={p['drug_score']:.3f}, volume={p['volume']:.1f}, "
              f"nb_asph={int(p['nb_asph'])}{marker}")
    return pockets


def pocket_center_from_pqr_text(text):
    xs, ys, zs = [], [], []
    for line in text.splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        parts = line.split()
        try:
            xs.append(float(parts[-5]))
            ys.append(float(parts[-4]))
            zs.append(float(parts[-3]))
        except (IndexError, ValueError):
            continue
    if not xs:
        return None
    return (sum(xs) / len(xs), sum(ys) / len(ys), sum(zs) / len(zs))


def main():
    gene = sys.argv[1] if len(sys.argv) > 1 else "PIK3CA"
    run_fpocket(gene)


if __name__ == "__main__":
    main()
