"""
run_test_a.py — реальный Тест A (ретроспективная валидация докингом) для
одной мишени. Три фазы, каждая resumable/пропускаемая отдельно:

  1. build  - собирает активные (ChEMBL, IC50<=100нМ) + достраивает
              декои по свойствам (MW/logP/HBD/HBA + Tanimoto-фильтр
              несходства с активными, как в DUD-E) до заданного
              соотношения. Сохраняет runs/test_a_<GENE>/ligands.json.
              Сетевая фаза, быстрая (минуты), НЕ требует докинга.
  2. dock   - докует всё из ligands.json параллельно (--cpu фикс из
              benchmark_parallel_docking.py, чтобы не было oversubscription),
              пишет runs/test_a_<GENE>/results.jsonl построчно (resumable -
              при перезапуске уже задокированные лиганды пропускаются).
              ТЯЖЁЛАЯ фаза (часы). Запускать ТОЛЬКО из консоли с закрытым
              VS Code.
  3. analyze - читает results.jsonl, считает BEDROC/AUC/EF/Тест N/Тест A'
              через bedroc_calibration.py, сохраняет report.json.

  4. funnel  - консенсус-ранг (сумма ранга по Vina + ранга по gnina) по
              уже готовым результатам dock, топ-20% (настраивается)
              передокуются заново с exhaustiveness=32 (в 4 раза тщательнее)
              + свежий gnina - в funnel_results.jsonl. Дёшево (только 20%
              библиотеки), но именно этому финальному списку стоит
              доверять больше всего.
  5. top     - печатает итоговый консенсус-топ ПОСЛЕ funnel.

Использование:
    python run_test_a.py build   [GENE] [N_ACTIVES] [DECOY_RATIO]
    python run_test_a.py dock    [GENE] [N_WORKERS] [TIMEOUT_SEC] [GNINA: 1 или 0, по умолчанию 1]
    python run_test_a.py analyze [GENE]
    python run_test_a.py funnel  [GENE] [N_WORKERS] [TIMEOUT_SEC] [TOP_FRACTION, по умолчанию 0.20]
    python run_test_a.py top     [GENE] [TOP_N, по умолчанию 20]
    python run_test_a.py all     [GENE] [N_ACTIVES] [DECOY_RATIO] [N_WORKERS]  - build+dock+analyze подряд

    по умолчанию: PIK3CA, 100 активных, ratio 30 (реально достижимый
    может быть меньше - см. предупреждение при build), 6 воркеров.
"""
import json
import multiprocessing
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gene_target_utils import get_chembl_new_client, find_ligand_center  # noqa: E402
from dock_existing_candidates import dock_smiles_isolated  # noqa: E402
from benchmark_parallel_docking import cpu_per_worker  # noqa: E402
import bedroc_calibration as bc  # noqa: E402

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONTROL_PATH = os.path.join(BASE_DIR, "control.json")
CONFIRMED_PATH = os.path.join(BASE_DIR, "confirmed_structures.json")
SCREENING_PATH = os.path.join(BASE_DIR, "target_screening_results.json")
DUDE_DIR = os.path.join(BASE_DIR, "dude_datasets")

TANIMOTO_MAX_SIMILARITY = 0.35  # как в DUD-E - декой не должен быть похож ни на один активный
MW_TOLERANCE = 30.0
LOGP_TOLERANCE = 1.0


def _load_protocol_yaml():
    import yaml
    with open(os.path.join(BASE_DIR, "config", "protocol.yaml"), encoding="utf-8") as f:
        return yaml.safe_load(f)


CANDIDATES_PER_ACTIVE = _load_protocol_yaml().get("chembl_decoy_build", {}).get("candidates_per_active", 60)
DOCK_TIMEOUT = 300  # было 120 - слишком часто спотыкались об таймаут на обычных (не самых крупных) молекулах при exhaustiveness=8
EXHAUSTIVENESS = 8


def _load_confirmed(gene):
    with open(CONFIRMED_PATH, encoding="utf-8") as f:
        data = json.load(f)
    if gene not in data:
        raise RuntimeError(f"{gene} нет в {CONFIRMED_PATH} - структура ещё не подтверждена")
    return data[gene]


def receptor_pdbqt_path(gene):
    return os.path.join(BASE_DIR, "structures", f"{_load_confirmed(gene)['pdb_id']}_receptor.pdbqt")


def ligand_structure_pdb_path(gene):
    return os.path.join(BASE_DIR, "structures", f"{_load_confirmed(gene)['pdb_id']}.pdb")


def _load_screening_row(gene):
    if not os.path.exists(SCREENING_PATH):
        return None
    with open(SCREENING_PATH, encoding="utf-8") as f:
        rows = json.load(f)
    return next((r for r in rows if r.get("gene") == gene), None)


def target_chembl_id_for(gene):
    row = _load_screening_row(gene)
    if row and row.get("target_chembl_id"):
        return row["target_chembl_id"]
    raise RuntimeError(f"Нет target_chembl_id для {gene} в {SCREENING_PATH}")


def dude_code_for(gene):
    """None, если мишени нет в DUD-E (напр. PIK3CA) - тогда используем
    ChEMBL-live build (fetch_actives + generate_decoys), а не DUD-E."""
    row = _load_screening_row(gene)
    return row.get("dude_code") if row else None


def run_dir(gene):
    d = os.path.join(BASE_DIR, "runs", f"test_a_{gene}")
    os.makedirs(d, exist_ok=True)
    return d


def check_control():
    if os.path.exists(CONTROL_PATH):
        try:
            with open(CONTROL_PATH, encoding="utf-8") as f:
                return bool(json.load(f).get("abort"))
        except Exception:
            return False
    return False


# ============================== ФАЗА 1: build ==============================

def fetch_actives(target_chembl_id, n_actives):
    nc = get_chembl_new_client()
    activity = nc.activity
    print(f"[build] запрашиваю активные (IC50<=100нМ) для {target_chembl_id}...")
    records = list(
        activity.filter(
            target_chembl_id=target_chembl_id, standard_type="IC50",
            standard_units="nM", standard_value__lte=100,
        ).only(["molecule_chembl_id", "standard_value"])
    )
    best = {}
    for r in records:
        cid = r["molecule_chembl_id"]
        val = r.get("standard_value")
        if val is None:
            continue
        val = float(val)
        if cid not in best or val < best[cid]:
            best[cid] = val
    # берём самые сильнодействующие n_actives - меньше шума от погранично-активных
    chosen_ids = sorted(best, key=best.get)[:n_actives]
    print(f"[build] уникальных активных всего: {len(best)}, беру {len(chosen_ids)} самых сильных")

    nc_mol = nc.molecule
    actives = []
    for cid in chosen_ids:
        try:
            rec = nc_mol.get(cid)
            smiles = rec.get("molecule_structures", {}).get("canonical_smiles")
        except Exception as e:
            print(f"  [warn] {cid}: не удалось получить SMILES ({str(e)[:100]}), пропуск")
            continue
        if not smiles:
            continue
        actives.append({"chembl_id": cid, "smiles": smiles, "ic50_nm": best[cid], "label": 1})
    print(f"[build] активных с валидным SMILES: {len(actives)}")
    return actives, set(best.keys())


def mol_properties(smiles):
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Lipinski
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return {
        "mw": Descriptors.MolWt(mol),
        "logp": Descriptors.MolLogP(mol),
        "hbd": Lipinski.NumHDonors(mol),
        "hba": Lipinski.NumHAcceptors(mol),
    }


def fingerprint(smiles):
    from rdkit import Chem
    from rdkit.Chem import AllChem
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)


def generate_decoys(actives, exclude_chembl_ids, target_ratio):
    """DUD-E-style: property-matched (MW/logP кандидата в пределах допуска
    от активного), но Tanimoto-НЕсходные ни с одним активным (порог 0.35 -
    декой не должен быть химическим аналогом активного, иначе это не
    декой, а слабый агонист/родственная структура)."""
    from rdkit import Chem, DataStructs

    nc = get_chembl_new_client()
    mol_client = nc.molecule

    active_props = []
    active_fps = []
    for a in actives:
        props = mol_properties(a["smiles"])
        fp = fingerprint(a["smiles"])
        if props is None or fp is None:
            continue
        active_props.append(props)
        active_fps.append(fp)

    target_total = int(len(actives) * target_ratio)
    print(f"[build] цель по декоям: {target_total} (соотношение 1:{target_ratio})")

    decoy_pool = {}
    for i, props in enumerate(active_props):
        if check_control():
            break
        mw_lo, mw_hi = props["mw"] - MW_TOLERANCE, props["mw"] + MW_TOLERANCE
        logp_lo, logp_hi = props["logp"] - LOGP_TOLERANCE, props["logp"] + LOGP_TOLERANCE
        try:
            candidates = list(
                mol_client.filter(
                    molecule_properties__full_mwt__range=(mw_lo, mw_hi),
                    molecule_properties__alogp__range=(logp_lo, logp_hi),
                    molecule_structures__isnull=False,
                ).only(["molecule_chembl_id", "molecule_structures"])[:CANDIDATES_PER_ACTIVE]
            )
        except Exception as e:
            print(f"  [warn] запрос декоев для активного #{i} упал: {str(e)[:150]}")
            continue
        for c in candidates:
            cid = c["molecule_chembl_id"]
            if cid in exclude_chembl_ids or cid in decoy_pool:
                continue
            smiles = c.get("molecule_structures", {}).get("canonical_smiles")
            if smiles:
                decoy_pool[cid] = smiles
        print(f"  [{i+1}/{len(active_props)}] пул декоев пока: {len(decoy_pool)}")
        if len(decoy_pool) >= target_total * 2:
            # с запасом на Tanimoto-отсев ниже - не молотим ChEMBL дальше без нужды
            break
        time.sleep(1)

    print(f"[build] Tanimoto-фильтр несходства (порог {TANIMOTO_MAX_SIMILARITY}) на {len(decoy_pool)} кандидатах...")
    decoys = []
    for cid, smiles in decoy_pool.items():
        fp = fingerprint(smiles)
        if fp is None:
            continue
        max_sim = max((DataStructs.TanimotoSimilarity(fp, afp) for afp in active_fps), default=0.0)
        if max_sim < TANIMOTO_MAX_SIMILARITY:
            decoys.append({"chembl_id": cid, "smiles": smiles, "label": 0, "max_tanimoto_to_active": max_sim})
        if len(decoys) >= target_total:
            break

    achieved_ratio = len(decoys) / len(actives) if actives else 0
    print(f"[build] декоев прошло Tanimoto-фильтр: {len(decoys)} "
          f"(реально достижимое соотношение 1:{achieved_ratio:.1f}, цель была 1:{target_ratio})")
    return decoys, achieved_ratio


def phase_build(gene, n_actives, decoy_ratio):
    target_chembl_id = target_chembl_id_for(gene)
    actives, all_active_ids = fetch_actives(target_chembl_id, n_actives)
    if len(actives) < 10:
        print(f"ОШИБКА: активных с валидным SMILES слишком мало ({len(actives)}), Тест A не имеет смысла")
        sys.exit(1)

    decoys, achieved_ratio = generate_decoys(actives, all_active_ids, decoy_ratio)
    if len(decoys) < 10:
        print(f"ОШИБКА: декоев после фильтра слишком мало ({len(decoys)}), Тест A не имеет смысла")
        sys.exit(1)

    ligands = actives + decoys
    out_path = os.path.join(run_dir(gene), "ligands.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "gene": gene, "target_chembl_id": target_chembl_id,
            "n_actives": len(actives), "n_decoys": len(decoys),
            "requested_ratio": decoy_ratio, "achieved_ratio": achieved_ratio,
            "ligands": ligands,
        }, f, indent=2)
    print(f"\n[build] Готово: {len(actives)} активных + {len(decoys)} декоев = {len(ligands)} лигандов")
    print(f"[build] Сохранено: {out_path}")


def phase_build_dude(gene, dude_code, n_actives_cap=None, n_decoys_cap=None):
    """Build из готового датасета DUD-E (dude_datasets/<code>/actives_final.ism,
    decoys_final.ism) вместо ChEMBL-live-запроса + самодельных property-matched
    декоев (см. phase_build) - для мишеней, у которых DUD-E есть, это лучше:
    опубликованная, рецензируемая методология подбора декоев, а не наша
    собственная. n_actives_cap/n_decoys_cap - для тестового прогона на малом
    масштабе перед полным (напр. build_dude(gene, 10, 300) на проверку багов)."""
    actives_path = os.path.join(DUDE_DIR, dude_code, "actives_final.ism")
    decoys_path = os.path.join(DUDE_DIR, dude_code, "decoys_final.ism")
    if not os.path.exists(actives_path) or not os.path.exists(decoys_path):
        print(f"ОШИБКА: нет {actives_path} или {decoys_path} - датасет DUD-E для {dude_code} не скачан")
        sys.exit(1)

    actives = []
    with open(actives_path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 2:
                continue
            smiles, zinc_id = parts[0], parts[1]
            chembl_id = parts[2] if len(parts) > 2 else f"ZINC{zinc_id}"
            actives.append({"chembl_id": chembl_id, "smiles": smiles, "label": 1})
    decoys = []
    with open(decoys_path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 2:
                continue
            smiles, zinc_id = parts[0], parts[1]
            decoys.append({"chembl_id": f"ZINC{zinc_id}", "smiles": smiles, "label": 0})

    # ВАЖНО: СЛУЧАЙНАЯ подвыборка с фикс. seed, НЕ первые N строк файла.
    # Раньше было actives[:n_actives_cap]/decoys[:n_decoys_cap] - префикс
    # DUD-E-файла decoys_final.ism систематически смещён по logP
    # (обнаружено на BRAF: медиана logP по первым 500 строкам = 2.54,
    # по случайным 500 из всех 9950 = 3.24, по равномерной выборке по
    # всему файлу = 3.20) - т.к. декои в файле сгруппированы по тому,
    # к какому активному они были подобраны, а не перемешаны. Обрезка
    # префиксом при n_decoys_cap < len(decoys) молча ломала как раз то
    # property-matching (MW/logP), ради которого DUD-E и используется
    # вместо своих декоев (см. докстринг выше).
    import random
    rng = random.Random(0)
    if n_actives_cap is not None and n_actives_cap < len(actives):
        actives = rng.sample(actives, n_actives_cap)
    if n_decoys_cap is not None and n_decoys_cap < len(decoys):
        decoys = rng.sample(decoys, n_decoys_cap)

    ligands = actives + decoys
    achieved_ratio = len(decoys) / len(actives) if actives else 0
    out_path = os.path.join(run_dir(gene), "ligands.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "gene": gene, "source": "DUD-E", "dude_code": dude_code,
            "n_actives": len(actives), "n_decoys": len(decoys),
            "achieved_ratio": achieved_ratio, "ligands": ligands,
        }, f, indent=2)
    print(f"[build_dude] {gene} ({dude_code}): {len(actives)} активных + {len(decoys)} декоев "
          f"= {len(ligands)} лигандов (соотношение 1:{achieved_ratio:.1f})")
    print(f"[build_dude] Сохранено: {out_path}")


# ============================== ФАЗА 2: dock ==============================

def _dock_one(args):
    chembl_id, smiles, label, box_center, box_size, receptor, workdir, tag, cpu, timeout, gnina, exhaustiveness = args
    t0 = time.time()
    result = dock_smiles_isolated(
        smiles, receptor, box_center, box_size, workdir, tag=tag,
        exhaustiveness=exhaustiveness, timeout=timeout, cpu=cpu, gnina_rescore=gnina,
    )
    score, gnina_data = result if gnina else (result, None)
    row = {
        "chembl_id": chembl_id, "smiles": smiles, "label": label,
        "docking_score_kcal_mol": score, "dock_sec": time.time() - t0,
        "heavy_atom_count": _heavy_atoms(smiles),
    }
    if gnina:
        row["gnina_cnn_score"] = gnina_data.get("cnn_score") if gnina_data else None
        row["gnina_cnn_affinity"] = gnina_data.get("cnn_affinity") if gnina_data else None
        row["gnina_vina_affinity"] = gnina_data.get("vina_affinity_gnina") if gnina_data else None
    return row


def _heavy_atoms(smiles):
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smiles)
    return mol.GetNumHeavyAtoms() if mol else None


def load_completed_ids(results_path):
    done = set()
    if not os.path.exists(results_path):
        return done
    with open(results_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if rec.get("docking_score_kcal_mol") is not None:
                    done.add(rec["chembl_id"])
            except json.JSONDecodeError:
                continue
    return done


def phase_dock(gene, n_workers, timeout=None, gnina=True):
    """Ранее провалившиеся (score=None) лиганды НЕ считаются сделанными
    (load_completed_ids не засчитывает score=None) и автоматически
    передокуются при повторном запуске этой фазы. Это важно: 87% фейлов
    первого прогона (804 из 1581, из них 666 ровно на 120.1-120.2с) были
    не плохой химией, а таймаутом - --cpu-фикс от oversubscription
    (см. cpu_per_worker()) урезает потоки на воркер, и часть настоящих
    крупных ингибиторов перестаёт укладываться в 120с. Передай больший
    timeout и/или меньше n_workers (больше --cpu на процесс) для повторного
    прогона именно застрявших лигандов."""
    d = run_dir(gene)
    ligands_path = os.path.join(d, "ligands.json")
    if not os.path.exists(ligands_path):
        print(f"ОШИБКА: нет {ligands_path} - сначала запусти фазу build"); sys.exit(1)
    with open(ligands_path, encoding="utf-8") as f:
        data = json.load(f)

    receptor = receptor_pdbqt_path(gene)
    ligand_struct = ligand_structure_pdb_path(gene)
    ligand_info = find_ligand_center(ligand_struct)
    box_center, box_size = ligand_info["center"], ligand_info["box_size"]

    results_path = os.path.join(d, "results.jsonl")
    already_done = load_completed_ids(results_path)
    todo = [lig for lig in data["ligands"] if lig["chembl_id"] not in already_done]
    print(f"[dock] всего лигандов: {len(data['ligands'])}, уже успешно сделано: {len(already_done)}, "
          f"осталось (включая ранее провалившиеся по таймауту): {len(todo)}")
    if not todo:
        print("[dock] всё уже задокировано."); return

    workdir = os.path.join(d, "dock_tmp")
    os.makedirs(workdir, exist_ok=True)
    cpu = cpu_per_worker(n_workers)
    effective_timeout = timeout if timeout is not None else DOCK_TIMEOUT
    print(f"[dock] воркеров: {n_workers}, --cpu на процесс: {cpu if cpu else 'не ограничено'}, "
          f"exhaustiveness={EXHAUSTIVENESS}, timeout={effective_timeout}с, gnina-рескоринг: {gnina}")

    tasks = [
        (lig["chembl_id"], lig["smiles"], lig["label"], box_center, box_size,
         receptor, workdir, f"testA_{gene}_{lig['chembl_id']}", cpu, effective_timeout, gnina, EXHAUSTIVENESS)
        for lig in todo
    ]

    t_start = time.time()
    n_done_this_run = 0
    with open(results_path, "a", encoding="utf-8") as out_f:
        if n_workers == 1:
            for task in tasks:
                if check_control():
                    print("[control] control.json просит остановиться."); break
                res = _dock_one(task)
                out_f.write(json.dumps(res, ensure_ascii=False) + "\n")
                out_f.flush()
                n_done_this_run += 1
                if n_done_this_run % 20 == 0:
                    elapsed = time.time() - t_start
                    rate = n_done_this_run / elapsed
                    eta_sec = (len(tasks) - n_done_this_run) / rate if rate else 0
                    print(f"  [{n_done_this_run}/{len(tasks)}] {elapsed/60:.1f} мин прошло, "
                          f"ETA ~{eta_sec/60:.1f} мин")
        else:
            with multiprocessing.Pool(processes=n_workers) as pool:
                for res in pool.imap_unordered(_dock_one, tasks):
                    out_f.write(json.dumps(res, ensure_ascii=False) + "\n")
                    out_f.flush()
                    n_done_this_run += 1
                    if n_done_this_run % 20 == 0:
                        elapsed = time.time() - t_start
                        rate = n_done_this_run / elapsed
                        eta_sec = (len(tasks) - n_done_this_run) / rate if rate else 0
                        print(f"  [{n_done_this_run}/{len(tasks)}] {elapsed/60:.1f} мин прошло, "
                              f"ETA ~{eta_sec/60:.1f} мин")
                        if check_control():
                            print("[control] control.json просит остановиться, завершаю текущую партию.")
                            pool.terminate()
                            break

    print(f"\n[dock] Фаза докинга: {n_done_this_run} лигандов за {(time.time()-t_start)/60:.1f} мин")
    print(f"[dock] Результаты: {results_path}")


# ============================== ФАЗА 2.5: funnel (консенсус + доуточнение) =====

FUNNEL_TOP_FRACTION = 0.20
FUNNEL_EXHAUSTIVENESS = 32  # 4x обычного (8) - для короткого списка это оправданно


def phase_funnel(gene, n_workers, top_fraction=FUNNEL_TOP_FRACTION,
                  exhaustiveness=FUNNEL_EXHAUSTIVENESS, timeout=None):
    """Двухэтапная воронка: сырой Vina-скор ненадёжен как итоговый критерий
    (на реальном прогоне PIK3CA - 0 активных в топ-10 по Vina против 4 из 10
    по gnina), поэтому вместо того чтобы доверять Этапу 1 (дешёвый
    exhaustiveness=8 по всей библиотеке) напрямую, считаем консенсус-ранг
    (сумма ранга по Vina + ранга по gnina - оба должны согласиться, что
    молекула хорошая) и ТОЛЬКО топ-20% по консенсусу передокуем заново с
    exhaustiveness=32 (в 4 раза тщательнее) + свежий gnina-рескоринг.
    Дёшево на масштабе (весь остальной пайплайн как был), точно там, где
    важно (финальные кандидаты)."""
    d = run_dir(gene)
    results_path = os.path.join(d, "results.jsonl")
    if not os.path.exists(results_path):
        print(f"ОШИБКА: нет {results_path} - сначала запусти фазу dock"); sys.exit(1)

    by_id = {}
    with open(results_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            cid = rec["chembl_id"]
            prev = by_id.get(cid)
            if prev is None or rec.get("docking_score_kcal_mol") is not None:
                by_id[cid] = rec

    scored = [r for r in by_id.values()
              if r.get("docking_score_kcal_mol") is not None and r.get("gnina_cnn_score") is not None]
    if len(scored) < 10:
        print(f"ОШИБКА: только {len(scored)} лигандов с обеими оценками (Vina+gnina) - "
              f"мало для консенсуса, нужно хотя бы 10"); sys.exit(1)

    # ранг 0 = лучший. Vina: меньше (отрицательнее) = лучше. gnina: больше = лучше.
    for i, r in enumerate(sorted(scored, key=lambda r: r["docking_score_kcal_mol"])):
        r["_vina_rank"] = i
    for i, r in enumerate(sorted(scored, key=lambda r: -r["gnina_cnn_score"])):
        r["_gnina_rank"] = i
    for r in scored:
        r["_consensus_rank_sum"] = r["_vina_rank"] + r["_gnina_rank"]
    scored.sort(key=lambda r: r["_consensus_rank_sum"])

    n_top = max(1, int(len(scored) * top_fraction))
    shortlist = scored[:n_top]
    print(f"[funnel] консенсус (сумма рангов Vina+gnina) посчитан на {len(scored)} лигандах, "
          f"беру топ-{top_fraction*100:.0f}% = {n_top} на доуточнение (exhaustiveness={exhaustiveness})")

    shortlist_path = os.path.join(d, "funnel_shortlist.json")
    with open(shortlist_path, "w", encoding="utf-8") as f:
        json.dump([{"chembl_id": r["chembl_id"], "smiles": r["smiles"], "label": r["label"],
                    "stage1_vina": r["docking_score_kcal_mol"], "stage1_gnina": r["gnina_cnn_score"],
                    "consensus_rank_sum": r["_consensus_rank_sum"]} for r in shortlist],
                  f, indent=2)

    receptor = receptor_pdbqt_path(gene)
    ligand_struct = ligand_structure_pdb_path(gene)
    ligand_info = find_ligand_center(ligand_struct)
    box_center, box_size = ligand_info["center"], ligand_info["box_size"]

    funnel_results_path = os.path.join(d, "funnel_results.jsonl")
    already_done = load_completed_ids(funnel_results_path)
    todo = [r for r in shortlist if r["chembl_id"] not in already_done]
    print(f"[funnel] уже доуточнено ранее: {len(shortlist) - len(todo)}, осталось: {len(todo)}")
    if not todo:
        print("[funnel] всё уже доуточнено."); return

    workdir = os.path.join(d, "funnel_dock_tmp")
    os.makedirs(workdir, exist_ok=True)
    cpu = cpu_per_worker(n_workers)
    # выше exhaustiveness -> дольше на лиганд, чем в основной фазе dock -
    # без явного timeout берём с ещё большим запасом (было 600, тоже
    # часто не хватало на exhaustiveness=32 - см. 231/270 успеха с
    # timeout=400 на реальном прогоне PIK3CA).
    effective_timeout = timeout if timeout is not None else 900
    print(f"[funnel] воркеров: {n_workers}, --cpu на процесс: {cpu if cpu else 'не ограничено'}, "
          f"exhaustiveness={exhaustiveness}, timeout={effective_timeout}с")

    tasks = [
        (r["chembl_id"], r["smiles"], r["label"], box_center, box_size,
         receptor, workdir, f"funnel_{gene}_{r['chembl_id']}", cpu, effective_timeout, True, exhaustiveness)
        for r in todo
    ]

    t_start = time.time()
    n_done_this_run = 0
    with open(funnel_results_path, "a", encoding="utf-8") as out_f:
        if n_workers == 1:
            for task in tasks:
                if check_control():
                    print("[control] control.json просит остановиться."); break
                res = _dock_one(task)
                out_f.write(json.dumps(res, ensure_ascii=False) + "\n")
                out_f.flush()
                n_done_this_run += 1
                if n_done_this_run % 10 == 0:
                    elapsed = time.time() - t_start
                    rate = n_done_this_run / elapsed
                    eta_sec = (len(tasks) - n_done_this_run) / rate if rate else 0
                    print(f"  [{n_done_this_run}/{len(tasks)}] {elapsed/60:.1f} мин прошло, ETA ~{eta_sec/60:.1f} мин")
        else:
            with multiprocessing.Pool(processes=n_workers) as pool:
                for res in pool.imap_unordered(_dock_one, tasks):
                    out_f.write(json.dumps(res, ensure_ascii=False) + "\n")
                    out_f.flush()
                    n_done_this_run += 1
                    if n_done_this_run % 10 == 0:
                        elapsed = time.time() - t_start
                        rate = n_done_this_run / elapsed
                        eta_sec = (len(tasks) - n_done_this_run) / rate if rate else 0
                        print(f"  [{n_done_this_run}/{len(tasks)}] {elapsed/60:.1f} мин прошло, ETA ~{eta_sec/60:.1f} мин")
                        if check_control():
                            print("[control] control.json просит остановиться, завершаю текущую партию.")
                            pool.terminate()
                            break

    print(f"\n[funnel] Доуточнение: {n_done_this_run} лигандов за {(time.time()-t_start)/60:.1f} мин")
    print(f"[funnel] Результаты: {funnel_results_path}")
    print(f"[funnel] Короткий список (до доуточнения): {shortlist_path}")


def report_funnel_top(gene, top_n=20):
    """Печатает финальный консенсус-топ ПОСЛЕ доуточнения (funnel_results.jsonl,
    exhaustiveness=32) - это и есть итоговый список кандидатов, которому
    стоит доверять больше всего в текущем пайплайне."""
    d = run_dir(gene)
    funnel_results_path = os.path.join(d, "funnel_results.jsonl")
    if not os.path.exists(funnel_results_path):
        print(f"ОШИБКА: нет {funnel_results_path} - сначала запусти фазу funnel"); sys.exit(1)

    by_id = {}
    with open(funnel_results_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            cid = rec["chembl_id"]
            prev = by_id.get(cid)
            if prev is None or rec.get("docking_score_kcal_mol") is not None:
                by_id[cid] = rec

    scored = [r for r in by_id.values()
              if r.get("docking_score_kcal_mol") is not None and r.get("gnina_cnn_score") is not None]
    if not scored:
        print("Нет ни одного доуточнённого лиганда с обеими оценками."); return

    for i, r in enumerate(sorted(scored, key=lambda r: r["docking_score_kcal_mol"])):
        r["_vina_rank"] = i
    for i, r in enumerate(sorted(scored, key=lambda r: -r["gnina_cnn_score"])):
        r["_gnina_rank"] = i
    for r in scored:
        r["_consensus_rank_sum"] = r["_vina_rank"] + r["_gnina_rank"]
    scored.sort(key=lambda r: r["_consensus_rank_sum"])

    print(f"\n=== ФИНАЛЬНЫЙ КОНСЕНСУС-ТОП после доуточнения (exhaustiveness={FUNNEL_EXHAUSTIVENESS}), {gene} ===")
    print(f"{'#':>3} {'chembl_id':<16} {'label':>5} {'vina':>8} {'gnina':>7}  smiles")
    for i, r in enumerate(scored[:top_n], 1):
        print(f"{i:>3} {r['chembl_id']:<16} {r['label']:>5} {r['docking_score_kcal_mol']:>8.2f} "
              f"{r['gnina_cnn_score']:>7.3f}  {r['smiles']}")
    n_active = sum(1 for r in scored[:top_n] if r["label"] == 1)
    print(f"\nактивных (уже известных) в топ-{top_n}: {n_active}/{top_n}")


# ============================== ФАЗА 3: analyze ==============================

def phase_analyze(gene):
    import pandas as pd

    d = run_dir(gene)
    results_path = os.path.join(d, "results.jsonl")
    if not os.path.exists(results_path):
        print(f"ОШИБКА: нет {results_path} - сначала запусти фазу dock"); sys.exit(1)

    # Дедупликация по chembl_id: при retry (см. phase_dock) старые
    # score=None записи для передокованных лигандов остаются в файле
    # (append, не перезапись) - без дедупа они попали бы в анализ как
    # лишние NaN-строки дважды на один и тот же лиганд. Правило: если
    # для chembl_id есть хоть одна запись с непустым score - берём
    # последнюю такую; иначе берём последнюю запись как есть (значит,
    # лиганд так и не задокировался ни разу).
    by_id = {}
    with open(results_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            cid = rec["chembl_id"]
            prev = by_id.get(cid)
            if prev is None or rec.get("docking_score_kcal_mol") is not None:
                by_id[cid] = rec
    rows = list(by_id.values())
    df = pd.DataFrame(rows)
    n_total = len(df)
    n_scored = df["docking_score_kcal_mol"].notna().sum()
    print(f"[analyze] всего лигандов в results.jsonl: {n_total}, успешно задокировано: {n_scored}")
    if n_scored < n_total:
        print(f"[analyze] [warn] {n_total - n_scored} лигандов не задокировались (таймаут/ошибка подготовки) - "
              f"это ожидаемо для части молекул, но если доля большая - стоит разобраться, прежде чем доверять BEDROC")

    alpha = 20.0  # стандартный alpha для "early recognition" BEDROC (Truchon & Bayly)
    report = bc.run_test_a(df, "docking_score_kcal_mol", "label", alpha=alpha, n_bootstrap=1000, n_permutations=1000)
    test_n = bc.run_test_n_shuffle(df, "docking_score_kcal_mol", "label", alpha=alpha, n_permutations=1000)
    test_a_prime = bc.run_test_a_prime_size_bias(df, "docking_score_kcal_mol", "heavy_atom_count", r2_threshold=0.3)

    full_report = {"gene": gene, "n_total": n_total, "n_scored": int(n_scored),
                   "test_a": report, "test_n_summary": {k: v for k, v in test_n.items() if k != "values"},
                   "test_a_prime": test_a_prime}

    # gnina CNN-рескоринг, если есть (см. gnina_rescore=True в phase_dock).
    # bedroc_calibration._scores_matrix сортирует по score_col ASCENDING
    # (для Vina "меньше=лучше" - подходит как есть), а у CNNscore/CNNaffinity
    # ЗНАК ОБРАТНЫЙ (больше=лучше, это вероятность/предсказанное сродство) -
    # инвертируем знак перед тем же общим кодом, а не переписываем метрику.
    gnina_report = None
    if "gnina_cnn_score" in df.columns and df["gnina_cnn_score"].notna().sum() >= 10:
        df["_neg_gnina_cnn_score"] = -df["gnina_cnn_score"]
        n_gnina_scored = df["gnina_cnn_score"].notna().sum()
        print(f"[analyze] gnina CNN-скор доступен для {n_gnina_scored}/{n_total} лигандов - считаю BEDROC отдельно")
        gnina_report = bc.run_test_a(df, "_neg_gnina_cnn_score", "label", alpha=alpha, n_bootstrap=1000, n_permutations=1000)
        full_report["test_a_gnina_cnn_score"] = gnina_report
        full_report["n_gnina_scored"] = int(n_gnina_scored)
    else:
        print("[analyze] gnina CNN-скор недоступен (не запускался с gnina_rescore=True) - только Vina-скор")

    out_path = os.path.join(d, "report.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)

    print(f"\n=== ОТЧЁТ Теста A ({gene}) — по Vina-скору ===")
    print(f"  BEDROC (alpha={alpha}): {report['bedroc_observed']:.3f} "
          f"(случайный базовый уровень: {report['bedroc_random_baseline_empirical_mean']:.3f} "
          f"+/- {report['bedroc_random_baseline_empirical_std']:.3f})")
    print(f"  95% CI (бутстрап): [{report['bedroc_bootstrap_ci_95_lo']:.3f}, {report['bedroc_bootstrap_ci_95_hi']:.3f}]")
    print(f"  p-value (перестановочный тест): {report['permutation_p_value']:.4f}")
    print(f"  AUC-ROC: {report['auc_roc']:.3f}")
    print(f"  EF_1%: {report.get('EF_1%'):.2f}, EF_5%: {report.get('EF_5%'):.2f}, EF_10%: {report.get('EF_10%'):.2f}")
    print(f"  Тест A' (смещение по размеру): R^2={test_a_prime.get('r2', 'н/д')}, "
          f"обнаружено смещение: {test_a_prime.get('size_bias_detected', 'н/д')}")
    if gnina_report:
        print(f"\n=== ОТЧЁТ Теста A ({gene}) — по gnina CNN-скору ===")
        print(f"  BEDROC (alpha={alpha}): {gnina_report['bedroc_observed']:.3f} "
              f"(случайный базовый уровень: {gnina_report['bedroc_random_baseline_empirical_mean']:.3f} "
              f"+/- {gnina_report['bedroc_random_baseline_empirical_std']:.3f})")
        print(f"  95% CI (бутстрап): [{gnina_report['bedroc_bootstrap_ci_95_lo']:.3f}, {gnina_report['bedroc_bootstrap_ci_95_hi']:.3f}]")
        print(f"  p-value (перестановочный тест): {gnina_report['permutation_p_value']:.4f}")
        print(f"  AUC-ROC: {gnina_report['auc_roc']:.3f}")
        print(f"  EF_1%: {gnina_report.get('EF_1%'):.2f}, EF_5%: {gnina_report.get('EF_5%'):.2f}, EF_10%: {gnina_report.get('EF_10%'):.2f}")
    print(f"\nПолный отчёт: {out_path}")


def _run_build(gene, arg3, arg4):
    """Мишени с DUD-E (dude_code задан в target_screening_results.json)
    берут готовые актив/декой из dude_datasets/ (phase_build_dude) -
    arg3/arg4 тогда означают ОПЦИОНАЛЬНЫЕ потолки (для тестового прогона
    на малом масштабе перед полным, напр. build GENE 10 300). Мишени без
    DUD-E (PIK3CA) идут через ChEMBL-live + property-matched декои
    (phase_build) - там arg3/arg4 это n_actives/decoy_ratio, как раньше."""
    code = dude_code_for(gene)
    if code:
        n_act = int(arg3) if arg3 is not None else None
        n_dec = int(arg4) if arg4 is not None else None
        phase_build_dude(gene, code, n_actives_cap=n_act, n_decoys_cap=n_dec)
    else:
        n_actives = int(arg3) if arg3 is not None else 100
        decoy_ratio = int(arg4) if arg4 is not None else 30
        phase_build(gene, n_actives, decoy_ratio)


def main():
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(1)
    phase = sys.argv[1]
    gene = sys.argv[2] if len(sys.argv) > 2 else "PIK3CA"
    try:
        _load_confirmed(gene)
    except RuntimeError as e:
        print(f"ОШИБКА: {e}")
        sys.exit(1)

    arg3 = sys.argv[3] if len(sys.argv) > 3 else None
    arg4 = sys.argv[4] if len(sys.argv) > 4 else None

    if phase == "build":
        _run_build(gene, arg3, arg4)
    elif phase == "dock":
        n_workers = int(arg3) if arg3 is not None else 6
        timeout = int(arg4) if arg4 is not None else None
        gnina = sys.argv[5] != "0" if len(sys.argv) > 5 else True
        phase_dock(gene, n_workers, timeout=timeout, gnina=gnina)
    elif phase == "analyze":
        phase_analyze(gene)
    elif phase == "funnel":
        n_workers = int(arg3) if arg3 is not None else 6
        timeout = int(arg4) if arg4 is not None else None
        top_fraction = float(sys.argv[5]) if len(sys.argv) > 5 else FUNNEL_TOP_FRACTION
        phase_funnel(gene, n_workers, top_fraction=top_fraction, timeout=timeout)
    elif phase == "top":
        top_n = int(arg3) if arg3 is not None else 20
        report_funnel_top(gene, top_n=top_n)
    elif phase == "all":
        n_workers = int(sys.argv[5]) if len(sys.argv) > 5 else 6
        _run_build(gene, arg3, arg4)
        phase_dock(gene, n_workers)
        phase_analyze(gene)
    else:
        print(__doc__); sys.exit(1)


if __name__ == "__main__":
    main()
