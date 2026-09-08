"""
target_screening.py — шаг 1 отчёта: скрининг 15 кандидатных мишеней
через живой ChEMBL API + RCSB API. Не выбирает структуры автоматически
(проект уже наступал на эти грабли — 9CMK вместо 4JPS для PIK3CA) —
только собирает кандидатов и данные для ручного решения.
"""
import json
import sys
import time

sys.path.insert(0, r"C:\Users\adilh\Desktop\onco-target-explorer")
from gene_target_utils import get_chembl_new_client, GeneTargetError, _request_with_retry, RCSB_SEARCH_URL

UNIPROT_SEARCH_URL = "https://rest.uniprot.org/uniprotkb/search"


def find_uniprot_accession(gene_name):
    """Независимый от ChEMBL источник UniProt accession — напрямую из
    UniProt REST API. Нужен, потому что раньше accession брался только
    из chembl target.target_components, и весь RCSB-поиск (которому нужен
    только accession, не ChEMBL) вставал колом при падении ChEMBL,
    хотя сам RCSB был живой. UniProt — отдельный бэкенд, не chembl_webservices_2."""
    resp = _request_with_retry(
        "get", UNIPROT_SEARCH_URL, timeout=20,
        params={
            "query": f"gene:{gene_name} AND organism_id:9606 AND reviewed:true",
            "format": "json", "fields": "accession,protein_name",
        },
    )
    if resp.status_code != 200:
        raise GeneTargetError(f"UniProt вернул статус {resp.status_code} для {gene_name}")
    results = resp.json().get("results", [])
    if not results:
        raise GeneTargetError(f"UniProt: ничего не найдено для {gene_name}")
    return results[0]["primaryAccession"]


def probe_chembl_available():
    """Один прямой пробный запрос перед всем циклом по 15 генам — чтобы
    не тратить retry-бюджет (3 попытки x 8с в get_chembl_new_client(),
    вызывается по 4 раза на ген) на сервис, который уже не отвечает.
    Если ChEMBL сейчас лежит - переключаемся на UniProt-only режим для
    accession и полностью пропускаем блок активностей (актив/неактив),
    оставляя эти поля None с пометкой причины."""
    try:
        get_chembl_new_client()
        return True
    except Exception as e:
        print(f"[probe] ChEMBL сейчас недоступен ({str(e)[:150]}...) — "
              f"переключаюсь на UniProt-only режим, блок активностей пропускается для всех генов.")
        return False

GENES = ["ESR1", "AR", "CA2", "PIK3CA", "CDK2", "EGFR", "ABL1", "HSP90AA1",
         "PARP1", "BRD4", "HDAC2", "PPARG", "MDM2", "BCL2", "BRAF"]

# DUD-E (dude.docking.org/targets) использует свои краткие коды, не всегда
# совпадающие с HGNC gene symbol - сопоставлено вручную по официальному списку.
DUDE_CODE_BY_GENE = {
    "ESR1": "ESR1", "AR": "ANDR", "CA2": "CAH2", "CDK2": "CDK2",
    "EGFR": "EGFR", "ABL1": "ABL1", "HSP90AA1": "HS90A", "PARP1": "PARP1",
    "HDAC2": "HDAC2", "PPARG": "PPARG", "BRAF": "BRAF",
    # не найдены в DUD-E: PIK3CA, BRD4, MDM2, BCL2
}


def _with_retry(fn, attempts=5, delay_sec=8.0, what=""):
    """EBI/ChEMBL под нагрузкой изредка (а сегодня не так уж и изредка)
    отдаёт транзиентные 500 — тот же принцип, что _chembl_call_with_retry
    в module_generative/iterative_finetune_loop.py, просто локально в этом
    скрипте, чтобы не тащить импорт из модуля с тяжёлыми MolGPT-зависимостями."""
    last_exc = None
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as e:
            last_exc = e
            if attempt < attempts:
                print(f"    [retry {attempt}/{attempts}] {what}: {str(e)[:150]}... жду {delay_sec:.0f}с")
                time.sleep(delay_sec)
    raise last_exc


def find_chembl_target(gene_name):
    """БАГ, найденный на реальном прогоне: icontains на КОРОТКИХ символах
    гена (напр. "AR") матчит подстроку ВНУТРИ любого синонима - "AR" как
    подстрока встречается в "carbonic" (c-AR-bonic), из-за чего AR и CA2
    получили ОДИНАКОВЫЙ результат (CHEMBL205, Carbonic Anhydrase 2) -
    полностью не относящаяся к андрогенному рецептору мишень, молча
    выбранная первой из списка. Теперь сначала пробуем ТОЧНОЕ совпадение
    синонима (iexact) - в ChEMBL символ гена почти всегда есть отдельным
    точным алиасом - и только если пусто, откатываемся на icontains
    с явным предупреждением, что совпадение слабое."""
    new_client = get_chembl_new_client()
    target = new_client.target

    candidates = _with_retry(
        lambda: list(
            target.filter(target_synonym__iexact=gene_name, organism="Homo sapiens")
            .only(["target_chembl_id", "pref_name", "target_type", "organism"])
        ),
        what=f"target lookup (exact) {gene_name}",
    )
    weak_match = False
    if not candidates:
        weak_match = True
        print(f"    [warn] точного совпадения синонима для {gene_name} нет - "
              f"откатываюсь на подстроковый поиск (слабее, легко ложное совпадение "
              f"на коротких символах гена)")
        candidates = _with_retry(
            lambda: list(
                target.filter(target_synonym__icontains=gene_name, organism="Homo sapiens")
                .only(["target_chembl_id", "pref_name", "target_type", "organism"])
            ),
            what=f"target lookup (substring fallback) {gene_name}",
        )
    if not candidates:
        raise GeneTargetError(f"нет мишени для {gene_name}")
    single_protein = [c for c in candidates if c.get("target_type") == "SINGLE PROTEIN"]
    chosen = single_protein[0] if single_protein else candidates[0]
    full = _with_retry(lambda: target.get(chosen["target_chembl_id"]), what=f"target details {gene_name}")
    uniprot = None
    for comp in full.get("target_components", []):
        if comp.get("accession"):
            uniprot = comp["accession"]
            break

    # Независимая перекрёстная проверка через UniProt (другой бэкенд,
    # не подвержен той же ошибке ChEMBL-поиска по синонимам) - если не
    # совпадает с тем, что вернул ChEMBL, это явный сигнал неправильного
    # match'а (как было с AR/CA2), а не просто "разные источники,
    # разные ID".
    accession_mismatch = False
    try:
        uniprot_independent = find_uniprot_accession(gene_name)
        if uniprot and uniprot_independent and uniprot != uniprot_independent:
            accession_mismatch = True
            print(f"    [ОШИБКА] ChEMBL UniProt ({uniprot}) НЕ совпадает с независимым "
                  f"UniProt-поиском ({uniprot_independent}) для {gene_name} - похоже на "
                  f"неправильный match ChEMBL-мишени, ЭТУ ЗАПИСЬ НАДО ПРОВЕРИТЬ ВРУЧНУЮ")
    except Exception:
        pass  # сверка best-effort - её отказ не должен ронять основной поиск

    return {
        "target_chembl_id": chosen["target_chembl_id"],
        "pref_name": chosen.get("pref_name"),
        "uniprot_accession": uniprot,
        "weak_synonym_match": weak_match,
        "uniprot_accession_mismatch": accession_mismatch,
    }


def count_distinct_molecules(activity_qs, limit=5000):
    """Считает УНИКАЛЬНЫЕ molecule_chembl_id, с защитным потолком на число
    записей (чтобы не зависнуть на очень изученных мишенях типа EGFR) —
    если упёрлись в потолок, помечаем как 'limit_hit', число будет
    заниженной оценкой снизу, но для решения '>=50 активных?' этого
    достаточно."""
    seen = set()
    hit_limit = False
    for i, rec in enumerate(activity_qs):
        seen.add(rec["molecule_chembl_id"])
        if i + 1 >= limit:
            hit_limit = True
            break
    return len(seen), hit_limit


def screen_gene(gene, chembl_available):
    print(f"\n=== {gene} ===")
    row = {"gene": gene, "chembl_available_at_screening_time": chembl_available}

    info = None
    if chembl_available:
        try:
            info = find_chembl_target(gene)
        except GeneTargetError as e:
            print(f"  ChEMBL target не найден ({e}) — пробую UniProt напрямую")
        except Exception as e:
            print(f"  ChEMBL упал на этом гене ({str(e)[:150]}...) — пробую UniProt напрямую")

    if info is not None:
        row.update(info)
        print(f"  ChEMBL: {info['target_chembl_id']} ({info['pref_name']}), UniProt {info['uniprot_accession']}")
    else:
        # ChEMBL целиком лежит или конкретно этот ген не нашёлся - accession
        # берём из UniProt напрямую, это независимый бэкенд. target_chembl_id/
        # pref_name останутся None - без них не сходить за activity ниже,
        # но PDB-поиску они не нужны.
        row["target_chembl_id"] = None
        row["pref_name"] = None
        try:
            row["uniprot_accession"] = find_uniprot_accession(gene)
            print(f"  UniProt (напрямую, ChEMBL недоступен): {row['uniprot_accession']}")
        except Exception as e:
            row["uniprot_accession"] = None
            row["error"] = f"ни ChEMBL, ни UniProt не дали accession: {e}"
            print(f"  ОШИБКА: {row['error']}")
            row["pdb_candidates"] = []
            row["reject_reasons"] = ["нет UniProt accession ни из одного источника"]
            row["passes_screening"] = False
            row["dude_code"] = DUDE_CODE_BY_GENE.get(gene)
            row["in_dude"] = gene in DUDE_CODE_BY_GENE
            return row

    n_active = 0
    n_inactive = 0
    if not chembl_available or row.get("target_chembl_id") is None:
        row["n_compounds_with_pchembl"] = None
        row["n_active_ic50_100nm"] = None
        row["n_inactive_pchembl_lt5"] = None
        row["activity_data_status"] = (
            "chembl_unavailable_at_screening_time" if not chembl_available else "no_chembl_target_id"
        )
        print(f"  блок активностей (Тест A: актив/неактив) пропущен — "
              f"{row['activity_data_status']}, довзять при повторном скрининге позже")
        row["ratio_1_30_achievable"] = None
        row["min_50_actives_ok"] = None
    else:
        new_client = get_chembl_new_client()
        activity = new_client.activity

        # 1. compounds с pChEMBL вообще
        try:
            n_pchembl, hit1 = _with_retry(
                lambda: count_distinct_molecules(
                    activity.filter(target_chembl_id=row["target_chembl_id"], pchembl_value__isnull=False)
                    .only(["molecule_chembl_id"])
                ),
                what=f"pChEMBL count {gene}",
            )
            row["n_compounds_with_pchembl"] = n_pchembl
            row["n_compounds_with_pchembl_limit_hit"] = hit1
            print(f"  соединений с pChEMBL: {n_pchembl}{'+ (упёрлись в потолок 5000)' if hit1 else ''}")
        except Exception as e:
            row["n_compounds_with_pchembl"] = None
            print(f"  ОШИБКА запроса pChEMBL (после ретраев): {str(e)[:200]}")
        time.sleep(2)

        # 2. активные: IC50 <= 100 нМ
        try:
            n_active, hit2 = _with_retry(
                lambda: count_distinct_molecules(
                    activity.filter(
                        target_chembl_id=row["target_chembl_id"], standard_type="IC50",
                        standard_units="nM", standard_value__lte=100,
                    ).only(["molecule_chembl_id"])
                ),
                what=f"active count {gene}",
            )
            row["n_active_ic50_100nm"] = n_active
            row["n_active_limit_hit"] = hit2
            print(f"  активных (IC50<=100нМ): {n_active}{'+ (потолок)' if hit2 else ''}")
        except Exception as e:
            row["n_active_ic50_100nm"] = None
            print(f"  ОШИБКА запроса активных (после ретраев): {str(e)[:200]}")
        time.sleep(2)

        # 3. неактивные: pChEMBL < 5
        try:
            n_inactive, hit3 = _with_retry(
                lambda: count_distinct_molecules(
                    activity.filter(
                        target_chembl_id=row["target_chembl_id"], pchembl_value__lt=5,
                        pchembl_value__isnull=False,
                    ).only(["molecule_chembl_id"])
                ),
                what=f"inactive count {gene}",
            )
            row["n_inactive_pchembl_lt5"] = n_inactive
            row["n_inactive_limit_hit"] = hit3
            print(f"  неактивных (pChEMBL<5): {n_inactive}{'+ (потолок)' if hit3 else ''}")
        except Exception as e:
            row["n_inactive_pchembl_lt5"] = None
            print(f"  ОШИБКА запроса неактивных (после ретраев): {str(e)[:200]}")
        time.sleep(2)

        # 4. достижимо ли 1:30 при min 50 активных.
        # БАГ, найденный на реальном прогоне: `or 0` ниже склеивал
        # "запрос не удался после ретраев (None)" с "подтверждённо
        # 0 активных" - в результате ESR1/PIK3CA/ABL1/BRD4 (у ChEMBL
        # target resolve прошёл, а activity-запросы потом упали после
        # 4 ретраев) ложно помечались "активных < 50 (0)" и жёстко
        # отсеивались, хотя на самом деле это все известные, хорошо
        # изученные мишени с тысячами активных в ChEMBL - данные просто
        # не удалось получить В ЭТОТ РАЗ. Теперь None остаётся None
        # (статус "не проверено", уходит в pending, не в hard reject).
        n_active_raw = row.get("n_active_ic50_100nm")
        n_inactive_raw = row.get("n_inactive_pchembl_lt5")
        if n_active_raw is None or n_inactive_raw is None:
            row["ratio_1_30_achievable"] = None
            row["min_50_actives_ok"] = None
            print(f"  >=50 активных: не проверено (запрос активных/неактивных не удался после ретраев)")
        else:
            row["ratio_1_30_achievable"] = bool(n_active_raw >= 50 and n_inactive_raw >= n_active_raw * 30)
            row["min_50_actives_ok"] = bool(n_active_raw >= 50)
            print(f"  >=50 активных: {row['min_50_actives_ok']}, достижимо 1:30: {row['ratio_1_30_achievable']}")

    # 5. PDB структуры с лигандом, разрешение <2.5А
    row["pdb_candidates"] = []
    if row["uniprot_accession"]:
        try:
            query = {
                "query": {
                    "type": "group", "logical_operator": "and",
                    "nodes": [
                        {"type": "terminal", "service": "text", "parameters": {
                            "attribute": "rcsb_polymer_entity_container_identifiers.reference_sequence_identifiers.database_accession",
                            "operator": "exact_match", "value": row["uniprot_accession"]}},
                        {"type": "terminal", "service": "text", "parameters": {
                            "attribute": "rcsb_polymer_entity_container_identifiers.reference_sequence_identifiers.database_name",
                            "operator": "exact_match", "value": "UniProt"}},
                        {"type": "terminal", "service": "text", "parameters": {
                            "attribute": "rcsb_entry_info.resolution_combined",
                            "operator": "less", "value": 2.5}},
                        {"type": "terminal", "service": "text", "parameters": {
                            "attribute": "rcsb_entry_info.nonpolymer_entity_count",
                            "operator": "greater", "value": 0}},
                    ],
                },
                "return_type": "entry",
                "request_options": {"paginate": {"start": 0, "rows": 8},
                                     "sort": [{"sort_by": "rcsb_entry_info.resolution_combined", "direction": "asc"}]},
            }
            resp = _request_with_retry("post", RCSB_SEARCH_URL, json=query, timeout=30)
            if resp.status_code == 200:
                hits = [h["identifier"] for h in resp.json().get("result_set", [])]
                row["pdb_candidates"] = hits
                print(f"  PDB-кандидаты (разрешение <2.5А, есть гетеро-группа): {hits}")
            elif resp.status_code == 204:
                print("  PDB-кандидатов не найдено (204)")
            else:
                print(f"  RCSB вернул статус {resp.status_code}")
        except Exception as e:
            print(f"  ОШИБКА запроса RCSB: {e}")
    else:
        print("  нет UniProt accession — PDB-поиск пропущен")

    # 6. DUD-E
    row["dude_code"] = DUDE_CODE_BY_GENE.get(gene)
    row["in_dude"] = gene in DUDE_CODE_BY_GENE
    print(f"  в DUD-E: {row['in_dude']}" + (f" (код {row['dude_code']})" if row["in_dude"] else ""))

    # 7. причина отсева. min_50_actives_ok is None означает "неизвестно,
    # ChEMBL был недоступен" - это НЕ повод отсеивать мишень навсегда,
    # просто пометка "довзять активности при повторном скрининге".
    reasons = []
    pending = []
    if row.get("min_50_actives_ok") is False:
        reasons.append(f"активных < 50 ({n_active})")
    elif row.get("min_50_actives_ok") is None:
        pending.append("число активных не проверено (ChEMBL был недоступен)")
    if not row["pdb_candidates"]:
        reasons.append("нет PDB-структуры с разрешением <2.5А и лигандом")
    row["reject_reasons"] = reasons
    row["pending_recheck_reasons"] = pending
    row["passes_screening"] = len(reasons) == 0 and len(pending) == 0
    row["passes_screening_pdb_only"] = len(reasons) == 0
    print(f"  ПРОШЛА ОТБОР: {row['passes_screening']}"
          + (f" (причины отсева: {reasons})" if reasons else "")
          + (f" (требует довскрининга: {pending})" if pending else ""))

    return row


def main():
    chembl_available = probe_chembl_available()
    all_rows = []
    for gene in GENES:
        row = screen_gene(gene, chembl_available)
        all_rows.append(row)
        time.sleep(3)  # не долбить RCSB/UniProt подряд без пауз

    out_path = r"C:\Users\adilh\AppData\Local\Temp\claude\C--Users-adilh\432ec4d2-31e7-4c13-ab8a-908f44f6e3ba\scratchpad\target_screening_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_rows, f, ensure_ascii=False, indent=2)
    print(f"\n\nСохранено: {out_path}")
    if not chembl_available:
        print("\nВНИМАНИЕ: ChEMBL был недоступен весь прогон. PDB-кандидаты и UniProt "
              "accession для всех генов получены и надёжны, но активность/неактивность "
              "(нужна для Теста A) не проверена ни для одной мишени — скрипт нужно "
              "перезапустить позже, когда ChEMBL восстановится, чтобы дозаполнить эти поля.")


if __name__ == "__main__":
    main()
