"""
run_all_chembl_builds.py - строит ChEMBL-live датасет (~10000 лигандов
на мишень: активные + property-matched декои) для ВСЕХ 11 подтверждённых
мишеней, одна за другой. Заменяет собой DUD-E-построение для BRAF/CDK2
(единообразие источника декоев по явному запросу пользователя - старые
DUD-E-based ligands.json/results.jsonl забэкаплены в
runs/_pre_chembl_live_backup/ перед перезаписью).

n_actives=300, decoy_ratio=34 -> цель ~10200 лигандов на мишень (не для
всех мишеней реально наберётся столько активных/декоев - логируется
честно, не подгоняется).

Использование:
    python run_all_chembl_builds.py
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_test_a as rta  # noqa: E402

N_ACTIVES = 300
DECOY_RATIO = 34  # 300*34 = 10200 целевых лигандов

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROGRESS_PATH = os.path.join(BASE_DIR, "runs", "_chembl_builds_progress.json")


def load_progress():
    if os.path.exists(PROGRESS_PATH):
        with open(PROGRESS_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_progress(progress):
    with open(PROGRESS_PATH, "w", encoding="utf-8") as f:
        json.dump(progress, f, indent=2, ensure_ascii=False)


def main():
    with open(os.path.join(BASE_DIR, "confirmed_structures.json"), encoding="utf-8") as f:
        genes = list(json.load(f).keys())

    progress = load_progress()
    print(f"[chembl_builds] мишеней всего: {len(genes)}, уже сделано: "
          f"{sum(1 for g in genes if progress.get(g) == 'done')}")

    for gene in genes:
        if progress.get(gene) == "done":
            print(f"[chembl_builds] {gene}: уже собран, пропускаю")
            continue
        print(f"\n{'='*60}\n[chembl_builds] {gene}: старт ({time.strftime('%H:%M:%S')})\n{'='*60}")
        t0 = time.time()
        try:
            rta.phase_build(gene, N_ACTIVES, DECOY_RATIO)
            progress[gene] = "done"
        except SystemExit:
            print(f"[chembl_builds] {gene}: phase_build вызвал sys.exit (см. вывод выше) - "
                  f"недостаточно активных/декоев для этой мишени, помечаю как failed")
            progress[gene] = "failed"
        except Exception as e:
            print(f"[chembl_builds] {gene}: ОШИБКА {e}")
            progress[gene] = "error"
        save_progress(progress)
        print(f"[chembl_builds] {gene}: заняло {(time.time()-t0)/60:.1f} мин")

    print(f"\n[chembl_builds] ВСЁ ГОТОВО: {progress}")


if __name__ == "__main__":
    main()
