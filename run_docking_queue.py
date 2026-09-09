"""
run_docking_queue.py - главная очередь докинга: проходит по ВСЕМ мишеням
с готовым ligands.json (либо только по прошедшим Stage 1 гейт, если
confirmed_structures.json/results уже есть), докует каждую (Vina+gnina,
resume-safe - phase_dock сам пропускает уже сделанные лиганды).

Смысл: часть докинга можно сделать ЗДЕСЬ, на слабой машине, ДО переноса
кода на мощную - phase_dock's load_completed_ids() продолжит ровно с
того места, где остановились, на новой машине. Ни один лиганд не
передокуется дважды, прогресс не теряется при переносе/прерывании.

Порядок: сортирует мишени по размеру датасета (сначала маленькие) -
максимизирует число ПОЛНОСТЬЮ законченных мишеней за ограниченное время,
а не равномерный частичный прогресс по всем сразу (частичный прогресс
одинаково полезен для переноса, но полностью законченные мишени сразу
дают usable Stage 3 результаты уже здесь).

Использование:
    python run_docking_queue.py [--only-gated]
    --only-gated: докать только мишени, прошедшие Stage 1 (status=pass
                  в confirmed_structures.json/results/<GENE>/stage_1.json)
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "controls"))
import run_test_a as rta  # noqa: E402
import protocol  # noqa: E402

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
N_WORKERS, _cpu_preview = protocol.get_n_workers_and_cpu()  # автоопределение по железу + 80%-бюджету, не захардкожено


def gate_status(gene):
    path = os.path.join(BASE_DIR, "results", gene, "stage_1.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    return d.get("test05_redock", {}).get("status")


def main():
    only_gated = "--only-gated" in sys.argv

    runs_dir = os.path.join(BASE_DIR, "runs")
    genes = []
    for name in os.listdir(runs_dir):
        if name.startswith("test_a_"):
            gene = name[len("test_a_"):]
            lig_path = os.path.join(runs_dir, name, "ligands.json")
            if os.path.exists(lig_path):
                genes.append(gene)

    if only_gated:
        genes = [g for g in genes if gate_status(g) == "pass"]

    # сортировка по размеру датасета - сначала маленькие, чтобы больше
    # мишеней успело полностью задокаться за ограниченное время
    def dataset_size(gene):
        with open(os.path.join(runs_dir, f"test_a_{gene}", "ligands.json"), encoding="utf-8") as f:
            return len(json.load(f)["ligands"])
    genes.sort(key=dataset_size)

    print(f"[queue] мишеней в очереди: {len(genes)} (only_gated={only_gated})")
    for g in genes:
        print(f"  {g}: {dataset_size(g)} лигандов, gate={gate_status(g)}")

    for gene in genes:
        print(f"\n{'='*60}\n[queue] {gene}: старт докинга ({time.strftime('%H:%M:%S')})\n{'='*60}")
        t0 = time.time()
        try:
            rta.phase_dock(gene, N_WORKERS)
        except Exception as e:
            print(f"[queue] {gene}: ОШИБКА {e} - перехожу к следующей мишени")
        print(f"[queue] {gene}: {(time.time()-t0)/60:.1f} мин на этой сессии")

    print("\n[queue] Очередь пройдена целиком (или прервана) - можно перезапускать, resume сохранится")


if __name__ == "__main__":
    main()
