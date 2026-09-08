"""
Test 16 (Stage 6, по явной команде) — PoseCheck (Harris et al. 2023):
стерические клэши + энергия напряжения лиганда (UFF) для СГЕНЕРИРОВАННЫХ
молекул.

КРИТИЧНО: должно считаться на СЫРЫХ позах генеративной модели ДО
редокинга - редокинг чинит ровно те дефекты, которые этот тест должен
обнаружить, и тест на передокованных позах будет бессмысленным
(искусственно "красивым").

СТАТУС ДАННЫХ: DiffSBDD/generate_smiles_seeded.py уже умеет генерировать
и сохранять СЫРЫЕ (relax_iter=0, без пост-обработки) 3D-позы в SDF - см.
controls/run_generation_test_b.py, который его вызывает и складывает
результат в runs/test_b_<GENE>/. Если для GENE такой генерации ещё не
было - этот скрипт явно говорит, что данных нет, вместо того чтобы
подставлять заглушки.

Использование:
    python controls/test16_posecheck.py GENE --poses-dir PATH/TO/SDF_OR_PDB_DIR --protein PATH/TO/PROTEIN.pdb
    (без --poses-dir команда только объясняет, чего не хватает, и выходит)
"""
import argparse
import os
import sys

# posecheck внутри вызывает hydride.exe (протонирование белка) через
# subprocess БЕЗ полного пути - когда python.exe запускается не через
# `conda activate` (полным путём напрямую, как везде в этом проекте), Scripts/
# окружения не попадает в PATH, hydride не находится, subprocess падает
# с ошибкой ОС не в UTF-8 (не расшифровывается посередине posecheck же
# кода) - маскируя настоящую причину под UnicodeDecodeError. Чиним здесь,
# один раз, при любом способе запуска этого скрипта.
_env_scripts_dir = os.path.join(os.path.dirname(sys.executable), "Scripts")
if _env_scripts_dir not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _env_scripts_dir + os.pathsep + os.environ.get("PATH", "")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import protocol  # noqa: E402


def explain_missing_data(gene):
    msg = (
        f"[test16] {gene}: НЕТ ДАННЫХ ДЛЯ ЭТОГО ТЕСТА.\n"
        f"  PoseCheck нужен набор СЫРЫХ 3D-поз, которые генеративная модель выдала\n"
        f"  ДО редокинга/доработки - в проекте сейчас нет отдельного этапа 'Test B'\n"
        f"  (массовая генерация с сохранением координат), только module_generative/\n"
        f"  iterative_finetune_loop.py, который оценивает молекулы по докинг-скору,\n"
        f"  а не сохраняет сырые сгенерированные конформации в файл.\n"
        f"  Чтобы посчитать этот тест по-настоящему, нужно:\n"
        f"    1) запустить генерацию (DiffSBDD/molgen) с явным сохранением каждой\n"
        f"       сгенерированной 3D-структуры (SDF/PDB) ДО какого-либо редокинга;\n"
        f"    2) передать путь к этой директории через --poses-dir.\n"
        f"  Никаких выдуманных чисел вместо этого не считаю."
    )
    print(msg)
    return {"error": "нет сохранённых сырых генеративных поз (Test B не запускался)", "explanation": msg}


def run_on_poses(gene, poses_dir, protein_path, force=False):
    if protocol.test_already_done(gene, 6, "test16_posecheck", force):
        print(f"[test16] {gene}: уже посчитано, пропускаю. --force для пересчёта")
        return protocol.load_stage(gene, 6)["test16_posecheck"]

    protocol.print_banner("test16")
    from posecheck import PoseCheck
    import glob

    # posecheck.utils.loading.load_protein_from_pdb() удаляет свой временный
    # протонированный PDB через os.remove() сразу после чтения его MDAnalysis -
    # на Windows файловый хендл MDAnalysis иногда ещё не отпущен ОС в этот
    # момент, os.remove кидает PermissionError [WinError 32] (на Linux/Mac,
    # судя по всему, не воспроизводится - иначе апстрим это бы уже поймал).
    # Это баг в чужом пакете, не в нашем коде - патчим точечно os.remove
    # ТОЛЬКО внутри модуля posecheck.utils.loading, чтобы просто не падать
    # на недостающем удалении временного файла (сам файл безвреден, лежит
    # в системном temp и будет убран ОС).
    import posecheck.utils.loading as _pc_loading

    def _safe_remove(path, _orig=os.remove):
        try:
            _orig(path)
        except PermissionError:
            pass

    _pc_loading.os.remove = _safe_remove

    # ВАЖНО: НЕ сканируем poses_dir напрямую и не работаем с файлами по месту.
    # posecheck.utils.loading.load_mols_from_sdf() сама пишет "<имя>_tmp.sdf"
    # РЯДОМ с каждым входным SDF (в ту же директорию) и в конце его удаляет -
    # но на Windows os.remove там периодически падает по той же гонке с
    # файловым хендлом, что и для белка (см. патч выше), и мы её так же
    # молча глушим. Если glob сканирует ЭТУ ЖЕ директорию - оставшийся
    # "_tmp.sdf" на следующем запуске подхватывается как ещё один "новый"
    # файл поз, а его собственная обработка плодит "_tmp_tmp.sdf" и т.д. -
    # самоподдерживающееся заражение директории (поймано именно так во
    # время проверки). Копируем входные файлы в одноразовую temp-директорию
    # ДО обработки - что бы posecheck туда ни дописал, оно никогда не
    # попадёт обратно в реальную poses_dir и не переживёт этот запуск.
    import shutil
    import tempfile

    src_files = sorted(glob.glob(os.path.join(poses_dir, "*.sdf")) + glob.glob(os.path.join(poses_dir, "*.pdb")))
    if not src_files:
        result = {"error": f"в {poses_dir} не найдено .sdf/.pdb файлов"}
        protocol.save_stage(gene, 6, {"test16_posecheck": result})
        print(f"[test16] {result['error']}")
        return result

    isolated_dir = tempfile.mkdtemp(prefix="posecheck_isolated_")
    pose_files = []
    for src in src_files:
        dst = os.path.join(isolated_dir, os.path.basename(src))
        shutil.copy2(src, dst)
        pose_files.append(dst)

    pc = PoseCheck()
    pc.load_protein_from_pdb(protein_path)

    clashes, strains = [], []
    per_pose = []
    for path in pose_files:
        try:
            pc.load_ligands_from_sdf(path) if path.endswith(".sdf") else pc.load_ligands_from_pdb(path)
            n_mols_in_file = len(pc.ligands)
            # calculate_clashes()/calculate_strain_energy() возвращают СПИСОК,
            # по одному значению НА КАЖДУЮ молекулу в self.ligands - индекс [0]
            # молча брал только первую молекулу файла и терял все остальные,
            # если в одном SDF несколько поз (обычный случай для реальной
            # генерации, где на сид приходится много молекул в одном файле).
            all_clashes = pc.calculate_clashes()
            all_strains = pc.calculate_strain_energy()
            for mol_idx, (n_clashes, strain) in enumerate(zip(all_clashes, all_strains)):
                clashes.append(n_clashes); strains.append(strain)
                per_pose.append({"file": os.path.basename(path), "mol_idx": mol_idx,
                                  "n_clashes": n_clashes, "strain_energy_kcal_mol": strain})
        except Exception as e:
            per_pose.append({"file": os.path.basename(path), "error": str(e)})

    shutil.rmtree(isolated_dir, ignore_errors=True)

    import numpy as np
    result = {
        "n_files": len(pose_files), "n_poses": len(per_pose), "n_scored": len(clashes),
        "mean_clashes": float(np.mean(clashes)) if clashes else None,
        "mean_strain_energy_kcal_mol": float(np.mean(strains)) if strains else None,
        "reference_scale_note": "~1200 ккал/моль напряжения наблюдалось в некоторых генеративных моделях (Harris et al. 2023) - ориентир, не порог",
        "per_pose": per_pose,
    }
    protocol.save_stage(gene, 6, {"test16_posecheck": result})

    print(f"\n=== Test 16 ({gene}): PoseCheck (сырые генеративные позы) ===")
    print(f"  посчитано: {len(clashes)} поз из {len(pose_files)} файлов")
    if clashes:
        print(f"  средние клэши: {result['mean_clashes']:.2f}, среднее напряжение: {result['mean_strain_energy_kcal_mol']:.1f} ккал/моль")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("gene", nargs="?", default="PIK3CA")
    parser.add_argument("--poses-dir", default=None)
    parser.add_argument("--protein", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if not args.poses_dir:
        explain_missing_data(args.gene)
        return
    if not args.protein:
        print("ОШИБКА: --poses-dir передан, но не передан --protein"); sys.exit(1)
    run_on_poses(args.gene, args.poses_dir, args.protein, args.force)


if __name__ == "__main__":
    main()
