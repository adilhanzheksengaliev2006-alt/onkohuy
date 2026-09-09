"""
Test 5 (Stage 1) — REAL redock. ЕДИНСТВЕННЫЙ гейт всего пайплайна:
RMSD >= gates.redock_rmsd_max (config/protocol.yaml) на этой стадии -
единственная законная причина остановить обработку мишени. Все
остальные 17 тестов НИКОГДА не исключают мишень.

Переиспользует run_redock.py (загрузка нативной позы, подготовка
свежего 3D-конформера из SMILES, сам докинг) - здесь только меняем
способ извлечения поз докинга: вместо одной лучшей позы берём ВСЕ
num_modes и считаем symmetry-corrected RMSD (spyrmsd, graph isomorphism)
для каждой, чтобы отдельно репортить top-1 / top-3 / best-of-9.

Использование:
    python controls/test05_redock.py GENE [--force]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import protocol  # noqa: E402
import run_redock as rr  # noqa: E402


def all_docked_conformers(pdbqt_path, template_smiles):
    """Как run_redock.mol_from_docked_pdbqt, но БЕЗ обрезки до одного
    конформера - возвращает mol со всеми позами (по одному конформеру
    на позу, в порядке убывания качества, как их пишет Vina)."""
    from meeko import PDBQTMolecule, RDKitMolCreate
    from rdkit import Chem

    pdbqt_mol = PDBQTMolecule.from_file(pdbqt_path, skip_typing=True)
    mols = RDKitMolCreate.from_pdbqt_mol(pdbqt_mol)
    if not mols:
        raise RuntimeError(f"meeko не смогла восстановить молекулу из {pdbqt_path}")
    mol = Chem.RemoveHs(mols[0])
    if mol.GetNumConformers() == 0:
        raise RuntimeError(f"meeko вернула молекулу без конформеров из {pdbqt_path}")
    return mol


def symmetry_rmsd_per_pose(native_mol, docked_mol_multi):
    """spyrmsd symmetry-corrected RMSD (graph isomorphism) для каждой
    позы отдельно, в исходном порядке (порядок = ранг Vina, 0 = лучшая)."""
    from spyrmsd.molecule import Molecule
    from spyrmsd.rmsd import rmsdwrapper
    from rdkit import Chem

    ref = Molecule.from_rdkit(native_mol)
    poses = []
    for conf_id in range(docked_mol_multi.GetNumConformers()):
        single = Chem.Mol(docked_mol_multi, confId=conf_id)
        # обрезаем до одного конформера, чтобы from_rdkit не путался
        single = Chem.Mol(single)
        poses.append(Molecule.from_rdkit(single))
    rmsds = rmsdwrapper(ref, poses, symmetry=True, strip=True)
    return [float(r) for r in rmsds]


def _save_error(gene, exc):
    """Гейт упал ИСКЛЮЧЕНИЕМ (сеть, отсутствующий файл, что угодно) - это
    НЕ то же самое, что "гейт пройден с RMSD выше порога". Раньше при
    любом исключении здесь скрипт просто падал (sys.exit/необработанный
    traceback), stage_1.json вообще не создавался, а run_controls.py
    молча продолжал на Stage 2+ как будто гейта не существовало (реально
    случилось в бою: sys.exit(1) второй раз - но uncaught exception ведёт
    себя так же). status="error" отличает "чинить и перезапускать" от
    status="fail" ("RMSD выше порога, мишень помечена, данные остаются")."""
    result = {"gene": gene, "status": "error", "error": str(exc), "passed_gate": False}
    protocol.save_stage(gene, 1, {"test05_redock": result})
    print(f"[test05] ОШИБКА (не RMSD-провал, а сбой выполнения): {exc}")
    print(f"[test05] status=error записан в stage_1.json - это НЕ 'не прошёл гейт по RMSD', "
          f"а 'гейт не удалось выполнить' - почини причину и перезапусти")
    return result


def run(gene, force=False):
    if protocol.test_already_done(gene, 1, "test05_redock", force):
        cached = protocol.load_stage(gene, 1)["test05_redock"]
        print(f"[test05] {gene}: уже посчитано (status={cached.get('status')}, "
              f"RMSD top-1={cached.get('rmsd_top1')}), пропускаю. --force для пересчёта")
        return cached

    protocol.print_banner("test05", ["gates.redock_rmsd_max"])
    threshold = protocol.cfg_get("gates", "redock_rmsd_max")
    num_modes = protocol.cfg_get("docking", "num_modes", default=9)
    timeout = protocol.cfg_get("docking", "timeout_sec", default=300)
    exhaustiveness = protocol.cfg_get("docking", "exhaustiveness", default=8)

    try:
        confirmed = rr.load_confirmed(gene)
        pdb_id = confirmed["pdb_id"]
        ligand_chembl_id = confirmed.get("ligand_chembl_id")
        print(f"[test05] {gene}: {pdb_id}, лиганд {confirmed.get('ligand_name')} ({ligand_chembl_id})")

        smiles = protocol.fetch_ligand_smiles(ligand_chembl_id, confirmed.get("ligand_name"))

        struct_dir = os.path.join(protocol.BASE_DIR, "structures")
        original_pdb = os.path.join(struct_dir, f"{pdb_id}.pdb")
        receptor_pdbqt = os.path.join(struct_dir, f"{pdb_id}_receptor.pdbqt")
        if not os.path.exists(original_pdb) or not os.path.exists(receptor_pdbqt):
            raise RuntimeError(f"нет {original_pdb} или {receptor_pdbqt}")

        ligand_info = rr.find_ligand_center(original_pdb)
        resname = ligand_info["resname"]
        box_center, box_size = ligand_info["center"], ligand_info["box_size"]

        native_pdb_block = rr.extract_native_ligand_pdb_block(original_pdb, resname)
        native_mol = rr.mol_with_correct_bonds_from_pdb_block(native_pdb_block, smiles)
        print(f"[test05] нативная поза: {native_mol.GetNumAtoms()} атомов (после removeHs)")

        workdir = os.path.join(protocol.BASE_DIR, "runs", f"redock_{gene}")
        os.makedirs(workdir, exist_ok=True)

        print("[test05] готовлю свежий 3D-конформер из SMILES (координаты кристалла НЕ используются)...")
        ligand_pdbqt = os.path.join(workdir, "fresh_ligand.pdbqt")
        if not rr.prepare_ligand_pdbqt(smiles, ligand_pdbqt):
            raise RuntimeError("не удалось подготовить свежий лиганд")

        print(f"[test05] докую (exhaustiveness={exhaustiveness}, num_modes={num_modes} - "
              f"дефолт самой vina.exe, run_vina() его явно не передаёт)...")
        out_pdbqt = os.path.join(workdir, "docked_out_multi.pdbqt")
        score = rr.run_vina(receptor_pdbqt, ligand_pdbqt, box_center, box_size, out_pdbqt,
                             exhaustiveness=exhaustiveness, timeout=timeout)
        if score is None:
            raise RuntimeError("докинг не удался")

        docked_multi = all_docked_conformers(out_pdbqt, smiles)
        n_poses = docked_multi.GetNumConformers()
        print(f"[test05] получено поз: {n_poses}")

        rmsds = symmetry_rmsd_per_pose(native_mol, docked_multi)
    except Exception as e:
        return _save_error(gene, e)

    rmsd_top1 = rmsds[0]
    rmsd_top3 = min(rmsds[:3]) if len(rmsds) >= 1 else None
    rmsd_best_of_all = min(rmsds)
    passed = rmsd_top1 < threshold

    result = {
        "gene": gene, "pdb_id": pdb_id, "ligand_chembl_id": ligand_chembl_id,
        "status": "pass" if passed else "fail",
        "docking_score_kcal_mol": score, "n_poses": n_poses,
        "rmsd_per_pose": rmsds,
        "rmsd_top1": rmsd_top1, "rmsd_top3": rmsd_top3, "rmsd_best_of_all": rmsd_best_of_all,
        "threshold": threshold, "passed_gate": bool(passed),
        "rmsd_method": "spyrmsd symmetry=True (graph isomorphism)",
    }
    protocol.save_stage(gene, 1, {"test05_redock": result})

    print(f"\n=== Test 5 ({gene}): REAL redock (ГЕЙТ) ===")
    print(f"  RMSD top-1: {rmsd_top1:.3f}A  top-3 best: {rmsd_top3:.3f}A  best-of-{n_poses}: {rmsd_best_of_all:.3f}A")
    print(f"  Порог гейта: {threshold}A")
    if passed:
        print(f"  ПРОШЁЛ ГЕЙТ - обработку {gene} продолжаем")
    else:
        print(f"  НЕ ПРОШЁЛ ГЕЙТ (RMSD {rmsd_top1:.3f}A >= {threshold}A) - status=fail, "
              f"данные остаются, это единственная законная причина остановить {gene}")
    return result


def main():
    gene = sys.argv[1] if len(sys.argv) > 1 else "PIK3CA"
    force = "--force" in sys.argv
    run(gene, force)


if __name__ == "__main__":
    main()
