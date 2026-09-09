"""
Test 15 (Stage 4) — PoseBusters (Buttenschoen et al., Chem Sci 2024):
проверка "физической разумности" сохранённых поз докинга (валентности,
внутримолекулярные клэши, столкновения с белком, разумная геометрия
кольца и т.д.).

ЧЕСТНО про ограничение данных: позы (полные PDBQT с координатами, не
только скор) СОХРАНЕНЫ только для 11 redock-мишеней
(runs/redock_<GENE>/docked_out_multi.pdbqt, если запускался test05) -
для основного Test A/funnel пайплайна (results.jsonl/funnel_results.jsonl)
сохраняется только скор, не координаты позы, так что PoseBusters по НИМ
запустить нельзя без передокинга. Если docked_out_multi.pdbqt для GENE
нет - явно печатаем это и не подставляем никаких выдуманных чисел.

Использование:
    python controls/test15_posebusters.py GENE [--force]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import protocol  # noqa: E402
import run_redock as rr  # noqa: E402


def run(gene, force=False):
    if protocol.test_already_done(gene, 4, "test15_posebusters", force):
        print(f"[test15] {gene}: уже посчитано, пропускаю. --force для пересчёта")
        return protocol.load_stage(gene, 4)["test15_posebusters"]

    protocol.print_banner("test15")

    pdbqt_path = os.path.join(protocol.BASE_DIR, "runs", f"redock_{gene}", "docked_out_multi.pdbqt")
    if not os.path.exists(pdbqt_path):
        result = {
            "error": f"НЕТ СОХРАНЁННЫХ ПОЗ для {gene}: {pdbqt_path} не существует. "
                     f"Сначала запусти controls/test05_redock.py {gene} - основной Test A/funnel "
                     f"пайплайн координаты поз вообще не сохраняет, только скор.",
        }
        protocol.save_stage(gene, 4, {"test15_posebusters": result})
        print(f"[test15] {result['error']}")
        return result

    confirmed = rr.load_confirmed(gene)
    pdb_id = confirmed["pdb_id"]
    receptor_pdb = os.path.join(protocol.BASE_DIR, "structures", f"{pdb_id}_clean.pdb")
    if not os.path.exists(receptor_pdb):
        receptor_pdb = os.path.join(protocol.BASE_DIR, "structures", f"{pdb_id}.pdb")

    smiles = protocol.fetch_ligand_smiles(confirmed.get("ligand_chembl_id"), confirmed.get("ligand_name"))

    from meeko import PDBQTMolecule, RDKitMolCreate
    from rdkit import Chem
    from posebusters import PoseBusters

    pdbqt_mol = PDBQTMolecule.from_file(pdbqt_path, skip_typing=True)
    mols = RDKitMolCreate.from_pdbqt_mol(pdbqt_mol)
    mol_multi = Chem.RemoveHs(mols[0])
    n_poses = mol_multi.GetNumConformers()
    print(f"[test15] {gene}: {n_poses} поз из {pdbqt_path}, белок={receptor_pdb}")

    pb = PoseBusters(config="redock")
    per_pose_results = []
    for conf_id in range(n_poses):
        pose_mol = Chem.Mol(mol_multi, confId=conf_id)
        try:
            df = pb.bust([pose_mol], mol_cond=receptor_pdb, full_report=False)
            row = df.iloc[0].to_dict()
        except Exception as e:
            row = {"error": str(e)}
        per_pose_results.append({"pose_idx": conf_id, **{str(k): (bool(v) if isinstance(v, (bool,)) else v) for k, v in row.items()}})

    # доля поз, прошедших ВСЕ проверки (все bool-колонки True). Берём
    # ОБЪЕДИНЕНИЕ ключей по ВСЕМ позам, а не только по первой - если первая
    # поза упала с ошибкой (пустой/усечённый набор колонок), а остальные
    # успешно прошли busting, проверки остальных поз не должны тихо теряться.
    bool_cols = sorted({k for r in per_pose_results for k, v in r.items()
                         if k not in ("pose_idx", "error") and isinstance(v, bool)})
    n_all_pass = sum(1 for r in per_pose_results if all(r.get(c, False) for c in bool_cols))
    violation_counts = {}
    for c in bool_cols:
        n_fail = sum(1 for r in per_pose_results if r.get(c) is False)
        if n_fail:
            violation_counts[c] = n_fail

    result = {
        "n_poses": n_poses, "n_all_checks_pass": n_all_pass,
        "pass_rate": n_all_pass / n_poses if n_poses else None,
        "checks": bool_cols, "violation_counts_by_check": violation_counts,
        "per_pose": per_pose_results,
    }
    protocol.save_stage(gene, 4, {"test15_posebusters": result})

    print(f"\n=== Test 15 ({gene}): PoseBusters ===")
    print(f"  прошли ВСЕ проверки: {n_all_pass}/{n_poses}")
    if violation_counts:
        print(f"  нарушения по типу:")
        for check, n_fail in sorted(violation_counts.items(), key=lambda kv: -kv[1]):
            print(f"    {check}: {n_fail}/{n_poses} поз провалили")
    return result


def main():
    gene = sys.argv[1] if len(sys.argv) > 1 else "PIK3CA"
    force = "--force" in sys.argv
    run(gene, force)


if __name__ == "__main__":
    main()
