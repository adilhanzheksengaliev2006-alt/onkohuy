"""
run_redock.py — настоящий redock RMSD контроль (замена прежнего
"разумный диапазон скора" псевдо-контроля).

Идея: берём со-кристаллизованный лиганд из PDB-структуры (его РЕАЛЬНУЮ
позу, определённую рентгеном), готовим ту же молекулу ЗАНОВО из чистого
SMILES (свежий 3D-конформер, координаты кристалла игнорируются), докуем
её в тот же бокс тем же Vina, и сравниваем позу докинга с реальной позой
через RMSD (с учётом симметрии - AllChem.GetBestRMS). RMSD < 2.0A -
стандартный порог "докинг воспроизводит эксперимент" в литературе;
мишени, не проходящие порог, по протоколу проекта должны выбрасываться
из дальнейшей работы, а не использоваться с оговорками.

Использование:
    python run_redock.py [GENE]
    по умолчанию: PIK3CA
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gene_target_utils import get_chembl_new_client, find_ligand_center  # noqa: E402
from dock_existing_candidates import prepare_ligand_pdbqt, run_vina  # noqa: E402

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIRMED_PATH = os.path.join(BASE_DIR, "confirmed_structures.json")
RMSD_PASS_THRESHOLD = 2.0


def load_confirmed(gene):
    with open(CONFIRMED_PATH, encoding="utf-8") as f:
        data = json.load(f)
    if gene not in data:
        print(f"ОШИБКА: {gene} нет в {CONFIRMED_PATH}")
        sys.exit(1)
    return data[gene]


def extract_native_ligand_pdb_block(pdb_path, resname):
    """Вытаскивает координаты со-кристаллизованного лиганда (ТОЛЬКО его,
    без белка/воды) из оригинального (нестрипнутого) PDB - как отдельный
    PDB-блок, который потом RDKit сможет прочитать."""
    import gemmi
    st = gemmi.read_structure(pdb_path)
    st.setup_entities()
    lines = []
    atom_serial = 1
    for model in st:
        for chain in model:
            for residue in chain:
                if residue.name != resname:
                    continue
                for atom in residue:
                    lines.append(
                        f"HETATM{atom_serial:>5} {atom.name:<4} {resname:<3} A{residue.seqid.num:>4}    "
                        f"{atom.pos.x:8.3f}{atom.pos.y:8.3f}{atom.pos.z:8.3f}  1.00  0.00"
                        f"          {atom.element.name:>2}"
                    )
                    atom_serial += 1
        break  # только первая модель
    if not lines:
        raise RuntimeError(f"Лиганд {resname} не найден в {pdb_path}")
    lines.append("END")
    return "\n".join(lines)


def _strip_explicit_hydrogens(raw_mol):
    from rdkit import Chem
    rw = Chem.RWMol(raw_mol)
    h_indices = [a.GetIdx() for a in rw.GetAtoms() if a.GetAtomicNum() == 1]
    for idx in sorted(h_indices, reverse=True):
        rw.RemoveAtom(idx)
    mol = rw.GetMol()
    mol.UpdatePropertyCache(strict=False)
    from rdkit.Chem import rdmolops
    rdmolops.FastFindRings(mol)
    return mol


def _mol_with_template_coords(raw_mol, template_smiles):
    """Вместо починки порядков связей у raw_mol (AssignBondOrdersFromTemplate
    на этой связке RDKit-версий давал молекулы с не восстановленной
    неявной валентностью - docked-поза выходила как C19H4 вместо C19H22,
    буквально радикалы, отследили через несколько раундов отладки) —
    берём ЭТАЛОННУЮ молекулу из SMILES (она гарантированно имеет верную
    валентность/ароматичность, т.к. только что распарсена RDKit) и просто
    ПЕРЕНОСИМ на неё 3D-координаты из raw_mol через MCS-соответствие
    атомов (bondCompare=CompareAny - совпадение только по связности/
    элементам, не по порядку связи, т.к. у raw_mol порядок связи может
    быть неверно угадан obabel'ем)."""
    from rdkit import Chem
    from rdkit.Chem import rdFMCS

    template = Chem.MolFromSmiles(template_smiles)
    if template is None:
        raise RuntimeError(f"RDKit не смог распарсить эталонный SMILES: {template_smiles}")

    mcs = rdFMCS.FindMCS(
        [template, raw_mol],
        bondCompare=rdFMCS.BondCompare.CompareAny,
        atomCompare=rdFMCS.AtomCompare.CompareElements,
        ringMatchesRingOnly=False, completeRingsOnly=False, timeout=30,
    )
    if mcs.numAtoms < template.GetNumAtoms():
        raise RuntimeError(
            f"MCS ({mcs.numAtoms} атомов) меньше эталона ({template.GetNumAtoms()} атомов) - "
            f"поза докинга/кристалла не покрывает всю молекулу целиком (альт-локации? "
            f"неполная электронная плотность?)"
        )
    patt = Chem.MolFromSmarts(mcs.smartsString)
    template_match = template.GetSubstructMatch(patt)
    raw_match = raw_mol.GetSubstructMatch(patt)
    if not template_match or not raw_match:
        raise RuntimeError("MCS нашёлся, но GetSubstructMatch не смог сопоставить атомы обратно")

    out_mol = Chem.Mol(template)
    conf = Chem.Conformer(out_mol.GetNumAtoms())
    raw_conf = raw_mol.GetConformer()
    # атомы шаблона, НЕ попавшие в MCS (не должно случаться при
    # numAtoms >= template.GetNumAtoms() выше, но на всякий случай
    # даём координату первого сматченного атома, а не (0,0,0)) -
    # реальный сигнал проблемы уже отловлен исключением выше.
    fallback_pos = raw_conf.GetAtomPosition(raw_match[0])
    template_to_raw = dict(zip(template_match, raw_match))
    for i in range(out_mol.GetNumAtoms()):
        raw_idx = template_to_raw.get(i)
        pos = raw_conf.GetAtomPosition(raw_idx) if raw_idx is not None else fallback_pos
        conf.SetAtomPosition(i, pos)
    out_mol.RemoveAllConformers()
    out_mol.AddConformer(conf, assignId=True)
    return out_mol


def mol_with_correct_bonds_from_pdb_block(pdb_block, template_smiles):
    from rdkit import Chem
    raw_mol = Chem.MolFromPDBBlock(pdb_block, sanitize=False, removeHs=False)
    if raw_mol is None:
        raise RuntimeError("RDKit не смог прочитать PDB-блок нативного лиганда")
    raw_mol = _strip_explicit_hydrogens(raw_mol)
    return _mol_with_template_coords(raw_mol, template_smiles)


def mol_from_docked_pdbqt(pdbqt_path, template_smiles, workdir):
    """Берёт ЛУЧШУЮ (первую) позу из выходного PDBQT Vina.

    РАНЬШЕ конвертировал через obabel (PDBQT->SDF), как в
    molecule_builder.py - но на ESR1 (6SBO, лиганд amcenestrant) это
    молча теряло 2 реальных тяжёлых атома на длинной вложенной
    BRANCH-цепочке (гибкий фторпропильный хвост) и подставляло вместо
    них dummy-атомы ("*", atomic num 0) - obabel's PDBQT-парсер не
    справляется с глубокой вложенностью BRANCH/ENDBRANCH. Обнаружено
    через MCS-проверку (36 атомов вместо 38 в шаблоне).

    ИСПРАВЛЕНО: meeko (тот же инструмент, что готовил лиганд) сама
    встраивает в PDBQT-вывод Vina строки REMARK SMILES / REMARK SMILES IDX
    с точным отображением атомов на исходный SMILES - используем meeko
    для восстановления молекулы вместо лосси-конвертации через obabel.
    Тест: канонический SMILES восстановленной молекулы совпал с
    эталонным ПОБУКВЕННО, 38/38 атомов после RemoveHs."""
    from meeko import PDBQTMolecule, RDKitMolCreate
    from rdkit import Chem

    pdbqt_mol = PDBQTMolecule.from_file(pdbqt_path, skip_typing=True)
    mols = RDKitMolCreate.from_pdbqt_mol(pdbqt_mol)
    if not mols:
        raise RuntimeError(f"meeko не смогла восстановить молекулу из {pdbqt_path}")
    mol = Chem.RemoveHs(mols[0])
    if mol.GetNumConformers() == 0:
        raise RuntimeError(f"meeko вернула молекулу без конформеров из {pdbqt_path}")
    # conformer 0 = первая (лучшая по скору) поза - Vina всегда пишет
    # позы в порядке убывания качества.
    if mol.GetNumConformers() > 1:
        for conf_id in range(mol.GetNumConformers() - 1, 0, -1):
            mol.RemoveConformer(conf_id)
    return mol


def compute_rmsd(native_mol, docked_mol):
    from rdkit.Chem import AllChem
    # copy, т.к. GetBestRMS двигает пробную молекулу при выравнивании
    return AllChem.GetBestRMS(docked_mol, native_mol)


def run_redock(gene):
    confirmed = load_confirmed(gene)
    pdb_id = confirmed["pdb_id"]
    ligand_chembl_id = confirmed["ligand_chembl_id"]
    print(f"[redock] {gene}: {pdb_id}, лиганд {confirmed.get('ligand_name')} ({ligand_chembl_id})")

    nc = get_chembl_new_client()
    rec = nc.molecule.get(ligand_chembl_id)
    smiles = rec.get("molecule_structures", {}).get("canonical_smiles")
    if not smiles:
        print("ОШИБКА: не удалось получить SMILES лиганда из ChEMBL"); sys.exit(1)
    print(f"[redock] эталонный SMILES (ChEMBL): {smiles}")

    struct_dir = os.path.join(BASE_DIR, "structures")
    original_pdb = os.path.join(struct_dir, f"{pdb_id}.pdb")
    receptor_pdbqt = os.path.join(struct_dir, f"{pdb_id}_receptor.pdbqt")
    if not os.path.exists(original_pdb) or not os.path.exists(receptor_pdbqt):
        print(f"ОШИБКА: нет {original_pdb} или {receptor_pdbqt}"); sys.exit(1)

    ligand_info = find_ligand_center(original_pdb)
    resname = ligand_info["resname"]
    box_center, box_size = ligand_info["center"], ligand_info["box_size"]
    print(f"[redock] нативный лиганд: resname={resname}, box_center={box_center}")

    print("[redock] извлекаю нативную позу...")
    native_pdb_block = extract_native_ligand_pdb_block(original_pdb, resname)
    native_mol = mol_with_correct_bonds_from_pdb_block(native_pdb_block, smiles)
    print(f"[redock] нативная поза: {native_mol.GetNumAtoms()} атомов (после removeHs)")

    workdir = os.path.join(BASE_DIR, "runs", f"redock_{gene}")
    os.makedirs(workdir, exist_ok=True)

    print("[redock] готовлю СВЕЖИЙ 3D-конформер из SMILES (координаты кристалла НЕ используются)...")
    ligand_pdbqt = os.path.join(workdir, "fresh_ligand.pdbqt")
    ok = prepare_ligand_pdbqt(smiles, ligand_pdbqt)
    if not ok:
        print("ОШИБКА: не удалось подготовить свежий лиганд из SMILES"); sys.exit(1)

    print("[redock] докую (exhaustiveness=8, тот же бокс)...")
    out_pdbqt = os.path.join(workdir, "docked_out.pdbqt")
    score = run_vina(receptor_pdbqt, ligand_pdbqt, box_center, box_size, out_pdbqt,
                      exhaustiveness=8, timeout=300)
    if score is None:
        print("ОШИБКА: докинг не удался"); sys.exit(1)
    print(f"[redock] докинг-скор лучшей позы: {score:.2f} ккал/моль")

    print("[redock] сравниваю позу докинга с нативной...")
    docked_mol = mol_from_docked_pdbqt(out_pdbqt, smiles, workdir)
    print(f"[redock] поза докинга: {docked_mol.GetNumAtoms()} атомов (после removeHs)")

    rmsd = compute_rmsd(native_mol, docked_mol)
    passed = rmsd < RMSD_PASS_THRESHOLD

    report = {
        "gene": gene, "pdb_id": pdb_id, "ligand_chembl_id": ligand_chembl_id,
        "ligand_smiles": smiles, "docking_score_kcal_mol": score,
        "rmsd_angstrom": rmsd, "threshold": RMSD_PASS_THRESHOLD, "passed": passed,
    }
    out_path = os.path.join(workdir, "redock_report.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\n=== ИТОГ REDOCK ({gene}, {pdb_id}) ===")
    print(f"  RMSD к нативной позе: {rmsd:.3f} A (порог {RMSD_PASS_THRESHOLD} A)")
    print(f"  {'ПРОШЁЛ' if passed else 'НЕ ПРОШЁЛ'} redock-контроль")
    if not passed:
        print(f"  По протоколу: структура с RMSD >= {RMSD_PASS_THRESHOLD}A должна быть ИСКЛЮЧЕНА "
              f"из дальнейшей работы (докинг не воспроизводит известную экспериментальную позу)")
    print(f"\nСохранено: {out_path}")
    return report


def main():
    gene = sys.argv[1] if len(sys.argv) > 1 else "PIK3CA"
    run_redock(gene)


if __name__ == "__main__":
    main()
