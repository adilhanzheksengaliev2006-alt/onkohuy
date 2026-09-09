"""
Test 6 (Stage 5, ДОРОГОЙ, ТОЛЬКО ПО ЯВНОЙ КОМАНДЕ) — cross-docking:
берём лиганды из ДРУГИХ кристаллических структур того же таргета,
докуем их в НАШУ (выбранную в confirmed_structures.json) структуру,
считаем RMSD к их "истинной" позе.

Т.к. разные PDB-записи имеют разные системы координат, "истинную" позу
альтернативного лиганда сначала нужно перенести в систему координат
нашего рецептора - делаем это суперпозицией белка (Bio.PDB Superimposer
по CA-атомам общих остатков), затем применяем ту же трансформацию к
координатам альтернативного лиганда.

НЕ запускается по умолчанию (test06_cross_docking.enabled_by_default:
false) - нужен --confirm и явный список ДОПОЛНИТЕЛЬНЫХ PDB ID с их
ligand_chembl_id/resname, которые пользователь должен предоставить сам
(в проекте сейчас нет готового списка альтернативных структур на мишень).

Использование:
    python controls/test06_cross_docking.py PIK3CA --confirm \\
        --alt-pdb 5UBT:resname_lig=XYZ:chembl=CHEMBL123456 [--force]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import protocol  # noqa: E402
import run_redock as rr  # noqa: E402
import test05_redock as t05  # noqa: E402


def fetch_alt_structure(pdb_id):
    import urllib.request
    dest = os.path.join(protocol.BASE_DIR, "structures", f"{pdb_id}.pdb")
    if not os.path.exists(dest):
        url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
        urllib.request.urlretrieve(url, dest)
    return dest


def _longest_chain_ca_sequence(struct):
    """Возвращает (chain_id, [(resid_tuple, one_letter_code, CA_atom), ...])
    для САМОЙ ДЛИННОЙ полипептидной цепи первой модели - эвристика "это и
    есть основной белок, а не пептид-лиганд/тег/вода". Chain-aware (в
    отличие от прежней версии, которая сравнивала номера остатков БЕЗ
    учёта цепи - в мультимерных структурах номера в разных цепях
    совпадают, что молча путало остатки одной цепи с другой)."""
    from Bio.PDB.Polypeptide import protein_letters_3to1

    best = None
    for model in struct:
        for chain in model:
            residues, seq = [], []
            for res in chain:
                if "CA" not in res or res.id[0] != " ":  # id[0]!=' ' - гетероатом/вода
                    continue
                aa = protein_letters_3to1.get(res.get_resname())
                if aa is None:
                    continue
                residues.append(res)
                seq.append(aa)
            if best is None or len(residues) > len(best[1]):
                best = (chain.id, residues, "".join(seq))
        break  # только первая модель
    if best is None or len(best[1]) < 20:
        raise RuntimeError("не нашлось достаточно длинной полипептидной цепи")
    return best


def superpose_and_transform_ligand(our_pdb_path, alt_pdb_path, alt_ligand_resname):
    """Суперпозиция alt_pdb на our_pdb по CA атомам, сопоставленным ЧЕРЕЗ
    ВЫРАВНИВАНИЕ ПОСЛЕДОВАТЕЛЬНОСТЕЙ (не по совпадению номера остатка -
    у разных PDB-депозиций той же цепи нумерация может отличаться из-за
    тегов/сдвигов/разных сконструированных границ, что раньше давало
    суперпозицию с RMSD белка ~26A - вообще не выравнивание). Затем
    применяем ту же трансформацию к координатам alt-лиганда. Возвращает
    атомы лиганда (Bio.PDB) с трансформированными координатами."""
    import numpy as np
    from Bio.Align import PairwiseAligner
    from Bio.PDB import PDBParser, Superimposer

    parser = PDBParser(QUIET=True)
    our_struct = parser.get_structure("our", our_pdb_path)
    alt_struct = parser.get_structure("alt", alt_pdb_path)

    our_chain_id, our_residues, our_seq = _longest_chain_ca_sequence(our_struct)
    alt_chain_id, alt_residues, alt_seq = _longest_chain_ca_sequence(alt_struct)
    print(f"[test06] our chain {our_chain_id} ({len(our_seq)} остатков), "
          f"alt chain {alt_chain_id} ({len(alt_seq)} остатков)")

    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.open_gap_score = -10
    aligner.extend_gap_score = -0.5
    aligner.substitution_matrix = None  # эквивалентно match/mismatch по умолчанию (+1/-1 через match_score/mismatch_score ниже)
    aligner.match_score = 2
    aligner.mismatch_score = -1
    alignment = aligner.align(our_seq, alt_seq)[0]

    fixed, moving = [], []
    n_identical = 0
    for (our_start, our_end), (alt_start, alt_end) in zip(*alignment.aligned):
        for offset in range(our_end - our_start):
            oi, ai = our_start + offset, alt_start + offset
            if our_seq[oi] != alt_seq[ai]:
                continue  # берём в суперпозицию только ИДЕНТИЧНЫЕ позиции - консервативное ядро
            n_identical += 1
            fixed.append(our_residues[oi]["CA"])
            moving.append(alt_residues[ai]["CA"])

    if len(fixed) < 20:
        raise RuntimeError(f"слишком мало идентичных выровненных остатков для суперпозиции ({len(fixed)})")

    sup = Superimposer()
    sup.set_atoms(fixed, moving)
    print(f"[test06] суперпозиция по {len(fixed)} идентичным выровненным CA "
          f"(из {n_identical} совпадений в выравнивании), RMSD белка={sup.rms:.3f}A")
    if sup.rms > 3.0:
        print(f"[test06] [warn] RMSD белка {sup.rms:.3f}A подозрительно высок для суперпозиции "
              f"одного и того же белка - структуры могут отличаться конформационно сильнее ожидаемого")

    lig_atoms = []
    for model in alt_struct:
        for chain in model:
            for res in chain:
                if res.get_resname().strip() == alt_ligand_resname:
                    lig_atoms.extend(res.get_atoms())
        break
    if not lig_atoms:
        raise RuntimeError(f"лиганд {alt_ligand_resname} не найден в {alt_pdb_path}")

    sup.apply(lig_atoms)  # применяет трансформацию in-place к координатам
    return lig_atoms


def lig_atoms_to_pdb_block(lig_atoms, resname):
    """ТОТ ЖЕ формат HETATM-строк, что run_redock.extract_native_ligand_pdb_block -
    чтобы дальше пройти через ПРОВЕРЕННУЮ (уже валидированную на 11 мишенях)
    rr.mol_with_correct_bonds_from_pdb_block(), которая делает перцепцию связей
    через MCS с эталонным SMILES, а не через голые координаты без графа."""
    lines = []
    for i, atom in enumerate(lig_atoms, start=1):
        x, y, z = atom.get_coord()
        element = (atom.element or "C").strip() or "C"
        lines.append(
            f"HETATM{i:>5} {atom.get_name():<4} {resname:<3} A{1:>4}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {element:>2}"
        )
    lines.append("END")
    return "\n".join(lines)


def run(gene, alt_specs, force=False):
    """alt_specs: список dict {'pdb_id':..., 'resname':..., 'chembl_id':...}"""
    if protocol.test_already_done(gene, 5, "test06_cross_docking", force):
        print(f"[test06] {gene}: уже посчитано, пропускаю. --force для пересчёта")
        return protocol.load_stage(gene, 5)["test06_cross_docking"]

    protocol.print_banner("test06", ["gates.redock_rmsd_max"])
    threshold = protocol.cfg_get("gates", "redock_rmsd_max")

    confirmed = rr.load_confirmed(gene)
    our_pdb_id = confirmed["pdb_id"]
    our_pdb_path = os.path.join(protocol.BASE_DIR, "structures", f"{our_pdb_id}.pdb")
    receptor_pdbqt = os.path.join(protocol.BASE_DIR, "structures", f"{our_pdb_id}_receptor.pdbqt")

    per_structure = []
    for spec in alt_specs:
        alt_pdb_id = spec["pdb_id"]
        print(f"[test06] {gene}: cross-dock {alt_pdb_id} -> {our_pdb_id}")
        try:
            alt_pdb_path = fetch_alt_structure(alt_pdb_id)
            lig_atoms = superpose_and_transform_ligand(our_pdb_path, alt_pdb_path, spec["resname"])
            true_coords_mean = sum(a.get_coord() for a in lig_atoms) / len(lig_atoms)

            smiles = protocol.fetch_ligand_smiles(spec.get("chembl_id"), spec.get("resname"))

            # ПРОВЕРЕННЫЙ путь построения референсной молекулы (перцепция
            # связей через MCS с эталонным SMILES) - тот же, что для родной
            # позы в run_redock.py, вместо голых координат без графа связей.
            pdb_block = lig_atoms_to_pdb_block(lig_atoms, spec["resname"])
            native_mol = rr.mol_with_correct_bonds_from_pdb_block(pdb_block, smiles)

            workdir = os.path.join(protocol.BASE_DIR, "runs", f"crossdock_{gene}_{alt_pdb_id}")
            os.makedirs(workdir, exist_ok=True)
            ligand_pdbqt = os.path.join(workdir, "fresh_ligand.pdbqt")
            if not rr.prepare_ligand_pdbqt(smiles, ligand_pdbqt):
                raise RuntimeError("не удалось подготовить лиганд")

            box_center = tuple(float(v) for v in true_coords_mean)
            box_size = (20, 20, 20)
            out_pdbqt = os.path.join(workdir, "docked_out_multi.pdbqt")
            score = rr.run_vina(receptor_pdbqt, ligand_pdbqt, box_center, box_size, out_pdbqt,
                                 exhaustiveness=8, timeout=300)
            if score is None:
                raise RuntimeError("докинг не удался")

            # то же извлечение всех поз + symmetry-corrected RMSD (spyrmsd,
            # graph isomorphism), что и в Test 5 - вместо позиционного
            # сопоставления координат без учёта атомного графа.
            docked_multi = t05.all_docked_conformers(out_pdbqt, smiles)
            rmsds = t05.symmetry_rmsd_per_pose(native_mol, docked_multi)

            per_structure.append({
                "alt_pdb_id": alt_pdb_id, "docking_score_kcal_mol": score,
                "n_poses": docked_multi.GetNumConformers(),
                "rmsd_top1": rmsds[0], "rmsd_best_of_all": min(rmsds),
                "passed": bool(rmsds[0] < threshold),
            })
        except Exception as e:
            per_structure.append({"alt_pdb_id": alt_pdb_id, "error": str(e)})

    result = {"gene": gene, "our_pdb_id": our_pdb_id, "per_structure": per_structure, "threshold": threshold}
    protocol.save_stage(gene, 5, {"test06_cross_docking": result})

    print(f"\n=== Test 6 ({gene}): cross-docking ===")
    for r in per_structure:
        print(f"  {r}")
    return result


def parse_alt_spec(s):
    parts = dict(kv.split("=") for kv in s.split(":") if "=" in kv)
    pdb_id = s.split(":")[0]
    return {"pdb_id": pdb_id, "resname": parts.get("resname_lig"), "chembl_id": parts.get("chembl")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("gene")
    parser.add_argument("--confirm", action="store_true")
    parser.add_argument("--alt-pdb", action="append", default=[],
                         help="формат PDBID:resname_lig=RES:chembl=CHEMBLxxxx, можно повторять")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if not args.confirm or not args.alt_pdb:
        print("[test06] Тест ДОРОГОЙ и требует --confirm И хотя бы один --alt-pdb от пользователя.")
        print("[test06] Ничего не делаю. Пример:")
        print("  python controls/test06_cross_docking.py PIK3CA --confirm --alt-pdb 5UBT:resname_lig=XYZ:chembl=CHEMBL123456")
        return

    alt_specs = [parse_alt_spec(s) for s in args.alt_pdb]
    run(args.gene, alt_specs, args.force)


if __name__ == "__main__":
    main()
