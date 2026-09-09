import os
import subprocess
import sys
import warnings
import tempfile

import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem.rdForceFieldHelpers import UFFOptimizeMolecule, UFFHasAllMoleculeParams

import utils
from constants import bonds1, bonds2, bonds3, margin1, margin2, margin3, \
    bond_dict

# ПРАВКА ПОД WINDOWS: см. подробный разбор в make_mol_openbabel — Python-
# биндинги openbabel (OBConversion.WriteFile/WriteString) стабильно пишут
# ПУСТОЙ результат в любой "химический" формат (sdf/mol/mol2/pdb) на этой
# машине, хотя чтение и распознавание связей работают. Отдельный обычный
# obabel.exe (не Python-биндинг) конвертирует ТЕ ЖЕ данные без проблем —
# подтверждено вручную. Похоже, Python-расширение ищет data-файлы
# (Library\share\openbabel\<версия>\) относительно python.exe, а не
# относительно своей DLL, и не находит их там, где их видит сам obabel.exe
# (Library\bin\). Поэтому XYZ->SDF конвертация идёт через сам бинарник
# subprocess'ом — тот же паттерн, что уже используется в проекте для
# vina.exe/mk_prepare_ligand.exe вместо капризных Python-биндингов.
OBABEL_EXE = os.path.join(os.path.dirname(os.path.abspath(sys.executable)), "Library", "bin", "obabel.exe")


def get_bond_order(atom1, atom2, distance):
    distance = 100 * distance  # We change the metric

    if atom1 in bonds3 and atom2 in bonds3[atom1] and distance < bonds3[atom1][atom2] + margin3:
        return 3  # Triple

    if atom1 in bonds2 and atom2 in bonds2[atom1] and distance < bonds2[atom1][atom2] + margin2:
        return 2  # Double

    if atom1 in bonds1 and atom2 in bonds1[atom1] and distance < bonds1[atom1][atom2] + margin1:
        return 1  # Single

    return 0      # No bond


def get_bond_order_batch(atoms1, atoms2, distances, dataset_info):
    if isinstance(atoms1, np.ndarray):
        atoms1 = torch.from_numpy(atoms1)
    if isinstance(atoms2, np.ndarray):
        atoms2 = torch.from_numpy(atoms2)
    if isinstance(distances, np.ndarray):
        distances = torch.from_numpy(distances)

    distances = 100 * distances  # We change the metric

    bonds1 = torch.tensor(dataset_info['bonds1'], device=atoms1.device)
    bonds2 = torch.tensor(dataset_info['bonds2'], device=atoms1.device)
    bonds3 = torch.tensor(dataset_info['bonds3'], device=atoms1.device)

    bond_types = torch.zeros_like(atoms1)  # 0: No bond

    # Single
    bond_types[distances < bonds1[atoms1, atoms2] + margin1] = 1

    # Double (note that already assigned single bonds will be overwritten)
    bond_types[distances < bonds2[atoms1, atoms2] + margin2] = 2

    # Triple
    bond_types[distances < bonds3[atoms1, atoms2] + margin3] = 3

    return bond_types


def make_mol_openbabel(positions, atom_types, atom_decoder):
    """
    Build an RDKit molecule using openbabel for creating bonds
    Args:
        positions: N x 3
        atom_types: N
        atom_decoder: maps indices to atom types
    Returns:
        rdkit molecule
    """
    atom_types = [atom_decoder[x] for x in atom_types]

    # ПРАВКИ ПОД WINDOWS (оригинал DiffSBDD рассчитан только на Linux, здесь
    # ловились два РАЗНЫХ бага, оба воспроизводились стабильно 100% времени
    # на реальных сгенерированных молекулах, не на синтетических тестах):
    #
    # 1. `with tempfile.NamedTemporaryFile() as tmp:` держит файл открытым
    #    эксклюзивно на всё тело блока — на POSIX неважно (несколько open()
    #    на один inode разрешены), на Windows open(tmp_file, 'w') внутри
    #    write_xyz_file падал с PermissionError, т.к. хендл ещё занят.
    #
    # 2. Даже после того как п.1 починили (закрыли хендл сразу), ОДИН И ТОТ
    #    ЖЕ путь использовался и для XYZ-входа, и для SDF-выхода
    #    (obConversion.ReadFile(tmp_file) затем WriteFile(tmp_file)).
    #    OpenBabel на Windows не освобождает файловый хендл после ReadFile
    #    (подтверждено отдельно: os.remove() сразу после ReadFile падал с
    #    "файл занят другим процессом") — поэтому WriteFile молча возвращал
    #    False и НЕ перезаписывал файл. Chem.SDMolSupplier затем пытался
    #    прочитать всё ещё XYZ-содержимое как SDF и падал с "Cannot convert
    #    'X -' to unsigned int" (парсер SDF спотыкался о числовые поля
    #    XYZ-строк) — ЭТО и есть настоящая причина ошибки, которая на
    #    первый взгляд выглядела как проблема самого XYZ-файла, а
    #    оказалась о повторном использовании одного пути для read+write.
    #    Фикс: два раздельных временных файла.
    tmp_in = tempfile.NamedTemporaryFile(delete=False, suffix=".xyz")
    tmp_in_file = tmp_in.name
    tmp_in.close()
    # tmp_out_file: путь БЕЗ пред-создания файла (mktemp, не
    # NamedTemporaryFile) — подозрение, что OpenBabel.WriteFile молча не
    # может открыть на запись УЖЕ существующий (пусть пустой) файл на
    # этой машине; пусть создаст сам.
    tmp_out_file = tempfile.mktemp(suffix=".sdf")
    try:
        # Write xyz file
        utils.write_xyz_file(positions, atom_types, tmp_in_file)

        # Convert to sdf file via obabel.exe CLI (не Python-биндинг — см.
        # комментарий у OBABEL_EXE вверху файла). obabel сам перцепирует
        # связи из 3D-координат (ConnectTheDots/PerceiveBondOrders), как
        # раньше делал OBConversion.ReadFile/WriteFile.
        result = subprocess.run(
            [OBABEL_EXE, "-ixyz", tmp_in_file, "-osdf", "-O", tmp_out_file],
            capture_output=True, text=True, timeout=60,
        )

        # Openbabel не всегда может достроить связи из точечного облака
        # (вырожденная геометрия конкретной сгенерированной молекулы —
        # ожидаемое, не редкое поведение при стохастической генерации) —
        # тогда конвертация не создаёт файл вовсе или создаёт пустой.
        # Chem.SDMolSupplier на несуществующий/пустой файл бросает OSError,
        # а не отдаёт [None] — нужна явная проверка, иначе одна неудачная
        # молекула роняет всю генерацию сида, а не просто пропускается
        # (тот же контракт, что и выше для tmp_mol is None).
        if result.returncode != 0 or not os.path.exists(tmp_out_file) or os.path.getsize(tmp_out_file) == 0:
            return None

        # Read sdf file with RDKit
        tmp_mol = Chem.SDMolSupplier(tmp_out_file, sanitize=False)[0]
    finally:
        for f in (tmp_in_file, tmp_out_file):
            try:
                os.remove(f)
            except OSError:
                pass

    # РЕАЛЬНЫЙ БАГ vendor-кода (не Windows-специфичный): openbabel иногда не
    # может собрать валидную структуру из точечного облака (вырожденная
    # геометрия конкретной сгенерированной молекулы) и пишет пустой/битый
    # SDF — тогда SDMolSupplier[0] возвращает None. Вызывающий код
    # (lightning_modules.py generate_ligands, строка с
    # `if mol is not None: molecules.append(mol)`) явно рассчитан на то,
    # что build_molecule() может вернуть None для одной неудачной молекулы
    # и она просто пропускается — но здесь до сих пор не было проверки,
    # и .GetAtoms() на None ронял AttributeError весь процесс генерации
    # (а не только эту одну молекулу). Без этой проверки одна "неудачная"
    # молекула в батче убивала бы всю генерацию сида.
    if tmp_mol is None:
        return None

    # Build new molecule. This is a workaround to remove radicals.
    mol = Chem.RWMol()
    for atom in tmp_mol.GetAtoms():
        mol.AddAtom(Chem.Atom(atom.GetSymbol()))
    mol.AddConformer(tmp_mol.GetConformer(0))

    for bond in tmp_mol.GetBonds():
        mol.AddBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(),
                    bond.GetBondType())

    return mol


def make_mol_edm(positions, atom_types, dataset_info, add_coords):
    """
    Equivalent to EDM's way of building RDKit molecules
    """
    n = len(positions)

    # (X, A, E): atom_types, adjacency matrix, edge_types
    # X: N (int)
    # A: N x N (bool) -> (binary adjacency matrix)
    # E: N x N (int) -> (bond type, 0 if no bond)
    pos = positions.unsqueeze(0)  # add batch dim
    dists = torch.cdist(pos, pos, p=2).squeeze(0).view(-1)  # remove batch dim & flatten
    atoms1, atoms2 = torch.cartesian_prod(atom_types, atom_types).T
    E_full = get_bond_order_batch(atoms1, atoms2, dists, dataset_info).view(n, n)
    E = torch.tril(E_full, diagonal=-1)  # Warning: the graph should be DIRECTED
    A = E.bool()
    X = atom_types

    mol = Chem.RWMol()
    for atom in X:
        a = Chem.Atom(dataset_info["atom_decoder"][atom.item()])
        mol.AddAtom(a)

    all_bonds = torch.nonzero(A)
    for bond in all_bonds:
        mol.AddBond(bond[0].item(), bond[1].item(),
                    bond_dict[E[bond[0], bond[1]].item()])

    if add_coords:
        conf = Chem.Conformer(mol.GetNumAtoms())
        for i in range(mol.GetNumAtoms()):
            conf.SetAtomPosition(i, (positions[i, 0].item(),
                                     positions[i, 1].item(),
                                     positions[i, 2].item()))
        mol.AddConformer(conf)

    return mol


def build_molecule(positions, atom_types, dataset_info, add_coords=False,
                   use_openbabel=True):
    """
    Build RDKit molecule
    Args:
        positions: N x 3
        atom_types: N
        dataset_info: dict
        add_coords: Add conformer to mol (always added if use_openbabel=True)
        use_openbabel: use OpenBabel to create bonds
    Returns:
        RDKit molecule
    """
    if use_openbabel:
        mol = make_mol_openbabel(positions, atom_types,
                                 dataset_info["atom_decoder"])
    else:
        mol = make_mol_edm(positions, atom_types, dataset_info, add_coords)

    return mol


def process_molecule(rdmol, add_hydrogens=False, sanitize=False, relax_iter=0,
                     largest_frag=False):
    """
    Apply filters to an RDKit molecule. Makes a copy first.
    Args:
        rdmol: rdkit molecule
        add_hydrogens
        sanitize
        relax_iter: maximum number of UFF optimization iterations
        largest_frag: filter out the largest fragment in a set of disjoint
            molecules
    Returns:
        RDKit molecule or None if it does not pass the filters
    """

    # build_molecule() может вернуть None (см. правку в make_mol_openbabel
    # выше) — Chem.Mol(None) падает с TypeError вместо того, чтобы
    # соблюсти контракт "None -> не прошла фильтр", который используется
    # везде в этой функции (см. return None ниже). Без этой проверки
    # починка в make_mol_openbabel просто сдвинула бы тот же краш на
    # одну строчку дальше.
    if rdmol is None:
        return None

    # Create a copy
    mol = Chem.Mol(rdmol)

    if sanitize:
        try:
            Chem.SanitizeMol(mol)
        except ValueError:
            warnings.warn('Sanitization failed. Returning None.')
            return None

    if add_hydrogens:
        mol = Chem.AddHs(mol, addCoords=(len(mol.GetConformers()) > 0))

    if largest_frag:
        mol_frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
        mol = max(mol_frags, default=mol, key=lambda m: m.GetNumAtoms())
        if sanitize:
            # sanitize the updated molecule
            try:
                Chem.SanitizeMol(mol)
            except ValueError:
                return None

    if relax_iter > 0:
        if not UFFHasAllMoleculeParams(mol):
            warnings.warn('UFF parameters not available for all atoms. '
                          'Returning None.')
            return None

        try:
            uff_relax(mol, relax_iter)
            if sanitize:
                # sanitize the updated molecule
                Chem.SanitizeMol(mol)
        except (RuntimeError, ValueError) as e:
            return None

    return mol


def uff_relax(mol, max_iter=200):
    """
    Uses RDKit's universal force field (UFF) implementation to optimize a
    molecule.
    """
    more_iterations_required = UFFOptimizeMolecule(mol, maxIters=max_iter)
    if more_iterations_required:
        warnings.warn(f'Maximum number of FF iterations reached. '
                      f'Returning molecule after {max_iter} relaxation steps.')
    return more_iterations_required


def filter_rd_mol(rdmol):
    """
    Filter out RDMols if they have a 3-3 ring intersection
    adapted from:
    https://github.com/luost26/3D-Generative-SBDD/blob/main/utils/chem.py
    """
    ring_info = rdmol.GetRingInfo()
    ring_info.AtomRings()
    rings = [set(r) for r in ring_info.AtomRings()]

    # 3-3 ring intersection
    for i, ring_a in enumerate(rings):
        if len(ring_a) != 3:
            continue
        for j, ring_b in enumerate(rings):
            if i <= j:
                continue
            inter = ring_a.intersection(ring_b)
            if (len(ring_b) == 3) and (len(inter) > 0): 
                return False

    return True
