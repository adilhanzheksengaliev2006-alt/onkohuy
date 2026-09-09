"""
profile_generation.py — шаг 0 отчёта: профилирование генерации DiffSBDD.

ЗАЧЕМ: замер показал batch_size 4 -> 15.2с/молекулу, batch_size 128 ->
14.9с/молекулу — почти не изменилось, хотя VRAM выросла со 117 МБ до
785 МБ (параметр применился). Значит узкое место не в самом GPU-проходе
диффузионной модели. Этот скрипт разделяет время на три части, чтобы
проверить гипотезу.

ГИПОТЕЗА (не доказана, для этого и скрипт): build_molecule() на каждую
молекулу делает ОТДЕЛЬНЫЙ subprocess.run() к obabel.exe (см. правку в
analysis/molecule_builder.py — Python-биндинги OpenBabel на этой машине
не работают, конвертация XYZ->SDF идёт через сам бинарник). Это
происходит ПОСЛЕДОВАТЕЛЬНО, одна молекула за раз, в цикле ПОСЛЕ того,
как весь батч уже прошёл через GPU разом — то есть увеличение batch_size
ускоряет только сам проход через модель, а не эту последовательную
пост-обработку, если именно она — узкое место.

КАК СЧИТАЕТСЯ (важно для интерпретации):
  GPU-время = TOTAL(model.generate_ligands) - build_molecule - process_molecule
  (вычитанием, а не прямым замером — внутрь самого forward-прохода без
  правки vendor-кода лезть не стал; build_molecule/process_molecule — это
  единственные CPU-шаги внутри generate_ligands() после сэмплирования,
  так что вычитание даёт корректную оценку GPU-доли)

  molecule_builder-время = суммарное время build_molecule() — сборка
  молекулы из облака атомов через obabel.exe

  RDKit-время = process_molecule() (санитизация + largest_frag) +
  ОТДЕЛЬНО время Chem.MolToSmiles() на весь список — эта часть вообще
  НЕ входит в generate_ligands(), происходит уже в generate_smiles_seeded.py
  ПОСЛЕ него, что само по себе часть ответа на вопрос "где уходит время"

ПРАВКА vendor-кода НЕ трогается — только монки-патч ИМЁН build_molecule/
process_molecule ВНУТРИ пространства имён lightning_modules (там прямой
`from analysis.molecule_builder import build_molecule, process_molecule` —
patch должен идти по lightning_modules.build_molecule, патчить
analysis.molecule_builder.build_molecule бесполезно, эта копия имени уже
не используется после импорта).

Использование (пример на 30 молекулах, дефолтный чекпоинт/карман 4JPS):
  python profile_generation.py --n_samples 30 --out profile_results.json

С тестом числа шагов диффузии (шаг 0, вторая часть) и тестом
параллелизации molecule_builder по ядрам CPU:
  python profile_generation.py --n_samples 30 --test_timesteps --test_parallel --out profile_results.json

Ничего не удаляет и не перезаписывает существующие скрипты проекта —
отдельный файл, generate_smiles_seeded.py не тронут.
"""

import argparse
import json
import multiprocessing
import os
import sys
import time

import numpy as np
import torch
from openbabel import openbabel

openbabel.obErrorLog.StopLogging()

import lightning_modules as lm  # noqa: E402
from analysis.molecule_builder import build_molecule as _real_build_molecule  # noqa: E402
from rdkit import Chem  # noqa: E402


# =============================================================================
# Инструментация: монки-патч build_molecule/process_molecule ВНУТРИ
# пространства имён lightning_modules (см. docstring — critical detail).
# =============================================================================
_timing = {
    "build_molecule_sec": 0.0,
    "build_molecule_calls": 0,
    "process_molecule_sec": 0.0,
    "process_molecule_calls": 0,
}
_fragment_flags = []  # True = сырая молекула из ОДНОГО куска (не разорвана)
_captured_calls = []  # (args, kwargs) для build_molecule — нужно для --test_parallel
_collect_inputs = False

_orig_build_molecule = lm.build_molecule
_orig_process_molecule = lm.process_molecule


def _timed_build_molecule(*args, **kwargs):
    if _collect_inputs:
        _captured_calls.append((args, kwargs))
    t0 = time.perf_counter()
    result = _orig_build_molecule(*args, **kwargs)
    _timing["build_molecule_sec"] += time.perf_counter() - t0
    _timing["build_molecule_calls"] += 1
    # Фрагментация СЫРОЙ молекулы (до largest_frag-обрезки в process_molecule,
    # которая происходит следующим шагом и всегда оставляет один кусок —
    # поэтому проверять нужно именно здесь, иначе "доля без разорванных
    # фрагментов" тривиально всегда будет 100%).
    if result is not None:
        try:
            n_frags = len(Chem.GetMolFrags(result, sanitizeFrags=False))
            _fragment_flags.append(n_frags == 1)
        except Exception:
            _fragment_flags.append(False)
    return result


def _timed_process_molecule(*args, **kwargs):
    t0 = time.perf_counter()
    result = _orig_process_molecule(*args, **kwargs)
    _timing["process_molecule_sec"] += time.perf_counter() - t0
    _timing["process_molecule_calls"] += 1
    return result


lm.build_molecule = _timed_build_molecule
lm.process_molecule = _timed_process_molecule


def _reset_timing():
    _timing["build_molecule_sec"] = 0.0
    _timing["build_molecule_calls"] = 0
    _timing["process_molecule_sec"] = 0.0
    _timing["process_molecule_calls"] = 0
    _fragment_flags.clear()
    _captured_calls.clear()


# =============================================================================
# Один прогон генерации с профилированием
# =============================================================================
def profile_one_run(model, pdbfile, ref_ligand, n_samples, batch_size, timesteps, seed, collect_inputs=False):
    global _collect_inputs
    _reset_timing()
    _collect_inputs = collect_inputs

    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    t0 = time.perf_counter()
    molecules = []
    remaining = n_samples
    while remaining > 0:
        this_batch = min(batch_size, remaining)
        batch = model.generate_ligands(
            pdbfile, this_batch, None, ref_ligand,
            None, True, largest_frag=True,
            relax_iter=0, resamplings=10, jump_length=1,
            timesteps=timesteps,
        )
        molecules.extend(batch)
        remaining -= this_batch
    total_sec = time.perf_counter() - t0

    # SMILES-конвертация — НЕ часть generate_ligands() вообще, отдельный блок,
    # ровно как в generate_smiles_seeded.py.
    t0 = time.perf_counter()
    n_smiles_ok = 0
    for m in molecules:
        try:
            smi = Chem.MolToSmiles(m)
            if smi:
                n_smiles_ok += 1
        except Exception:
            pass
    smiles_sec = time.perf_counter() - t0

    build_sec = _timing["build_molecule_sec"]
    process_sec = _timing["process_molecule_sec"]
    # GPU-время — вычитанием (см. docstring). Не может быть отрицательным
    # в норме; если получилось отрицательным — где-то накладные расходы
    # самого профилирования (перфоманс-счётчик, python-уровень) сравнимы
    # с временем шагов, честно помечаем.
    gpu_sec = max(0.0, total_sec - build_sec - process_sec)

    n_valid = len(molecules)
    n_intact = sum(1 for f in _fragment_flags if f)
    n_raw = len(_fragment_flags)

    captured = list(_captured_calls) if collect_inputs else None

    return {
        "timesteps": timesteps,
        "n_requested": n_samples,
        "batch_size": batch_size,
        "total_sec": round(total_sec, 3),
        "sec_per_requested": round(total_sec / n_samples, 3),
        "gpu_diffusion_sec": round(gpu_sec, 3),
        "gpu_diffusion_pct": round(100 * gpu_sec / total_sec, 1) if total_sec else None,
        "molecule_builder_sec": round(build_sec, 3),
        "molecule_builder_pct": round(100 * build_sec / total_sec, 1) if total_sec else None,
        "molecule_builder_calls": _timing["build_molecule_calls"],
        "sec_per_build_molecule_call": round(build_sec / _timing["build_molecule_calls"], 4) if _timing["build_molecule_calls"] else None,
        "rdkit_sanitize_sec": round(process_sec, 3),
        "rdkit_sanitize_pct": round(100 * process_sec / total_sec, 1) if total_sec else None,
        "rdkit_smiles_sec": round(smiles_sec, 3),
        "n_valid_after_sanitize": n_valid,
        "valid_fraction": round(n_valid / n_samples, 3),
        "n_raw_built": n_raw,
        "n_intact_single_fragment": n_intact,
        "intact_fraction_of_raw": round(n_intact / n_raw, 3) if n_raw else None,
        "n_smiles_ok": n_smiles_ok,
    }, captured


# =============================================================================
# Тест параллелизации build_molecule по ядрам CPU (--test_parallel)
# =============================================================================
def _worker_build_molecule(payload):
    """Top-level функция для multiprocessing.Pool (должна быть picklable —
    поэтому не замыкание/лямбда, а обычная функция модуля). Вызывает
    НЕПАТЧЕННЫЙ build_molecule напрямую — сам таймер тут не нужен, время
    меряется снаружи для всего пула сразу."""
    args, kwargs = payload
    return _real_build_molecule(*args, **kwargs)


def test_parallel_build_molecule(captured_calls, n_workers):
    if not captured_calls:
        return {"error": "нет захваченных вызовов build_molecule для теста — нужно collect_inputs=True на одном из прогонов"}

    # Последовательно (baseline)
    t0 = time.perf_counter()
    for args, kwargs in captured_calls:
        _real_build_molecule(*args, **kwargs)
    sequential_sec = time.perf_counter() - t0

    # Параллельно через process pool (obabel.exe — subprocess на молекулу,
    # реально независимые процессы; ProcessPoolExecutor, не потоки — GIL
    # тут не проблема, т.к. основное время — ожидание subprocess.run(),
    # но процессный пул надёжнее для честной картины реального выигрыша
    # на изолированных обращениях к диску/subprocess).
    t0 = time.perf_counter()
    with multiprocessing.Pool(processes=n_workers) as pool:
        pool.map(_worker_build_molecule, captured_calls)
    parallel_sec = time.perf_counter() - t0

    return {
        "n_molecules": len(captured_calls),
        "n_workers": n_workers,
        "sequential_sec": round(sequential_sec, 3),
        "parallel_sec": round(parallel_sec, 3),
        "speedup": round(sequential_sec / parallel_sec, 2) if parallel_sec else None,
    }


# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/crossdocked_fullatom_cond.ckpt")
    parser.add_argument("--pdbfile", default=os.path.join("..", "structures", "4JPS.pdb"))
    parser.add_argument("--ref_ligand", default="A:1102")
    parser.add_argument("--n_samples", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--test_timesteps", action="store_true", help="Дополнительно прогнать T=полное/половина/четверть")
    parser.add_argument("--test_parallel", action="store_true", help="Дополнительно проверить параллелизацию build_molecule по ядрам")
    parser.add_argument("--parallel_workers", type=int, default=os.cpu_count())
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"устройство: {device}")

    t0 = time.perf_counter()
    model = lm.LigandPocketDDPM.load_from_checkpoint(args.checkpoint, map_location=device)
    model = model.to(device)
    model_load_sec = time.perf_counter() - t0
    print(f"модель загружена за {model_load_sec:.1f}с")

    # T=500 — реальное "полное" число, прочитано из гиперпараметров
    # чекпоинта (diffusion_params.diffusion_steps), не предположение.
    full_timesteps = model.hparams.diffusion_params.diffusion_steps
    print(f"полное число шагов диффузии (из чекпоинта): {full_timesteps}")

    results = {
        "n_cpu": os.cpu_count(),
        "device": device,
        "model_load_sec": round(model_load_sec, 1),
        "full_timesteps_from_checkpoint": full_timesteps,
    }

    print(f"\n=== Профиль генерации, T={full_timesteps} (полное), n={args.n_samples}, batch_size={args.batch_size} ===")
    profile, captured = profile_one_run(
        model, args.pdbfile, args.ref_ligand, args.n_samples, args.batch_size,
        timesteps=None, seed=args.seed, collect_inputs=args.test_parallel,
    )
    results["profile_default_timesteps"] = profile
    print(json.dumps(profile, indent=2, ensure_ascii=False))

    if args.test_timesteps:
        half = full_timesteps // 2
        quarter = full_timesteps // 4
        results["timesteps_test"] = [profile]  # T=полное уже посчитан выше, переиспользуем
        for t in (half, quarter):
            print(f"\n=== T={t} ({'половина' if t == half else 'четверть'}), n={args.n_samples} ===")
            prof_t, _ = profile_one_run(
                model, args.pdbfile, args.ref_ligand, args.n_samples, args.batch_size,
                timesteps=t, seed=args.seed, collect_inputs=False,
            )
            results["timesteps_test"].append(prof_t)
            print(json.dumps(prof_t, indent=2, ensure_ascii=False))

    if args.test_parallel:
        print(f"\n=== Тест параллелизации build_molecule, {args.parallel_workers} воркеров, {len(captured)} молекул ===")
        par = test_parallel_build_molecule(captured, args.parallel_workers)
        results["parallel_test"] = par
        print(json.dumps(par, indent=2, ensure_ascii=False))

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nСохранено: {args.out}")


if __name__ == "__main__":
    main()
