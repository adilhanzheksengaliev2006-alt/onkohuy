"""
generate_smiles_seeded.py — генерация лигандов DiffSBDD с явным сидом.

Зачем отдельный скрипт: официальный generate_ligands.py не принимает
--seed (нет управления воспроизводимостью по сидам), а для пайплайна
с 8 фиксированными сидами это обязательно. Тонкая обёртка вокруг того
же lightning_modules.LigandPocketDDPM, что использует сам
generate_ligands.py — модель и метод генерации не меняются.

Запускается ВНУТРИ env diffsbdd (python другой, несовместимый с env
molgen, где стоит остальной пайплайн проекта) — поэтому это отдельный
процесс, а не импортируемый модуль.

Идемпотентность для resume: если outfile (SMILES) и sdf_outfile уже
существуют и непустые, генерация для этого сида пропускается — сама
генерация занимает минуты и её не нужно повторять при перезапуске
пайплайна после сбоя.

Использование:
  python generate_smiles_seeded.py <checkpoint> <pdbfile> <ref_ligand>
      --seed <int> --n_samples <int> --outfile <smiles.json>
      --sdf_outfile <mols.sdf> [--timesteps <int>]
"""

import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
from openbabel import openbabel

openbabel.obErrorLog.StopLogging()

import utils  # noqa: E402
from lightning_modules import LigandPocketDDPM  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=str)
    parser.add_argument("pdbfile", type=str)
    parser.add_argument("ref_ligand", type=str)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--n_samples", type=int, required=True)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--outfile", type=str, required=True)
    parser.add_argument("--sdf_outfile", type=str, required=True)
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--sanitize", action="store_true", default=True)
    args = parser.parse_args()

    if os.path.exists(args.outfile) and os.path.getsize(args.outfile) > 0:
        print(f"[seed {args.seed}] уже сгенерировано, пропуск: {args.outfile}")
        return

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # ВАЖНО: дефолт НЕ n_samples одним батчем. У DiffSBDD-пилота (rtx a4000,
    # заведомо много VRAM) generate_ligands.py вызывался без --batch_size
    # и получал n_samples=60 одним куском — там это было безопасно. У нас
    # GTX 1650 с 4 ГБ VRAM (см. structures/README про железо) — тот же
    # паттерн на n_samples=1000 почти наверняка даст OOM. Явный маленький
    # дефолт заставляет либо пройти замер шага 8 промта (батч 1/2/4) и
    # передать безопасное значение осознанно, либо получить безопасный,
    # пусть и не самый быстрый, результат "из коробки".
    batch_size = args.batch_size or min(4, args.n_samples)
    print(f"[seed {args.seed}] batch_size={batch_size}" + (" (дефолт, не задан явно — см. комментарий в коде)" if args.batch_size is None else ""))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[seed {args.seed}] устройство: {device}")

    t0 = time.time()
    model = LigandPocketDDPM.load_from_checkpoint(args.checkpoint, map_location=device)
    model = model.to(device)
    load_sec = time.time() - t0
    print(f"[seed {args.seed}] модель загружена за {load_sec:.1f}с")

    # Верификация чекпоинта по гиперпараметрам, СОХРАНЁННЫМ ВНУТРИ файла —
    # не по имени файла. Повод: issue #57 в DiffSBDD (github.com/arneschneuing/
    # DiffSBDD/issues/57) — человек утверждал, что crossdocked_fullatom_cond.ckpt
    # на самом деле Cα-модель, судя по форме тензора atom_encoder. Проверка
    # вручную (сравнение с dataset_params в constants.py) показала, что вывод
    # был ошибочным — форма тензора совпадает у нескольких datasets с разным
    # смыслом, вводит в заблуждение. Надёжный источник истины — вот эти два
    # поля, которые PyTorch Lightning сохраняет в hparams при чекпоинте.
    dataset_hp = getattr(model.hparams, "dataset", None) if hasattr(model, "hparams") else None
    pocket_repr_hp = getattr(model.hparams, "pocket_representation", None) if hasattr(model, "hparams") else None
    print(f"[seed {args.seed}] чекпоинт: dataset={dataset_hp!r}, pocket_representation={pocket_repr_hp!r}")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    t0 = time.time()
    molecules = []
    remaining = args.n_samples
    while remaining > 0:
        this_batch = min(batch_size, remaining)
        batch = model.generate_ligands(
            args.pdbfile, this_batch, None, args.ref_ligand,
            None, args.sanitize, largest_frag=True,
            relax_iter=0, resamplings=10, jump_length=1,
            timesteps=args.timesteps,
        )
        molecules.extend(batch)
        remaining -= this_batch
    gen_sec = time.time() - t0

    # Пик VRAM за генерацию этого сида. Повод: issue #40 в DiffSBDD — у
    # пользователя 20 молекул (num_nodes_lig=50) заняли 5.3 ГБ, больше
    # всей нашей карты (4 ГБ). Без телеметрии по каждому сиду шаг 8
    # промта (замер пикового VRAM) остался бы разовым экспериментом, а не
    # постоянным контролем — здесь пишется в каждый прогон бесплатно.
    peak_vram_mb = round(torch.cuda.max_memory_allocated(device) / 1024**2, 1) if torch.cuda.is_available() else None
    if peak_vram_mb is not None:
        print(f"[seed {args.seed}] пик VRAM: {peak_vram_mb:.0f} МБ")

    os.makedirs(os.path.dirname(os.path.abspath(args.sdf_outfile)), exist_ok=True)
    utils.write_sdf_file(args.sdf_outfile, molecules)

    from rdkit import Chem

    # Подозрительные размеры колец: авторы DiffSBDD прямо пишут в статье,
    # что "very small and very large ring systems are typically
    # over-represented in DiffSBDD molecules" — известное, признанное
    # смещение генератора. НЕ фильтруем (как PAINS/Brenk/ADMET в проекте —
    # считаем, не отсеиваем на этом шаге), только считаем и помечаем,
    # чтобы это было видно на этапе анализа, а не терялось молча.
    UNUSUAL_RING_MIN, UNUSUAL_RING_MAX = 3, 8

    def _has_unusual_ring(mol):
        ri = mol.GetRingInfo()
        return any(not (UNUSUAL_RING_MIN <= len(r) <= UNUSUAL_RING_MAX) for r in ri.AtomRings())

    smiles_list = []
    n_none = 0
    n_unusual_ring = 0
    for m in molecules:
        if m is None:
            n_none += 1
            continue
        try:
            smi = Chem.MolToSmiles(m)
        except Exception:
            smi = None
        if smi is None:
            n_none += 1
            continue
        if _has_unusual_ring(m):
            n_unusual_ring += 1
        smiles_list.append(smi)

    result = {
        "seed": args.seed,
        "checkpoint": os.path.abspath(args.checkpoint),
        "checkpoint_dataset": dataset_hp,
        "checkpoint_pocket_representation": pocket_repr_hp,
        "timesteps": args.timesteps,
        "batch_size": batch_size,
        "peak_vram_mb": peak_vram_mb,
        "n_requested": args.n_samples,
        "n_generated_raw": len(molecules),
        "n_none_or_unparseable": n_none,
        "n_valid_smiles": len(smiles_list),
        "n_unusual_ring_size": n_unusual_ring,
        "model_load_sec": round(load_sec, 1),
        "generation_sec": round(gen_sec, 1),
        "smiles": smiles_list,
    }
    with open(args.outfile, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(
        f"[seed {args.seed}] готово: {len(smiles_list)}/{args.n_samples} валидных SMILES "
        f"({n_unusual_ring} с необычным циклом), генерация {gen_sec:.1f}с "
        f"({gen_sec / max(len(molecules), 1):.2f}с/молекулу)"
    )


if __name__ == "__main__":
    main()
