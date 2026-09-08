"""
run_controls.py — оркестратор стадийного пайплайна из 18 тестов.

ЕДИНСТВЕННЫЙ гейт: Stage 1 (Test 5, redock RMSD). Если он не пройден -
Stage 2+ для этой мишени не запускаются (печатается явное предупреждение),
но это единственное исключение из общего правила "ничего не исключаем".
Все остальные стадии считаются и пишут числа/флаги, даже если что-то
выглядит "плохо".

Resume: каждый тест сам пропускает пересчёт, если результат уже есть в
results/<GENE>/stage_N.json (см. protocol.test_already_done) - --force
пересчитывает конкретно запрошенную стадию.

Stage 5/6 (тесты 6,8,16) НЕ входят в --stage all по умолчанию - это
дорогие тесты по протоколу, запускаются только их собственными скриптами
напрямую с --confirm.

Использование:
    python controls/run_controls.py GENE --stage 0
    python controls/run_controls.py GENE --stage 1
    python controls/run_controls.py GENE --stage 2 --smoke
    python controls/run_controls.py GENE --stage 3
    python controls/run_controls.py GENE --stage 4
    python controls/run_controls.py GENE --stage 7   (после того как 0..4 сделаны для всех нужных мишеней)
    python controls/run_controls.py GENE --stage all [--smoke]
    добавь --force чтобы пересчитать конкретную стадию
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import protocol  # noqa: E402
import test01_dataset_bias as t01  # noqa: E402
import test02_ligand_only_baseline as t02  # noqa: E402
import test03_ave_bias as t03  # noqa: E402
import test04_scaffold_bias as t04  # noqa: E402
import test05_redock as t05  # noqa: E402
import test07_label_shuffle as t07  # noqa: E402
import test09_seed_variance as t09  # noqa: E402
import test10_exhaustiveness as t10  # noqa: E402
import test11_screening_power as t11  # noqa: E402
import test12_ranking_power as t12  # noqa: E402
import test13_size_bias as t13  # noqa: E402
import test14_ligand_efficiency as t14  # noqa: E402
import test15_posebusters as t15  # noqa: E402
import test17_pocket_descriptors as t17  # noqa: E402
import test18_bias_vs_pocket as t18  # noqa: E402


def stage0(gene, force):
    print(f"\n########## STAGE 0 ({gene}): тесты 1-4 ##########")
    t01.run(gene, force)
    t02.run(gene, force)
    t03.run(gene, force)
    t04.run(gene, force)


def stage1(gene, force):
    print(f"\n########## STAGE 1 ({gene}): тест 5 - ГЕЙТ ##########")
    result = t05.run(gene, force)
    return result.get("passed_gate", False)


def stage2(gene, force, smoke):
    print(f"\n########## STAGE 2 ({gene}): тесты 9-10 ##########")
    t09.run(gene, force, smoke)
    t10.run(gene, force, smoke)


def stage3(gene, force):
    print(f"\n########## STAGE 3 ({gene}): тесты 7,11-14 ##########")
    t07.run(gene, force)
    t11.run(gene, force)
    t12.run(gene, force)
    t13.run(gene, force)
    t14.run(gene, force)


def stage4(gene, force):
    print(f"\n########## STAGE 4 ({gene}): тест 15 ##########")
    t15.run(gene, force)


def stage7(gene, force):
    print(f"\n########## STAGE 7 ({gene}): тест 17 + сводная тест 18 ##########")
    t17.run(gene, force)
    t18.run(force)


def gate_passed(gene):
    data = protocol.load_stage(gene, 1)
    return bool(data and data.get("test05_redock", {}).get("passed_gate"))


def run_all(gene, force, smoke):
    stage0(gene, force)
    passed = stage1(gene, force)
    if not passed:
        status = (protocol.load_stage(gene, 1) or {}).get("test05_redock", {}).get("status")
        if status == "error":
            print(f"\n[run_controls] {gene}: гейт Test 5 НЕ ВЫПОЛНИЛСЯ (status=error, сбой "
                  f"выполнения - сеть/файлы/др., НЕ RMSD-провал). Stage 2+ НЕ запускаю. "
                  f"Почини причину сбоя и перезапусти '--stage 1 --force', это не 'мишень плохая'.")
        else:
            print(f"\n[run_controls] {gene}: НЕ прошёл гейт Test 5 (status=fail, RMSD выше порога) - "
                  f"Stage 2+ для этой мишени НЕ запускаю. Это единственная стадия, "
                  f"где отказ останавливает обработку мишени.")
        return
    stage2(gene, force, smoke)
    stage3(gene, force)
    stage4(gene, force)
    stage7(gene, force)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("gene")
    parser.add_argument("--stage", required=True, choices=["0", "1", "2", "3", "4", "7", "all"])
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Stage 2: маленькая подвыборка вместо полной")
    args = parser.parse_args()

    print(f"[run_controls] {args.gene}, stage={args.stage}, force={args.force}, smoke={args.smoke}")

    if args.stage == "0":
        stage0(args.gene, args.force)
    elif args.stage == "1":
        stage1(args.gene, args.force)
    elif args.stage == "2":
        if not gate_passed(args.gene):
            print(f"[run_controls] [warn] {args.gene} ещё не прошёл Stage 1 гейт (или он не запускался) - "
                  f"запусти --stage 1 сначала. Продолжаю по явной просьбе, но интерпретируй с осторожностью.")
        stage2(args.gene, args.force, args.smoke)
    elif args.stage == "3":
        stage3(args.gene, args.force)
    elif args.stage == "4":
        stage4(args.gene, args.force)
    elif args.stage == "7":
        stage7(args.gene, args.force)
    elif args.stage == "all":
        run_all(args.gene, args.force, args.smoke)


if __name__ == "__main__":
    main()
