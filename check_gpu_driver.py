"""
check_gpu_driver.py - определяет NVIDIA GPU + версию драйвера через
nvidia-smi, сравнивает с минимумом, нужным для gnina (CUDA). НЕ ставит
драйвер сам - только определяет и, если версия не подходит, печатает
явную рекомендацию, что и откуда поставить. Автоустановка драйвера
намеренно не делается: это системное изменение, которое может задеть
другой софт, обычно требует перезагрузки, и его нельзя тихо накатывать
без ведома человека за компьютером.

ЧЕСТНО: минимальная версия ниже - консервативная эвристика (совместимость
с CUDA 12.x, на чём собрано большинство современных gnina-сборок), не
гарантированное значение, снятое именно с требований конкретной сборки
gnina, которая будет использоваться - если gnina всё равно упадёт на
"подходящей" по этой проверке версии, ошибка самого gnina/CUDA при
первом реальном запуске - более авторитетный сигнал, чем эта проверка.

Использование:
    python check_gpu_driver.py
"""
import subprocess
import sys

MIN_DRIVER_VERSION = (525, 60)  # эвристика под CUDA 12.x, не проверенный минимум именно для gnina
DRIVER_DOWNLOAD_URL = "https://www.nvidia.com/Download/index.aspx"


def parse_version(v):
    parts = v.strip().split(".")
    return tuple(int(p) for p in parts[:2] if p.isdigit())


def main():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15,
        )
    except FileNotFoundError:
        print("[gpu_check] nvidia-smi не найден - либо нет NVIDIA GPU, либо драйвер не установлен вообще.")
        print(f"[gpu_check] Без GPU/драйвера gnina (CNN-рескоринг) работать не будет - "
              f"докинг пойдёт только через Vina, без CNN-части. Поставить драйвер: {DRIVER_DOWNLOAD_URL}")
        sys.exit(1)
    except subprocess.TimeoutExpired:
        print("[gpu_check] nvidia-smi завис/не ответил за 15с - неясное состояние драйвера, проверь вручную.")
        sys.exit(1)

    if result.returncode != 0 or not result.stdout.strip():
        print(f"[gpu_check] nvidia-smi вернул ошибку: {result.stderr[:300]}")
        print(f"[gpu_check] Похоже, драйвер не работает корректно. Переустановить: {DRIVER_DOWNLOAD_URL}")
        sys.exit(1)

    line = result.stdout.strip().splitlines()[0]
    gpu_name, driver_version = [x.strip() for x in line.split(",")]
    print(f"[gpu_check] GPU: {gpu_name}")
    print(f"[gpu_check] Версия драйвера: {driver_version}")

    version_tuple = parse_version(driver_version)
    if version_tuple < MIN_DRIVER_VERSION:
        print(f"\n[gpu_check] [!] Версия драйвера {driver_version} СТАРЕЕ рекомендуемой "
              f"{'.'.join(map(str, MIN_DRIVER_VERSION))} (эвристика под CUDA 12.x, не точный минимум gnina).")
        print(f"[gpu_check] Рекомендация: обновить драйвер здесь -> {DRIVER_DOWNLOAD_URL}")
        print(f"[gpu_check] Без обновления gnina (CNN-рескоринг) может не запуститься или упасть с ошибкой "
              f"несовместимости CUDA - но сам Vina-докинг это не остановит, только gnina-часть.")
        sys.exit(2)
    else:
        print(f"[gpu_check] Версия драйвера в порядке (>= {'.'.join(map(str, MIN_DRIVER_VERSION))}).")
        sys.exit(0)


if __name__ == "__main__":
    main()
