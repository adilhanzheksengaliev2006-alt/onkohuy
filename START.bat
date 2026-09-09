@echo off
REM START.bat - "скачал и запустил". Сам находит conda, сам создаёт
REM окружение "molgen" и ставит зависимости, если их ещё нет (только
REM при первом запуске - несколько минут), потом сразу стартует
REM докинг-очередь. Требует интернет ТОЛЬКО при первом запуске (для
REM установки пакетов) - дальше работает полностью офлайн.
REM
REM ЧЕСТНО: WSL2 + gnina + fpocket этот файл НЕ ставит - это отдельный,
REM специфичный для железа шаг (сборка gnina под конкретную видеокарту/
REM драйвер). Без него докинг всё равно пойдёт (Vina), но без
REM CNN-рескоринга gnina и без fpocket-стадии - см. SETUP_NEW_MACHINE.md.

setlocal enabledelayedexpansion

set CONDA_BAT=
for %%P in (
    "%USERPROFILE%\miniconda3\condabin\conda.bat"
    "%USERPROFILE%\anaconda3\condabin\conda.bat"
    "%LOCALAPPDATA%\miniconda3\condabin\conda.bat"
    "%LOCALAPPDATA%\anaconda3\condabin\conda.bat"
    "C:\ProgramData\miniconda3\condabin\conda.bat"
    "C:\ProgramData\anaconda3\condabin\conda.bat"
) do (
    if exist %%P (
        set CONDA_BAT=%%P
    )
)

echo ============================================
echo   Onco-Target-Explorer - запуск
echo ============================================
echo.

cd /d "%~dp0"

if "!CONDA_BAT!"=="" (
    echo [ОШИБКА] Не нашёл conda/miniconda на этом компьютере.
    echo Сначала поставь Miniconda: https://www.anaconda.com/download
    echo При установке отметь "Add to PATH" - тогда этот файл сам всё найдёт.
    pause
    exit /b 1
)

echo [1/3] Проверяю conda-окружение "molgen"...
call "!CONDA_BAT!" env list | findstr /C:"molgen" >nul
if errorlevel 1 (
    echo   Окружения нет - создаю и ставлю пакеты ^(один раз, несколько минут, нужен интернет^)...
    call "!CONDA_BAT!" create -n molgen python=3.10.20 -y
    call "!CONDA_BAT!" activate molgen
    pip install -r environment_molgen_requirements.txt
) else (
    echo   Уже есть, пропускаю установку.
    call "!CONDA_BAT!" activate molgen
)

echo.
echo [2/4] Проверка видеокарты/драйвера (для gnina CNN-рескоринга)...
python check_gpu_driver.py
if errorlevel 2 (
    echo   Драйвер старый - см. рекомендацию выше. Продолжаю без остановки:
    echo   Vina-докинг это не блокирует, только gnina-часть может не заработать.
)
if errorlevel 1 if not errorlevel 2 (
    echo   Видеокарты NVIDIA не найдено - продолжаю, будет работать только Vina.
)

echo.
echo [3/4] Проверка офлайн-готовности мишеней...
python check_offline_readiness.py
if errorlevel 1 (
    echo.
    echo [ОШИБКА] check_offline_readiness.py упал - смотри сообщение выше.
    pause
    exit /b 1
)

echo.
echo [4/4] Запуск главной очереди докинга (Vina, resume-safe)...
echo Это будет работать долго - можно закрывать/открывать это окно,
echo прогресс не потеряется (сохраняется после каждого лиганда).
echo Без отдельно настроенного WSL2+gnina докинг всё равно идёт (Vina),
echo просто без CNN-рескоринга - см. SETUP_NEW_MACHINE.md, если он нужен.
echo.

python run_docking_queue.py --only-gated

echo.
echo ============================================
echo   Очередь остановлена (закончилась или прервана)
echo ============================================
pause
