@echo off
REM START.bat - запускает весь пайплайн одной кнопкой (двойной клик).
REM Сам находит conda в стандартных местах установки - не требует,
REM чтобы conda была заранее прописана в PATH/инициализирована для cmd.

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
echo   Onco-Target-Explorer - запуск докинг-очереди
echo ============================================
echo.

cd /d "%~dp0"

if "!CONDA_BAT!"=="" (
    echo [ОШИБКА] Не нашёл conda в стандартных местах установки.
    echo Открой этот файл в блокноте и добавь свой путь в список CONDA_BAT выше,
    echo либо запусти вручную из Anaconda Prompt:
    echo   conda activate molgen
    echo   python run_docking_queue.py --only-gated
    pause
    exit /b 1
)

call "!CONDA_BAT!" activate molgen
if errorlevel 1 (
    echo.
    echo [ОШИБКА] Не удалось активировать conda-окружение "molgen".
    echo Проверь: conda env list  - оно должно быть в списке.
    echo Если его нет - сначала выполни шаги из SETUP_NEW_MACHINE.md.
    pause
    exit /b 1
)

echo [1/2] Проверка офлайн-готовности мишеней...
python check_offline_readiness.py
if errorlevel 1 (
    echo.
    echo [ОШИБКА] check_offline_readiness.py упал - смотри сообщение выше.
    pause
    exit /b 1
)

echo.
echo [2/2] Запуск главной очереди докинга (Vina+gnina, resume-safe)...
echo Это будет работать долго - можно закрывать/открывать это окно,
echo прогресс не потеряется (сохраняется после каждого лиганда).
echo.

python run_docking_queue.py --only-gated

echo.
echo ============================================
echo   Очередь остановлена (закончилась или прервана)
echo ============================================
pause
