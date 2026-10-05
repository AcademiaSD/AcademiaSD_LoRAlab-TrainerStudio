@echo off
setlocal EnableExtensions EnableDelayedExpansion

title AcademiaSD - LoRAlab Trainer Studio Updater
color 0B

cd /d "%~dp0"

echo ================================================================
echo   ACADEMIASD - LORALAB TRAINER STUDIO UPDATER
echo   [EN] Repository Update Utility
echo   [ES] Utilidad de Actualizacion del Repositorio
echo ================================================================
echo.

rem 1. Check Git
where git >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Git is not installed or not available in PATH.
    echo [ERROR] Git no esta instalado o no esta disponible en el PATH.
    echo.
    echo [EN] Please install Git from https://git-scm.com/
    echo [ES] Por favor instala Git desde https://git-scm.com/
    echo [EN] Or run Install_LoRAlab-TrainerStudio.bat, which installs it.
    echo [ES] O ejecuta Install_LoRAlab-TrainerStudio.bat, que lo instala.
    echo.
    pause
    exit /b 1
)

rem 2. Check .git
if not exist ".git" (
    echo [INFO] Initializing Git repository...
    echo [INFO] Inicializando repositorio Git...
    echo.
    git init >nul 2>&1
    git remote add origin https://github.com/AcademiaSD/AcademiaSD_LoRAlab-TrainerStudio.git >nul 2>&1
)

rem 3. Ensure remote URL
git remote set-url origin https://github.com/AcademiaSD/AcademiaSD_LoRAlab-TrainerStudio.git >nul 2>&1

echo [EN] Syncing latest updates from GitHub...
echo [ES] Sincronizando ultimas actualizaciones desde GitHub...
echo.

rem 4. Rama a actualizar: la de esta carpeta (main, dev...); main si no hay ninguna
rem    Branch to update: this folder's branch (main, dev...); main if there is none
set "BRANCH="
for /f "delims=" %%b in ('git symbolic-ref --short -q HEAD 2^>nul') do set "BRANCH=%%b"
if not defined BRANCH set "BRANCH=main"

git fetch origin !BRANCH! >nul 2>&1
if errorlevel 1 (
    set "BRANCH=main"
    git fetch origin main >nul 2>&1
    if errorlevel 1 (
        set "BRANCH=master"
        git fetch origin master >nul 2>&1
    )
)

echo [EN] Branch: !BRANCH!
echo [ES] Rama: !BRANCH!
echo.

rem 5. Force reset
git checkout -f !BRANCH! >nul 2>&1
git reset --hard origin/!BRANCH! >nul 2>&1

if errorlevel 1 (
    echo.
    echo [ERROR] Failed to update repository from GitHub.
    echo [ERROR] No se pudo actualizar el repositorio desde GitHub.
    echo.
    pause
    exit /b 1
)

echo.
echo ================================================================
echo [OK] Repository updated successfully! / Repositorio actualizado!
echo ================================================================
echo.
echo [EN] Update process completed.
echo [ES] Proceso de actualizacion completado.
echo.

rem 6. Librerias nuevas o actualizadas de requirements.txt / New or updated libraries from requirements.txt
echo Actualizando librerias de Python... / Updating Python libraries...
set "PY_EXE=%~dp0venv\Scripts\python.exe"
if defined LORALAB_PYTHON set "PY_EXE=%LORALAB_PYTHON%"
if exist "!PY_EXE!" (
    "!PY_EXE!" -m pip install -r "%~dp0requirements.txt"
    if errorlevel 1 echo [AVISO] No se pudieron actualizar las librerias / [WARNING] Could not update the libraries.
) else (
    echo [AVISO] No hay entorno: ejecuta Install_LoRAlab-TrainerStudio.bat / [WARNING] No environment: run Install_LoRAlab-TrainerStudio.bat
)
echo.
pause
