@echo off
setlocal EnableExtensions
title AcademiaSD - LoRAlab Trainer Studio Launcher
cd /d "%~dp0"
set "PYTHON_EXE=%~dp0venv\Scripts\python.exe"
:: LORALAB_PYTHON: el python de tu propio entorno (conda, uv...) en lugar del venv / your own environment's python instead of the venv
if defined LORALAB_PYTHON set "PYTHON_EXE=%LORALAB_PYTHON%"

if not exist "%PYTHON_EXE%" (
    echo [ERROR] No se ha encontrado el entorno virtual / Virtual environment not found:
    echo   %PYTHON_EXE%
    echo.
    echo Ejecuta primero Install_LoRAlab-TrainerStudio.bat / Run Install_LoRAlab-TrainerStudio.bat first.
    echo.
    pause
    exit /b 1
)

"%PYTHON_EXE%" "%~dp0scripts\launcher.py"
pause
endlocal
