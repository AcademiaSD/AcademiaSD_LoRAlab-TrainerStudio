@echo off
setlocal EnableExtensions
title AcademiaSD - LoRAlab Trainer Studio Launcher
cd /d "%~dp0"
set "PYTHON_EXE=%~dp0venv\Scripts\python.exe"

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
