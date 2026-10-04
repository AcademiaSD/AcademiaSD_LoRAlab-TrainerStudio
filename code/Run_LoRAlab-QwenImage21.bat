@echo off
setlocal EnableExtensions
title AcademiaSD LoRAlab Trainer Studio - Qwen-Image 2.1
cd /d "%~dp0.."
set "BASE_DIR=%CD%\"
set "PYTHON_EXE=%BASE_DIR%venv\Scripts\python.exe"
:: LORALAB_PYTHON: el python de tu propio entorno (conda, uv...) en lugar del venv / your own environment's python instead of the venv
if defined LORALAB_PYTHON set "PYTHON_EXE=%LORALAB_PYTHON%"

echo.
echo ================================================================
echo        ACADEMIASD LORALAB TRAINER STUDIO - QWEN-IMAGE 2.1
echo ================================================================
echo.

if not exist "%PYTHON_EXE%" (
    echo [ERROR] No se ha encontrado el entorno virtual / Virtual environment not found:
    echo   %PYTHON_EXE%
    echo.
    echo Ejecuta primero Install_LoRAlab-TrainerStudio.bat / Run Install_LoRAlab-TrainerStudio.bat first.
    echo.
    pause
    exit /b 1
)

if not exist "%BASE_DIR%scripts\server_qwenimage21.py" (
    echo [ERROR] No existe / Missing: scripts\server_qwenimage21.py
    pause
    exit /b 1
)

if not exist "%BASE_DIR%GUI\trainer_ui_qwenimage21.html" (
    echo [ERROR] No existe / Missing: GUI\trainer_ui_qwenimage21.html
    pause
    exit /b 1
)

echo Iniciando servidor web / Starting web server...
echo.
echo Cierra esta ventana para detener el servidor / Close this window to stop the server.
echo.
"%PYTHON_EXE%" "%BASE_DIR%scripts\server_qwenimage21.py"

echo.
echo ================================================================
echo El servidor ha finalizado / The server has stopped.
echo ================================================================
pause
endlocal
