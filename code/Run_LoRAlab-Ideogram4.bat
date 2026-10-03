@echo off
setlocal EnableExtensions
title AcademiaSD LoRAlab Trainer Studio - Ideogram 4
cd /d "%~dp0.."
set "BASE_DIR=%CD%\"
set "PYTHON_EXE=%BASE_DIR%venv\Scripts\python.exe"

echo.
echo ================================================================
echo        ACADEMIASD LORALAB TRAINER STUDIO - IDEOGRAM 4
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

if not exist "%BASE_DIR%scripts\server_ideogram4.py" (
    echo [ERROR] No existe / Missing: scripts\server_ideogram4.py
    pause
    exit /b 1
)

if not exist "%BASE_DIR%GUI\trainer_ui_ideogram4.html" (
    echo [ERROR] No existe / Missing: GUI\trainer_ui_ideogram4.html
    pause
    exit /b 1
)

echo Iniciando servidor web / Starting web server...
echo.
echo     http://127.0.0.1:5000
echo.
echo Cierra esta ventana para detener el servidor / Close this window to stop the server.
echo.
"%PYTHON_EXE%" "%BASE_DIR%scripts\server_ideogram4.py"

echo.
echo ================================================================
echo El servidor ha finalizado / The server has stopped.
echo ================================================================
pause
endlocal
