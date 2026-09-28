@echo off
setlocal EnableExtensions
title AcademiaSD - LTX-2.3 LoRAlab Trainer
cd /d "%~dp0"
set "BASE_DIR=%~dp0"
set "PYTHON_EXE=%BASE_DIR%venv\Scripts\python.exe"

echo.
echo ================================================================
echo        ACADEMIASD - LTX-2.3 LORALAB TRAINER
echo ================================================================
echo.

if not exist "%PYTHON_EXE%" (
    echo [ERROR] No se ha encontrado el entorno virtual / Virtual environment not found:
    echo   %PYTHON_EXE%
    echo.
    echo Ejecuta primero Install_LoRAlab.bat / Run Install_LoRAlab.bat first.
    echo.
    pause
    exit /b 1
)

if not exist "%BASE_DIR%server_ltx23.py" (
    echo [ERROR] No existe / Missing: server_ltx23.py
    pause
    exit /b 1
)

if not exist "%BASE_DIR%trainer_ui_ltx23.html" (
    echo [ERROR] No existe / Missing: trainer_ui_ltx23.html
    pause
    exit /b 1
)

echo Iniciando servidor web / Starting web server...
echo.
echo     http://127.0.0.1:5000
echo.
echo Cierra esta ventana para detener el servidor / Close this window to stop the server.
echo.
"%PYTHON_EXE%" "%BASE_DIR%server_ltx23.py"

echo.
echo ================================================================
echo El servidor ha finalizado / The server has stopped.
echo ================================================================
pause
endlocal
