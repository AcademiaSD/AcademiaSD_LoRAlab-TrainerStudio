@echo off
setlocal EnableExtensions
title AcademiaSD LoRAlab - Conversor Fibo 2.0 NF4
cd /d "%~dp0..\.."
set "BASE_DIR=%CD%\"
set "PYTHON_EXE=%BASE_DIR%venv\Scripts\python.exe"
:: LORALAB_PYTHON: el python de tu propio entorno (conda, uv...) en lugar del venv
if defined LORALAB_PYTHON set "PYTHON_EXE=%LORALAB_PYTHON%"

echo.
echo ================================================================
echo        ACADEMIASD LORALAB - CONVERSOR FIBO 2.0 NF4
echo ================================================================
echo.
echo  1. Coloca la carpeta del modelo sin cuantizar en:
echo       %BASE_DIR%Fibo2_Dev-Raw
echo  2. Este script escribe:
echo       %BASE_DIR%Fibo2_Dev-NF4
echo.

if not exist "%PYTHON_EXE%" (
    echo [ERROR] No se ha encontrado el entorno virtual / Virtual environment not found:
    echo   %PYTHON_EXE%
    echo.
    pause
    exit /b 1
)

"%PYTHON_EXE%" "%BASE_DIR%tools\fibo2\5_conversor_Fibo2_NF4.py" %*

echo.
echo ================================================================
echo Conversor finalizado / Converter finished.
echo ================================================================
pause
endlocal
