@echo off
setlocal
cd /d "%~dp0\..\.."
set "PYTHON_EXE=%CD%\venv\Scripts\python.exe"
if defined LORALAB_PYTHON set "PYTHON_EXE=%LORALAB_PYTHON%"
if not exist "%PYTHON_EXE%" (
  echo [ERROR] Trainer Studio Python environment not found: "%PYTHON_EXE%"
  pause
  exit /b 1
)
"%PYTHON_EXE%" tools\ernie\5_convert_ernie_nf4.py
if errorlevel 1 echo [ERROR] ERNIE NF4 conversion failed.
pause
endlocal
