@echo off
setlocal EnableExtensions
title AcademiaSD - Krea-2 NF4 Converter

rem El conversor usa rutas relativas a la raiz del LoRAlab (venv, Models\Krea-2-Raw, Models\Krea-2-NF4).
cd /d "%~dp0..\.."
call venv\Scripts\activate.bat

echo ====================================================
echo   ACADEMIASD - KREA-2 NF4 CONVERTER
echo   .\Models\Krea-2-Raw  -^>  .\Models\Krea-2-NF4
echo ====================================================
echo.
python tools\krea2\5_conversor_Krea2_NF4.py
pause
endlocal
