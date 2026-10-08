@echo off
setlocal EnableExtensions EnableDelayedExpansion

title Instalador Venv AcademiaSD LoRAlab Trainer Studio - AMD / AcademiaSD LoRAlab Trainer Studio Venv Installer - AMD (experimental)

:: ========================================================
:: CONFIGURACION / CONFIGURATION
:: ========================================================

set "BASE_DIR=%~dp0"
set "PYTHON_INSTALLER=%BASE_DIR%python-3.13.1-amd64.exe"
set "PYTHON_EXE="
set "GIT_INSTALLER=%BASE_DIR%Git-64-bit-installer.exe"

echo ========================================================
echo   INSTALADOR ACADEMIASD LORALAB TRAINER STUDIO / ACADEMIASD LORALAB TRAINER STUDIO INSTALLER
echo   Un solo entorno para / One environment for: LTX-2.3, Krea 2, MiniMax-H3, Qwen-Image 2.1, Z-Image, Anima, FLUX.2 Klein 9B, Ideogram 4, SDXL
echo   Entorno Python 3.13.1 + PyTorch ROCm / Python 3.13.1 + PyTorch ROCm Env (AMD, experimental)
echo   Solo GPUs AMD Radeon RX 7000 / RX 9000 / Strix Halo en Windows / AMD only, Windows
echo ========================================================
echo.
echo Carpeta del instalador / Installer folder:
echo %BASE_DIR%
echo.

echo [1/9] Comprobando Git / Checking Git...

:: Git solo lo usa Update_LoRAlab-TrainerStudio.bat: si falla, se avisa y la instalacion sigue.
:: Git is only used by Update_LoRAlab-TrainerStudio.bat: if it fails, warn and keep installing.
where git >nul 2>&1
if not errorlevel 1 goto GIT_OK
if exist "%ProgramFiles%\Git\cmd\git.exe" (
    set "PATH=%ProgramFiles%\Git\cmd;%PATH%"
    goto GIT_OK
)

echo Git no esta instalado. Se descargara e instalara (Windows pedira permiso).
echo Git is not installed. It will be downloaded and installed (Windows will ask for permission).
echo.
powershell -NoProfile -Command "$ProgressPreference='SilentlyContinue'; [Net.ServicePointManager]::SecurityProtocol=[Net.SecurityProtocolType]::Tls12; $r=Invoke-RestMethod 'https://api.github.com/repos/git-for-windows/git/releases/latest'; $a=$r.assets | Where-Object { $_.name -match '^Git-[0-9.]+-64-bit\.exe$' } | Select-Object -First 1; Invoke-WebRequest -Uri $a.browser_download_url -OutFile '%GIT_INSTALLER%'"
if not exist "%GIT_INSTALLER%" goto GIT_FAILED

echo Instalando Git... / Installing Git...
"%GIT_INSTALLER%" /VERYSILENT /NORESTART /NOCANCEL /SP- /SUPPRESSMSGBOXES
del "%GIT_INSTALLER%" >nul 2>&1
if exist "%ProgramFiles%\Git\cmd\git.exe" (
    set "PATH=%ProgramFiles%\Git\cmd;%PATH%"
    goto GIT_OK
)

:GIT_FAILED
echo.
echo [AVISO] No se pudo instalar Git. El entrenador funciona igual, pero para actualizar
echo         instalalo a mano desde https://git-scm.com/download/win
echo [WARNING] Git could not be installed. The trainer works anyway, but to update it
echo           install Git by hand from https://git-scm.com/download/win
echo.
goto GIT_DONE

:GIT_OK
echo [OK] Git disponible / Git available.
git --version

:GIT_DONE
echo.

echo [2/9] Comprobando Python 3.13 / Checking Python 3.13...

where python >nul 2>&1
if errorlevel 1 goto FIND_LOCAL_PYTHON

echo Python encontrado en PATH / Python found in PATH:
python --version
echo.
echo Comprobando version compatible / Checking compatible version...

python -c "import sys; exit(0 if sys.version_info[:2] == (3,13) else 1)" >nul 2>&1
if errorlevel 1 goto NOT_313_IN_PATH

echo [OK] Python 3.13 detectado / Python 3.13 detected.
set "PYTHON_EXE=python"
goto PYTHON_OK

:NOT_313_IN_PATH
echo [ADVERTENCIA/WARNING] Python encontrado pero no es 3.13 / Python found but it is not 3.13.

:: --------------------------------------------------------
:: Buscar Python 3.13 en ubicaciones habituales / Search Python 3.13 in common paths
:: --------------------------------------------------------

:FIND_LOCAL_PYTHON
echo.
echo Buscando instalaciones existentes de Python 3.13... / Searching for existing Python 3.13 installations...

if exist "%LocalAppData%\Programs\Python\Python313\python.exe" (
    set "PYTHON_EXE=%LocalAppData%\Programs\Python\Python313\python.exe"
    echo [OK] Encontrado / Found:
    echo %PYTHON_EXE%
    goto PYTHON_OK
)

if exist "%ProgramFiles%\Python313\python.exe" (
    set "PYTHON_EXE=%ProgramFiles%\Python313\python.exe"
    echo [OK] Encontrado / Found:
    echo %PYTHON_EXE%
    goto PYTHON_OK
)

if exist "%ProgramFiles(x86)%\Python313\python.exe" (
    set "PYTHON_EXE=%ProgramFiles(x86)%\Python313\python.exe"
    echo [OK] Encontrado / Found:
    echo %PYTHON_EXE%
    goto PYTHON_OK
)

:: ========================================================
:: INSTALAR PYTHON 3.13.1 / INSTALL PYTHON 3.13.1
:: ========================================================

echo.
echo [INFO] Python 3.13 no esta disponible en el sistema / Python 3.13 is not available on the system.
echo.

if exist "%PYTHON_INSTALLER%" goto DO_INSTALL

echo ========================================================
echo   ANALIZANDO E INICIANDO DESCARGA AUTOMATICA / ANALYZING AND STARTING AUTO DOWNLOAD
echo ========================================================
echo.
echo Analizando arquitectura del sistema... / Analyzing system architecture...

rem PyTorch requiere un sistema operativo de 64 bits (AMD64 / x64)
set "ARCH=amd64"
if "%PROCESSOR_ARCHITECTURE%"=="x86" (
    if not defined PROCESSOR_ARCHITEW6432 (
        echo [ERROR] Sistema de 32 bits detectado / 32-bit system detected.
        echo PyTorch y CUDA requieren obligatoriamente 64 bits / PyTorch and CUDA strictly require 64 bits.
        pause
        exit /b 1
    )
)

echo Arquitectura compatible detectada: 64 bits %PROCESSOR_ARCHITECTURE% / Compatible architecture detected: 64 bits %PROCESSOR_ARCHITECTURE%
echo Descargando Python 3.13.1 x64 desde el sitio oficial... / Downloading Python 3.13.1 x64 from official site...
echo.

rem Intento de descarga 1: curl
curl -L "https://www.python.org/ftp/python/3.13.1/python-3.13.1-amd64.exe" -o "%PYTHON_INSTALLER%"

if exist "%PYTHON_INSTALLER%" goto DOWNLOAD_OK

rem Intento de descarga 2: PowerShell
echo [INFO] curl fallo. Intentando con PowerShell... / curl failed. Trying with PowerShell...
powershell -Command "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri 'https://www.python.org/ftp/python/3.13.1/python-3.13.1-amd64.exe' -OutFile '%PYTHON_INSTALLER%'" >nul 2>&1

if not exist "%PYTHON_INSTALLER%" (
    echo ========================================================
    echo [ERROR] No se pudo descargar Python / Could not download Python.
    echo ========================================================
    echo No ha sido posible descargar automaticamente / Automatic download failed.
    echo Descarguelo manualmente desde su navegador / Download manually from browser:
    echo https://www.python.org/ftp/python/3.13.1/python-3.13.1-amd64.exe
    echo Guardelo como "python-3.13.1-amd64.exe" junto a este archivo BAT / Save it as "python-3.13.1-amd64.exe" next to this BAT file.
    echo.
    pause
    exit /b 1
)

:DOWNLOAD_OK
echo [OK] Descargado correctamente / Successfully downloaded: %PYTHON_INSTALLER%
echo.

:DO_INSTALL
echo ========================================================
echo   INSTALANDO PYTHON 3.13.1 / INSTALLING PYTHON 3.13.1
echo ========================================================
echo.
echo El instalador se ejecutara de fondo / Installer will run in the background.
echo Se instalara para el usuario actual anadiendose al PATH / Installing for current user and adding to PATH.
echo Esto puede tardar unos minutos. Espere... / This may take a few minutes. Please wait...
echo.

"%PYTHON_INSTALLER%" /quiet InstallAllUsers=0 PrependPath=1 Include_pip=1 Include_launcher=1 Include_test=0

if errorlevel 1 (
    echo.
    echo [ERROR] No se pudo instalar Python 3.13.1 / Could not install Python 3.13.1.
    echo.
    pause
    exit /b 1
)

echo [OK] Instalacion de Python finalizada / Python installation finished.
echo.

:: --------------------------------------------------------
:: Buscar Python instalado despues de la instalacion
:: --------------------------------------------------------

if exist "%LocalAppData%\Programs\Python\Python313\python.exe" (
    set "PYTHON_EXE=%LocalAppData%\Programs\Python\Python313\python.exe"
    goto PYTHON_OK
)

if exist "%ProgramFiles%\Python313\python.exe" (
    set "PYTHON_EXE=%ProgramFiles%\Python313\python.exe"
    goto PYTHON_OK
)

for /f "delims=" %%A in ('where python 2^>nul') do (
    set "PYTHON_EXE=%%A"
    goto PYTHON_OK
)

echo.
echo [ERROR] Python instalado pero no localizado / Python installed but not located.
echo Reinicie Windows y vuelva a ejecutar / Restart Windows and run again.
echo.
pause
exit /b 1

:: ========================================================
:: PYTHON OK
:: ========================================================

:PYTHON_OK

echo.
echo ========================================================
echo   PYTHON DETECTADO / PYTHON DETECTED
echo ========================================================
echo.
echo Ejecutable / Executable:
echo %PYTHON_EXE%
echo.

"%PYTHON_EXE%" --version

if errorlevel 1 (
    echo.
    echo [ERROR] Python no se ejecuta correctamente / Python cannot run properly.
    pause
    exit /b 1
)

"%PYTHON_EXE%" -c "import sys; exit(0 if sys.version_info[:2] == (3,13) else 1)" >nul 2>&1

if errorlevel 1 (
    echo.
    echo [ERROR] La version de Python no es compatible / Python version is not compatible.
    echo Se requiere Python 3.13.x / Python 3.13.x is required.
    echo.
    pause
    exit /b 1
)

echo [OK] Python 3.13 compatible detectado / Compatible Python 3.13 detected.

:: ========================================================
:: 2/8 - COMPROBAR VENV / CHECK VENV
:: ========================================================

echo.
echo [3/9] Comprobando modulo venv... / Checking venv module...

"%PYTHON_EXE%" -m venv --help >nul 2>&1

if errorlevel 1 (
    echo.
    echo [ERROR] El modulo venv de Python no esta disponible / Python venv module is not available.
    echo.
    pause
    exit /b 1
)

echo [OK] Modulo venv disponible / venv module available.

:: ========================================================
:: 3/8 - COMPROBAR GPU NVIDIA / CHECK NVIDIA GPU
:: ========================================================

echo.
echo [4/9] Comprobando GPU AMD... / Checking AMD GPU...
echo La deteccion de la GPU se realizara despues de instalar PyTorch / GPU detection will occur after PyTorch installation.
echo.
echo [OK] Continuando con la instalacion / Continuing installation.

echo.
echo [5/9] Preparando entorno virtual limpio... / Preparing clean virtual environment...

if exist "%BASE_DIR%venv" (
    echo.
    echo Entorno virtual anterior detectado. Eliminando... / Previous virtual environment detected. Removing...

    rmdir /s /q "%BASE_DIR%venv"

    if exist "%BASE_DIR%venv" (
        echo.
        echo [ERROR] No se pudo eliminar el venv anterior / Could not remove previous venv.
        echo Cierra cualquier programa en uso / Close any running program:
        echo     venv\Scripts\python.exe
        echo.
        pause
        exit /b 1
    )
)

echo.
echo Creando nuevo entorno virtual... / Creating new virtual environment...

"%PYTHON_EXE%" -m venv "%BASE_DIR%venv"

if errorlevel 1 (
    echo.
    echo [ERROR] No se pudo crear el entorno virtual / Could not create virtual environment.
    pause
    exit /b 1
)

:: ========================================================
:: 5/8 - ACTIVAR VENV / ACTIVATE VENV
:: ========================================================

echo.
echo [6/9] Activando entorno virtual... / Activating virtual environment...

call "%BASE_DIR%venv\Scripts\activate.bat"

if errorlevel 1 (
    echo.
    echo [ERROR] No se pudo activar el entorno virtual / Could not activate virtual environment.
    pause
    exit /b 1
)

set "VENV_PYTHON=%BASE_DIR%venv\Scripts\python.exe"

echo.
echo Python del entorno virtual / Virtual environment Python:
echo %VENV_PYTHON%

"%VENV_PYTHON%" --version

:: ========================================================
:: 6/8 - ACTUALIZAR HERRAMIENTAS / UPGRADE TOOLS
:: ========================================================

echo.
echo [7/9] Actualizando pip, setuptools y wheel... / Upgrading pip, setuptools, and wheel...

"%VENV_PYTHON%" -m pip install --upgrade pip setuptools wheel

if errorlevel 1 (
    echo.
    echo [ERROR] No se pudieron actualizar las herramientas / Could not upgrade Python tools.
    pause
    exit /b 1
)

:: ========================================================
:: 7/8 - INSTALAR PYTORCH / INSTALL PYTORCH
:: ========================================================

echo.
echo [8/9] Instalando PyTorch ROCm para AMD... / Installing PyTorch ROCm for AMD...
echo.
echo ========================================================
echo   AMD (EXPERIMENTAL) - ROCm TheRock nightly
echo ========================================================
echo.
echo Elige la familia de tu GPU / Choose your GPU family:
echo.
echo   1 = RX 7000 (RX 7600 ... RX 7900 XTX)      gfx110X-all
echo   2 = RX 9000 (RX 9060 XT / RX 9070 XT)      gfx120X-all
echo   3 = Ryzen AI Max (Strix Halo iGPU)         gfx1151
echo.
echo Otras GPUs AMD no estan soportadas por este instalador.
echo Other AMD GPUs are not supported by this installer.
echo Guia / Guide: https://github.com/CS1o/Stable-Diffusion-Info/wiki/Lora-Trainer-Setup-Guides
echo.
set "AMD_GFX="
choice /c 123 /n /m "Opcion / Option [1-3]: "
if errorlevel 3 (set "AMD_GFX=gfx1151") else if errorlevel 2 (set "AMD_GFX=gfx120X-all") else (set "AMD_GFX=gfx110X-all")
echo.
echo Se instalara PyTorch ROCm (%AMD_GFX%) / PyTorch ROCm (%AMD_GFX%) will be installed.
echo La instalacion puede tardar minutos / Installation may take minutes.
echo ========================================================
echo.

"%VENV_PYTHON%" -m pip uninstall torch torchvision torchaudio rocm rocm-sdk-core rocm-sdk-devel rocm-sdk-libraries-custom -y >nul 2>&1
"%VENV_PYTHON%" -m pip install --index-url https://rocm.nightlies.amd.com/v2/%AMD_GFX%/ torch torchvision torchaudio

if errorlevel 1 (
    echo.
    echo [ERROR] No se pudo instalar PyTorch / Could not install PyTorch.
    pause
    exit /b 1
)

:: ========================================================
:: INSTALAR DEPENDENCIAS / INSTALL DEPENDENCIES
:: ========================================================

echo.
echo Instalando Diffusers, Transformers, PEFT, BitsAndBytes y utilidades (requirements.txt)...
echo Installing Diffusers, Transformers, PEFT, BitsAndBytes and utilities (requirements.txt)...

set "AMD_REQ=%TEMP%\loralab_requirements_amd.txt"
findstr /v /i /r /c:"^torch" /c:"download.pytorch.org" "%BASE_DIR%requirements.txt" > "%AMD_REQ%"
"%VENV_PYTHON%" -m pip install -r "%AMD_REQ%"

if errorlevel 1 (
    echo.
    echo [ERROR] Error instalando dependencias / Error installing dependencies.
    pause
    exit /b 1
)

:: ========================================================
:: 8/8 - COMPROBACION FINAL / FINAL CHECK
:: ========================================================

echo.
echo [9/9] Verificando instalacion completa... / Verifying installation...
echo.

:: --------------------------------------------------------
:: PYTORCH
:: --------------------------------------------------------

echo ========================================================
echo PyTorch
echo ========================================================

"%VENV_PYTHON%" -c "import torch; print('PyTorch:', torch.__version__); print('ROCm/HIP:', torch.version.hip); print('CUDA disponible:', torch.cuda.is_available()); print('GPUs detectadas:', torch.cuda.device_count()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NINGUNA')"
if errorlevel 1 (
    echo.
    echo [ERROR] PyTorch no se inicializo correctamente / PyTorch failed to initialize.
    pause
    exit /b 1
)

:: --------------------------------------------------------
:: DIFFUSERS
:: --------------------------------------------------------

echo.
echo ========================================================
echo Diffusers
echo ========================================================

"%VENV_PYTHON%" -c "import diffusers; print('Diffusers:', diffusers.__version__); from diffusers import LTX2VideoTransformer3DModel; print('LTX-2.3: OK'); from diffusers import Krea2Transformer2DModel, AutoencoderKLQwenImage; print('Krea 2: OK'); from diffusers import AutoencoderKLMiniMaxH3, MiniMaxH3Scheduler; from diffusers.models.transformers import transformer_minimax_h3; print('MiniMax-H3: OK'); from diffusers import QwenImage21Pipeline, QwenImage21Transformer2DModel, AutoencoderKLQwenImage21; print('Qwen-Image 2.1: OK'); from diffusers import ZImagePipeline, ZImageTransformer2DModel; print('Z-Image: OK'); from diffusers import AnimaModularPipeline, AnimaTextConditioner, CosmosTransformer3DModel; print('Anima: OK'); from diffusers import Flux2KleinPipeline, Flux2Transformer2DModel, AutoencoderKLFlux2; print('FLUX.2 Klein 9B: OK'); from diffusers import Ideogram4Pipeline, Ideogram4Transformer2DModel; print('Ideogram 4: OK'); from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, AutoencoderKL; print('SDXL: OK')"

if errorlevel 1 (
    echo.
    echo [ERROR] Diffusers no incluye todos los modelos / Diffusers does not include every model.
    pause
    exit /b 1
)

echo [OK] Diffusers cargado correctamente / Diffusers loaded successfully.

:: --------------------------------------------------------
:: HUGGING FACE HUB
:: --------------------------------------------------------

echo.
echo ========================================================
echo Hugging Face Hub
echo ========================================================

"%VENV_PYTHON%" -c "import huggingface_hub; print('Hugging Face Hub:', huggingface_hub.__version__)"

if errorlevel 1 (
    echo.
    echo [ERROR] Hugging Face Hub no esta instalado correctamente / Hugging Face Hub is not installed properly.
    pause
    exit /b 1
)

echo [OK] Hugging Face Hub cargado correctamente / Hugging Face Hub loaded successfully.

:: --------------------------------------------------------
:: FLASK / PSUTIL (lanzador e interfaces web / launcher and web UIs)
:: --------------------------------------------------------

echo.
echo ========================================================
echo Flask / psutil
echo ========================================================

"%VENV_PYTHON%" -c "import flask, psutil; from importlib.metadata import version; print('Flask:', version('flask'), '| psutil:', psutil.__version__)"

if errorlevel 1 (
    echo.
    echo [ERROR] Flask o psutil no estan instalados / Flask or psutil is not installed.
    pause
    exit /b 1
)

echo [OK] Flask y psutil cargados correctamente / Flask and psutil loaded successfully.

:: --------------------------------------------------------
:: TRANSFORMERS
:: --------------------------------------------------------

echo.
echo ========================================================
echo Transformers
echo ========================================================

"%VENV_PYTHON%" -c "import transformers; from packaging.version import Version; print('Transformers:', transformers.__version__); from transformers import Qwen3VLForConditionalGeneration, Gemma3ForConditionalGeneration; exit(0 if Version(transformers.__version__) >= Version('5.17') else 1)"

if errorlevel 1 (
    echo.
    echo [ERROR] Se requiere Transformers 5.17+ con Qwen3-VL y Gemma 3 / Transformers 5.17+ with Qwen3-VL and Gemma 3 is required.
    pause
    exit /b 1
)

echo [OK] Transformers cargado correctamente / Transformers loaded successfully.

:: --------------------------------------------------------
:: PEFT
:: --------------------------------------------------------

echo.
echo ========================================================
echo PEFT
echo ========================================================

"%VENV_PYTHON%" -c "import peft; print('PEFT:', peft.__version__)"

if errorlevel 1 (
    echo.
    echo [ERROR] PEFT no esta instalado correctamente / PEFT is not installed properly.
    pause
    exit /b 1
)

echo [OK] PEFT cargado correctamente / PEFT loaded successfully.

:: --------------------------------------------------------
:: ACCELERATE
:: --------------------------------------------------------

echo.
echo ========================================================
echo Accelerate
echo ========================================================

"%VENV_PYTHON%" -c "import accelerate; print('Accelerate:', accelerate.__version__)"

if errorlevel 1 (
    echo.
    echo [ERROR] Accelerate no esta instalado correctamente / Accelerate is not installed properly.
    pause
    exit /b 1
)

echo [OK] Accelerate cargado correctamente / Accelerate loaded successfully.

:: --------------------------------------------------------
:: BITSANDBYTES
:: --------------------------------------------------------

echo.
echo ========================================================
echo BitsAndBytes
echo ========================================================

"%VENV_PYTHON%" -c "import bitsandbytes as bnb; print('BitsAndBytes:', bnb.__version__)"

if errorlevel 1 (
    echo.
    echo [ERROR] BitsAndBytes no se inicializo / BitsAndBytes failed to initialize.
    echo Es obligatorio para los modelos NF4 / It is required for the NF4 models.
    pause
    exit /b 1
)

echo [OK] BitsAndBytes cargado correctamente / BitsAndBytes loaded successfully.

:: --------------------------------------------------------
:: COMPROBAR CUDA / CHECK CUDA
:: --------------------------------------------------------

echo.
echo ========================================================
echo   RESULTADO DE LA COMPROBACION / CHECK RESULT
echo ========================================================
echo.

"%VENV_PYTHON%" -c "import torch; exit(0 if torch.cuda.is_available() else 1)"

if errorlevel 1 (
    echo [ADVERTENCIA/WARNING] PyTorch NO detecta la GPU AMD / PyTorch does NOT detect the AMD GPU.
    echo.
    echo Posibles causas / Possible causes:
    echo - Familia de GPU equivocada / Wrong GPU family: vuelve a ejecutar este instalador / run this installer again.
    echo - Driver AMD Adrenalin antiguo / Outdated AMD Adrenalin driver.
    echo - GPU no soportada por ROCm / GPU not supported by ROCm.
    echo.
) else (
    echo [OK] PyTorch detecta correctamente la GPU AMD / PyTorch correctly detects the AMD GPU.
)

:: ========================================================
:: FINAL
:: ========================================================

echo.
echo ========================================================
echo   INSTALACION COMPLETADA / INSTALLATION COMPLETED
echo ========================================================
echo.
echo El entorno virtual "venv" ha sido creado / Virtual environment "venv" created.
echo.
echo Python utilizado / Python used:
echo %PYTHON_EXE%
echo.
echo Python del entorno virtual / Virtual env Python:
echo %VENV_PYTHON%
echo.
echo Version de Python / Python version:
"%VENV_PYTHON%" --version
echo.
echo GPU detectada por PyTorch / GPU detected by PyTorch:
"%VENV_PYTHON%" -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO DETECTADA / NOT DETECTED')"
echo.
echo Version ROCm de PyTorch / PyTorch ROCm version:
"%VENV_PYTHON%" -c "import torch; print(torch.version.hip)"
echo.
echo Version Diffusers / Diffusers version:
"%VENV_PYTHON%" -c "import diffusers; print(diffusers.__version__)"
echo.
echo Version Transformers / Transformers version:
"%VENV_PYTHON%" -c "import transformers; print(transformers.__version__)"
echo.
echo Version Hugging Face Hub / Hugging Face Hub version:
"%VENV_PYTHON%" -c "import huggingface_hub; print(huggingface_hub.__version__)"
echo.
echo ========================================================
echo.
echo El entorno esta listo. Abre los entrenadores con Start_LoRAlab-TrainerStudio.bat
echo Environment is ready. Open the trainers with Start_LoRAlab-TrainerStudio.bat
echo.
echo [AMD] No ejecutes "Install_Triton&SageAtten220.bat": es solo para NVIDIA.
echo [AMD] Do not run "Install_Triton&SageAtten220.bat": it is NVIDIA only.
echo.
pause

endlocal