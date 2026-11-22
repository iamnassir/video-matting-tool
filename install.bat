@echo off
rem One-time setup on Windows. Creates the Python environment in this folder (.venv), installs
rem PyTorch (the NVIDIA build when an NVIDIA graphics card is present, otherwise the processor
rem build), the application's packages and the weights of the default model.
rem Afterwards: double-click start_web.bat.  Running it again is safe: finished steps are skipped.
rem
rem   install.bat                normal use (double-click)
rem   install.bat --no-pause     for scripts: do not wait for a key at the end
rem   set VMT_TORCH=cpu          force the processor build of PyTorch (or: cuda)
rem   set VMT_PYTHON=C:\path\to\python.exe     use this Python (3.10, 3.11 or 3.12)
setlocal EnableExtensions
title Video Matting Tool - setup
cd /d "%~dp0"
set "PAUSE_AT_END=1"
if /i "%~1"=="--no-pause" set "PAUSE_AT_END=0"

echo.
echo  Video Matting Tool - setup
echo  ==========================
echo.

rem ------------------------------------------------------------------ Python 3.10 - 3.12
set "PY="
if defined VMT_PYTHON call :try_python "%VMT_PYTHON%"
if not defined PY call :try_python py -3.10
if not defined PY call :try_python py -3.11
if not defined PY call :try_python py -3.12
if not defined PY call :try_python python
if defined PY goto :python_ok
echo  [!] Python 3.10, 3.11 or 3.12 was not found on this computer.
echo      Install it with the command below, close this window, then run install.bat again:
echo.
echo          winget install Python.Python.3.10
echo.
goto :failed
:python_ok
echo  Python: %PY%

rem ------------------------------------------------------------------ FFmpeg
if defined VMT_FFMPEG goto :ffmpeg_ok
where ffmpeg >nul 2>&1
if errorlevel 1 goto :ffmpeg_missing
where ffprobe >nul 2>&1
if errorlevel 1 goto :ffmpeg_missing
goto :ffmpeg_ok
:ffmpeg_missing
echo  [!] FFmpeg was not found. The tool needs the programs ffmpeg and ffprobe, version 7 or newer.
echo      Install it with the command below, close this window, then run install.bat again:
echo.
echo          winget install Gyan.FFmpeg
echo.
goto :failed
:ffmpeg_ok

rem ------------------------------------------------------------------ environment
if exist ".venv\Scripts\python.exe" goto :venv_ok
echo.
echo  [1/5] Creating the Python environment in the folder .venv ...
%PY% -m venv .venv
if errorlevel 1 goto :failed
:venv_ok
set "VPY=%CD%\.venv\Scripts\python.exe"
set "PIP_DISABLE_PIP_VERSION_CHECK=1"
"%VPY%" -m pip install --upgrade pip --no-cache-dir -q

rem ------------------------------------------------------------------ PyTorch
"%VPY%" -c "import torch, torchvision" >nul 2>&1
if not errorlevel 1 goto :torch_ok
set "TORCH_KIND=cpu"
nvidia-smi >nul 2>&1
if not errorlevel 1 set "TORCH_KIND=cuda"
if /i "%VMT_TORCH%"=="cpu" set "TORCH_KIND=cpu"
if /i "%VMT_TORCH%"=="cuda" set "TORCH_KIND=cuda"
echo.
if "%TORCH_KIND%"=="cuda" goto :torch_cuda
echo  [2/5] No NVIDIA graphics card found: installing PyTorch for the processor, about 200 MB ...
echo        The tool works without a graphics card, but slowly.
"%VPY%" -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu --no-cache-dir
if errorlevel 1 goto :failed
goto :torch_ok
:torch_cuda
echo  [2/5] NVIDIA graphics card found: installing PyTorch with CUDA 12.1, about 2.5 GB. This takes a while ...
"%VPY%" -m pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 --extra-index-url https://download.pytorch.org/whl/cu121 --no-cache-dir
if errorlevel 1 goto :failed
:torch_ok

rem ------------------------------------------------------------------ application packages
echo.
echo  [3/5] Installing the application's packages ...
"%VPY%" -m pip install -r requirements.txt -c constraints.txt --no-cache-dir
if errorlevel 1 goto :failed

rem ------------------------------------------------------------------ default model
echo.
echo  [4/5] Downloading the default model, Robust Video Matting, about 120 MB, from its official source ...
"%VPY%" main.py models download rvm
if errorlevel 1 goto :failed

rem ------------------------------------------------------------------ check
echo.
echo  [5/5] Checking the installation ...
"%VPY%" main.py system
if errorlevel 1 goto :failed

echo.
echo  Setup finished. Double-click start_web.bat to open the tool in your browser.
echo  More models, for example for objects or for the best hair detail: see docs\INSTALL.md, section 3.
echo.
if "%PAUSE_AT_END%"=="1" pause
exit /b 0

:failed
echo.
echo  Setup did not finish. The message above says what is missing or what went wrong.
echo.
if "%PAUSE_AT_END%"=="1" pause
exit /b 1

:try_python
%* -c "import sys; raise SystemExit(0 if (3, 10) <= sys.version_info[:2] <= (3, 12) else 1)" >nul 2>&1
if not errorlevel 1 set PY=%*
exit /b 0
