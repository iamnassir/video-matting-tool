@echo off
rem Starts the web interface and opens it in the browser. Keep this window open while you use it;
rem close it (or press Ctrl+C) to stop the server. Extra options are passed on, e.g. --port 9000.
title Video Matting Tool
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo The Python environment .venv was not found in this folder. See docs\INSTALL.md.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" main.py serve %*
echo.
echo The server has stopped.
pause
