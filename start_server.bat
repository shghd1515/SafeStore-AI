@echo off
chcp 65001 > nul
title SafeStore AI Backend Server

echo ========================================
echo    SafeStore AI - Backend Server Start
echo ========================================
echo.

REM Move to the folder where this batch file is located
cd /d "%~dp0"

REM Conda installation paths
set MINICONDA_DIR=C:\ProgramData\miniconda3
set CONDA_ACTIVATE=%MINICONDA_DIR%\Scripts\activate.bat

REM Check if conda exists
if not exist "%CONDA_ACTIVATE%" (
    echo [ERROR] Conda not found at: %CONDA_ACTIVATE%
    echo Please check your Miniconda installation.
    pause
    exit /b 1
)

echo [1/3] Activating conda environment 'hdc2'...
call "%CONDA_ACTIVATE%" hdc2
if errorlevel 1 (
    echo.
    echo [ERROR] Failed to activate 'hdc2' environment.
    echo Try running this in Anaconda Prompt:  conda env list
    pause
    exit /b 1
)

REM Check main.py exists
if not exist "main.py" (
    echo.
    echo [ERROR] main.py not found in: %CD%
    echo This batch file must be in the safestore_backend folder.
    pause
    exit /b 1
)

echo [2/3] Working directory: %CD%
echo [3/3] Starting server... (Press Ctrl+C to stop)
echo.
echo ----------------------------------------
echo  Dashboard:  http://127.0.0.1:8000
echo  API Docs:   http://127.0.0.1:8000/docs
echo ----------------------------------------
echo.

REM Auto open browser after 5 seconds (background)
start "" /B cmd /C "timeout /t 5 /nobreak > nul && start http://127.0.0.1:8000"

REM Start uvicorn server
uvicorn main:app --reload --port 8000

echo.
echo ========================================
echo    Server stopped
echo ========================================
pause
