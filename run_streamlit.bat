@echo off
setlocal
cd /d "%~dp0"

echo ========================================
echo Who Moves the Market - Streamlit Launcher
echo ========================================

if exist ".venv\Scripts\python.exe" goto check_packages

echo [1/3] Creating a Python virtual environment...
where py >nul 2>nul
if errorlevel 1 goto try_python
py -3 -m venv ".venv"
goto check_venv

:try_python
where python >nul 2>nul
if errorlevel 1 goto no_python
python -m venv ".venv"

:check_venv
if not exist ".venv\Scripts\python.exe" goto venv_failed

:check_packages
echo [2/3] Checking required packages...
".venv\Scripts\python.exe" -c "import streamlit, pandas, plotly, requests, google.generativeai" >nul 2>nul
if not errorlevel 1 goto launch

echo Installing packages. This can take a few minutes on the first run.
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto install_failed
".venv\Scripts\python.exe" -m pip install -r "requirements-dashboard.txt"
if errorlevel 1 goto install_failed

:launch
echo [3/3] Starting Streamlit...
echo Dashboard URL: http://localhost:8501
echo Keep this window open while using the dashboard.
".venv\Scripts\python.exe" -m streamlit run "dashboard_app.py" --server.headless false
if errorlevel 1 goto run_failed
goto end

:no_python
echo.
echo ERROR: Python was not found.
echo Install Python 3.10 or newer from https://www.python.org/downloads/
echo During installation, select "Add Python to PATH".
goto failed

:venv_failed
echo.
echo ERROR: Could not create the virtual environment.
goto failed

:install_failed
echo.
echo ERROR: Package installation failed.
echo Check your internet connection, then run this file again.
goto failed

:run_failed
echo.
echo ERROR: Streamlit stopped because of an application error.
goto failed

:failed
pause
exit /b 1

:end
endlocal
