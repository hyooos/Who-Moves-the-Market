@echo off
setlocal
cd /d "%~dp0"
set "PY=.venv\Scripts\python.exe"

if not exist "%PY%" (
  echo ERROR: Python virtual environment not found: %PY%
  pause
  exit /b 1
)

echo Refreshing already-collected news events only...
"%PY%" refresh_track2_news.py
if errorlevel 1 goto error

echo.
echo DONE. Now run run_streamlit.bat.
pause
exit /b 0

:error
echo.
echo ERROR: News refresh failed. Check the error message above.
pause
exit /b 1
