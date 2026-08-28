@echo off
setlocal
cd /d "%~dp0"
set "PY=..\..\.venv\Scripts\python.exe"

if not exist "%PY%" (
  echo ERROR: Python virtual environment not found: %PY%
  pause
  exit /b 1
)

echo [1/2] Checking and extending market-price range to 2025-10-23
"%PY%" ensure_news_price_range.py
if errorlevel 1 goto error

echo [2/2] Re-attaching news events to trading dates
"%PY%" refresh_track2_news.py
if errorlevel 1 goto error

echo.
echo DONE. Now run run_streamlit.bat from the repository root.
pause
exit /b 0

:error
echo.
echo ERROR: News date fix failed. Check the message above.
pause
exit /b 1
