@echo off
setlocal
cd /d "%~dp0"
set "PY=.venv\Scripts\python.exe"

if not exist "%PY%" (
  echo ERROR: Python virtual environment not found: %PY%
  pause
  exit /b 1
)

echo [1/4] Collecting full-period news: 2023-01-03 to 2025-10-23
"%PY%" collect_track2_news.py --start 2023-01-03 --end 2025-10-23 --source google --window-days 31
if errorlevel 1 goto error

echo [2/4] Building news events
"%PY%" build_track2_news_events.py
if errorlevel 1 goto error

echo [3/4] Ensuring market-price coverage through 2025-10-23
"%PY%" ensure_news_price_range.py
if errorlevel 1 goto error

echo [4/4] Merging news events into existing SNS results
"%PY%" refresh_track2_news.py
if errorlevel 1 goto error

echo.
echo DONE. Now run run_streamlit.bat.
pause
exit /b 0

:error
echo.
echo ERROR: Full news update failed. Check the error message above.
pause
exit /b 1
