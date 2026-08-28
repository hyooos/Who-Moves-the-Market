@echo off
setlocal
cd /d "%~dp0"
set "PY=.venv\Scripts\python.exe"

if not exist "%PY%" (
  echo ERROR: Python virtual environment not found: %PY%
  pause
  exit /b 1
)

echo [1/4] Collecting sample news: 2025-06-01 to 2025-06-30
"%PY%" scripts\track2\collect_track2_news.py --start 2025-06-01 --end 2025-06-30 --source google --window-days 15
if errorlevel 1 goto error

echo [2/4] Building news events
"%PY%" scripts\track2\build_track2_news_events.py
if errorlevel 1 goto error

echo [3/4] Ensuring market-price coverage through 2025-10-23
"%PY%" scripts\track2\ensure_news_price_range.py
if errorlevel 1 goto error

echo [4/4] Merging news events into existing SNS results
"%PY%" scripts\track2\refresh_track2_news.py
if errorlevel 1 goto error

echo.
echo DONE. Now run run_streamlit.bat.
pause
exit /b 0

:error
echo.
echo ERROR: News update failed. Check the error message above.
pause
exit /b 1
