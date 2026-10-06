@echo off
REM Starts ProductPulse AI and opens it at http://localhost:8501 (the link never changes).
REM If the app is already running, it just opens the browser.
cd /d "%~dp0"

powershell -NoProfile -Command "try { Invoke-WebRequest -UseBasicParsing http://localhost:8501/_stcore/health -TimeoutSec 2 | Out-Null; exit 0 } catch { exit 1 }"
if %errorlevel%==0 (
    start "" http://localhost:8501
    exit /b
)

if not exist ".venv\Scripts\python.exe" (
    echo The .venv folder is missing. Set it up first - see "Installation" in README.md.
    pause
    exit /b 1
)

title ProductPulse AI
echo Starting ProductPulse AI at http://localhost:8501 ...
echo Keep this window open while you use the app. Close it to stop the app.
".venv\Scripts\python.exe" -m streamlit run app.py --server.port 8501 --server.headless false
pause
