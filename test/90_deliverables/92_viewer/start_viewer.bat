@echo off
rem Start the SiClink spike-glyph viewer and open it in the default browser.
cd /d "%~dp0..\.."
set "URL=http://127.0.0.1:8765/"

powershell -NoProfile -Command "try { Invoke-WebRequest '%URL%' -UseBasicParsing -TimeoutSec 2 | Out-Null; exit 0 } catch { exit 1 }"
if %errorlevel%==0 (
    echo Viewer already running, opening browser.
    start "" "%URL%"
    exit /b 0
)

if not exist ".venv\Scripts\python.exe" (
    echo Missing .venv\Scripts\python.exe
    pause
    exit /b 1
)

rem Open the browser when the server responds. First start builds the OCF cache, about 1 minute.
start "" /b powershell -NoProfile -WindowStyle Hidden -Command "for ($i = 0; $i -lt 600; $i++) { try { Invoke-WebRequest '%URL%' -UseBasicParsing -TimeoutSec 2 | Out-Null; Start-Process '%URL%'; break } catch { Start-Sleep 1 } }"

echo Starting viewer at %URL% . Close this window or press Ctrl+C to stop.
".venv\Scripts\python.exe" -B test\92_viewer\server.py
pause
