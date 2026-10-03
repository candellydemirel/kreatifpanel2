@echo off
REM KreatifBot - Windows .exe derleme
cd /d "%~dp0"
if not exist .venv (
    py -3 -m venv .venv || python -m venv .venv
)
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements-dev.txt || goto :error
python -m pytest -q || goto :error
pyinstaller --noconfirm --clean --onefile --windowed --name KreatifBot ^
  --exclude-module PyQt5 --exclude-module PyQt6 --exclude-module tkinter --exclude-module matplotlib ^
  main.py || goto :error
echo.
echo Derleme tamamlandi: dist\KreatifBot.exe
pause
exit /b 0
:error
echo Derleme BASARISIZ.
pause
exit /b 1
