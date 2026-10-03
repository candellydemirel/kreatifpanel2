@echo off
chcp 65001 >nul
setlocal
rem KreatifBot -> Jarvis (jarvis2) kurulumu
set "JARVIS=%~1"
if "%JARVIS%"=="" set "JARVIS=%USERPROFILE%\Desktop\jarvis2"
if not exist "%JARVIS%\main.py" (
  echo jarvis2 klasoru bulunamadi: %JARVIS%
  set /p JARVIS=jarvis2 klasorunun tam yolunu yazin: 
)
set "PY=%JARVIS%\venv\Scripts\python.exe"
if not exist "%PY%" set "PY=python"
echo Kullanilan Python: %PY%
"%PY%" "%~dp0jarvis2_kur.py" "%JARVIS%"
echo.
pause
