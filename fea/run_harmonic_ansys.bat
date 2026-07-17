@echo off
REM Гармонические (резонансные) ANSYS-расчёты консоли.
REM Путь к ANSYS находится автоматически. Если нет - раскомментируй и впиши свой:
REM set ANSYS_EXE="C:\Program Files\ANSYS Inc\v241\ansys\bin\winx64\ANSYS241.exe"

cd /d "%~dp0\.."
python fea\run_harmonic_ansys.py %*
pause
