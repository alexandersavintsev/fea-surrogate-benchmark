@echo off
REM Запуск серии ANSYS-расчётов упругопластической консоли.
REM Путь к ANSYS обычно находится автоматически (из твоей упругой задачи).
REM Если НЕ нашёлся - раскомментируй строку ниже и впиши свой путь:
REM set ANSYS_EXE="C:\Program Files\ANSYS Inc\v241\ansys\bin\winx64\ANSYS241.exe"

cd /d "%~dp0\.."
python fea\run_plastic_ansys.py %*
pause
