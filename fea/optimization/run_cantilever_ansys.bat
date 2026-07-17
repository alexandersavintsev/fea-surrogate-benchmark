@echo off
REM ============================================================
REM Запуск ANSYS Mechanical APDL в batch-режиме на макросе консоли.
REM Замените путь ANSYS_EXE на свой (как в run_ansys.bat для Кирша).
REM ============================================================
set "ANSYS_EXE=C:\Program Files\ANSYS Inc\v241\ansys\bin\winx64\MAPDL.exe"
"%ANSYS_EXE%" -b -np 1 -s read -i "cantilever_plate_romai.mac" -o "output_cantilever.txt"
