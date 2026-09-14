@echo off
setlocal EnableExtensions EnableDelayedExpansion
chcp 65001 >nul
pushd "%~dp0" || exit /b 1

rem Small grid: 225 cases.
set "LENGTH_VALUES=0.1"
set "OUTER_RADIUS_VALUES=0.0286 0.0288 0.029 0.0292 0.0294"
set "PIN_RADIUS_VALUES=0.0125"
set "MOUNT_RADIUS_VALUES=0.0048 0.005 0.0052"
set "THICKNESS_VALUES=0.0166 0.0168 0.0170 0.0172 0.0174 0.0176 0.0178 0.0180 0.0182 0.0184 0.0186 0.0188 0.0190 0.0192 0.0194"
set "MESH_VALUES=0.006"
set "FORCE_VALUES=20000"
set "SHAPE=C3D10M"

set /a TOTAL=225
set /a TARGET=TOTAL
if defined LUG_MAX_RUNS set /a TARGET=LUG_MAX_RUNS
if !TARGET! LSS 1 (
  echo ERROR: LUG_MAX_RUNS must be positive.
  popd & exit /b 2
)
if !TARGET! GTR !TOTAL! set /a TARGET=TOTAL

if not exist "gsi_connecting_lug_parametric_analysis.py" (
  echo ERROR: Parametric Abaqus script is missing.
  popd & exit /b 2
)
if exist "results\" (
  echo ERROR: A results folder already exists; move or remove it first.
  popd & exit /b 2
)
if not exist "connecting_lug\" md "connecting_lug" || (popd & exit /b 2)

set /a attempted=0, succeeded=0, failed=0, next_id=1
:find_id
if exist "connecting_lug\connecting_lug_!next_id!\" (
  set /a next_id+=1
  goto :find_id
)

echo Abaqus connecting-lug batch: !TARGET! of !TOTAL! cases
for %%A in (!LENGTH_VALUES!) do for %%B in (!OUTER_RADIUS_VALUES!) do for %%C in (!PIN_RADIUS_VALUES!) do for %%D in (!MOUNT_RADIUS_VALUES!) do for %%E in (!THICKNESS_VALUES!) do for %%F in (!MESH_VALUES!) do for %%G in (!FORCE_VALUES!) do (
  if !attempted! LSS !TARGET! call :run_one %%A %%B %%C %%D %%E %%F %%G
)
echo Completed: attempted=!attempted! succeeded=!succeeded! failed=!failed!
popd
if !failed! GTR 0 (exit /b 1) else exit /b 0

:run_one
set /a attempted+=1
echo [!attempted!/!TARGET!] L=%1 R=%2 pin=%3 mount=%4 t=%5 mesh=%6 force=%7
set "CONNECTING_LUG_PARAMS=%1,%2,%3,%4,%5,%6,%7,%SHAPE%"
call abaqus cae noGUI=gsi_connecting_lug_parametric_analysis.py
if errorlevel 1 goto :run_failed
for %%R in (connecting_lug_structure.cae connecting_lug_results.odb stress_contour.jpg displacement_contour.jpg) do if not exist "results\%%R" goto :run_failed

set "DEST=connecting_lug\connecting_lug_!next_id!"
md "!DEST!" || exit /b 2
robocopy "results" "!DEST!" /E /COPY:DAT /DCOPY:DAT /R:2 /W:1 /NFL /NDL /NJH /NJS /NP >nul
if errorlevel 8 exit /b 2
(
  echo shank_length=%1
  echo outer_radius=%2
  echo pin_hole_radius=%3
  echo mounting_hole_radius=%4
  echo thickness=%5
  echo mesh_size=%6
  echo force_magnitude=%7
) > "!DEST!\parameters.txt"
set /a succeeded+=1, next_id+=1
:find_next_after_success
if exist "connecting_lug\connecting_lug_!next_id!\" (
  set /a next_id+=1
  goto :find_next_after_success
)
call :cleanup
exit /b 0

:run_failed
echo WARNING: simulation failed; continuing.
set /a failed+=1
call :cleanup
exit /b 0

:cleanup
if exist "%~dp0results\" rd /s /q "%~dp0results"
if exist "%~dp0abaqus.rpy" del /f /q "%~dp0abaqus.rpy"
exit /b 0
