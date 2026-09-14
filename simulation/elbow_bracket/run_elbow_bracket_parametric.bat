@echo off
setlocal EnableExtensions EnableDelayedExpansion
chcp 65001 >nul

rem ============================================================
rem Abaqus perforated elbow-bracket batch parametric analysis
rem
rem Parameter order:
rem   1. horizontal length
rem   2. vertical length
rem   3. bend radius
rem   4. arm width
rem   5. hole radius
rem   6. thickness
rem   7. mesh size
rem   8. total force magnitude
rem   9. force angle in degrees
rem
rem Combination counts are calculated from the parameter lists below.
rem ============================================================

rem ------------------------------------------------------------
rem Always run relative to the directory containing this script.
rem ------------------------------------------------------------
pushd "%~dp0" || (
    echo ERROR: Cannot enter the script directory.
    exit /b 1
)

rem ============================================================
rem Parameter values: SI units (m, N, degrees), linear elastic steel.
rem ============================================================

rem Both straight lengths must be at least 2.5 * the largest arm width.
rem Minimum required length is 0.150 m; use 0.160 m or greater.
set "HORIZONTAL_VALUES=0.16 0.18 0.20"
set "VERTICAL_VALUES=0.18 0.20 0.22"

rem Centerline bend radius; minimum inner radius is 0.030 m.
set "BEND_RADIUS_VALUES=0.06 0.07"
set "ARM_WIDTH_VALUES=0.05 0.06"

rem Minimum side ligament beside a hole is 0.015 m.
set "HOLE_RADIUS_VALUES=0.009 0.010 0.011 0.012"
set "THICKNESS_VALUES=0.018 0.020"

rem Fixed 4 mm seed for comparable samples: minimum thickness/seed = 4.5.
rem Initial mesh only; hole stresses still require a convergence study.
set "MESH_SIZE_VALUES=0.008"

rem Total end force in N, distributed over the free-end nodes by Python.
rem Conservative exploratory loads for the elastic-only material model.
rem Gross-section root bending estimate is below 80 MPa for this grid;
rem this is not a bound on local hole/fillet/clamp stresses or a yield check.
set "FORCE_MAGNITUDE_VALUES=2000"

rem Angle from global +X toward +Y; negative angles load right/down.
set "FORCE_ANGLE_VALUES=-45"

rem ============================================================
rem Count actual levels; full Cartesian grid by default (1944 runs).
rem ELBOW_MAX_RUNS remains available as an explicit short-run override.
rem ============================================================
set /a "TOTAL=1"
for %%P in (HORIZONTAL VERTICAL BEND_RADIUS ARM_WIDTH HOLE_RADIUS THICKNESS MESH_SIZE FORCE_MAGNITUDE FORCE_ANGLE) do (
    set /a "%%P_COUNT=0"
    for %%V in (!%%P_VALUES!) do set /a "%%P_COUNT+=1"
    set /a "TOTAL*=%%P_COUNT"
)
set /a "TARGET_TOTAL=TOTAL"

rem ------------------------------------------------------------
rem Optional test aid:
rem
rem Example:
rem
rem     set ELBOW_MAX_RUNS=10
rem     run_elbow_bracket_parametric.bat
rem
rem This stops after 10 attempted simulations.
rem ------------------------------------------------------------
if defined ELBOW_MAX_RUNS (

    set /a "TARGET_TOTAL=ELBOW_MAX_RUNS" 2>nul

    if !TARGET_TOTAL! LSS 1 (

        echo ERROR: ELBOW_MAX_RUNS must be a positive integer.

        popd
        exit /b 2
    )

    if !TARGET_TOTAL! GTR !TOTAL! (

        set /a "TARGET_TOTAL=TOTAL"
    )
)

rem ============================================================
rem Check required Python script
rem ============================================================
if not exist "%~dp0gsi_elbow_bracket_parametric_analysis.py" (

    echo ERROR: gsi_elbow_bracket_parametric_analysis.py was not found beside this script.

    popd
    exit /b 2
)

rem ============================================================
rem Make sure no stale simulation files exist before starting.
rem ============================================================
if exist "%~dp0results\" (

    echo ERROR: A results folder already exists.
    echo Move or remove it before starting.

    popd
    exit /b 2
)

if exist "%~dp0abaqus.rpy" (

    echo ERROR: abaqus.rpy already exists.
    echo Move or remove it before starting.

    popd
    exit /b 2
)

rem ============================================================
rem Output directory
rem ============================================================
set "RAW_ROOT=%~dp0elbow_bracket"

if not exist "%RAW_ROOT%\" (

    md "%RAW_ROOT%" || (

        echo ERROR: Cannot create the output folder "%RAW_ROOT%".

        popd
        exit /b 2
    )
)

rem ============================================================
rem Counters
rem
rem count     = next successful sample ID
rem attempted = total simulations attempted
rem succeeded = successful simulations
rem failed    = failed simulations
rem ============================================================
set /a "count=1"
set /a "attempted=0"
set /a "succeeded=0"
set /a "failed=0"

set "FATAL=0"

rem ============================================================
rem Find the first unused elbow_bracket_N directory.
rem ============================================================
call :FindNextID

rem ============================================================
rem tqdm-style progress bar
rem
rem 40 characters:
rem ########################################
rem ============================================================
set "BAR_FULL=########################################"
set "BAR_EMPTY=----------------------------------------"

set /a "BAR_WIDTH=40"

rem ============================================================
rem Start timing
rem ============================================================
call :NowCentiseconds START_CS

set /a "LAST_CS=START_CS"
set /a "ELAPSED_TOTAL_CS=0"

rem ============================================================
rem Header
rem ============================================================
echo.
echo ============================================================
echo Abaqus perforated elbow-bracket batch
echo ============================================================
echo Output directory: "%RAW_ROOT%"
echo.
echo Total parameter combinations: !TOTAL!

if not "!TARGET_TOTAL!"=="!TOTAL!" (

    echo Test limit: !TARGET_TOTAL! runs
)

echo.
echo Parameter values (m, N, degrees):
echo   Horizontal length : !HORIZONTAL_VALUES! [!HORIZONTAL_COUNT! levels]
echo   Vertical length   : !VERTICAL_VALUES! [!VERTICAL_COUNT! levels]
echo   Bend radius       : !BEND_RADIUS_VALUES! [!BEND_RADIUS_COUNT! levels]
echo   Arm width         : !ARM_WIDTH_VALUES! [!ARM_WIDTH_COUNT! levels]
echo   Hole radius       : !HOLE_RADIUS_VALUES! [!HOLE_RADIUS_COUNT! levels]
echo   Thickness         : !THICKNESS_VALUES! [!THICKNESS_COUNT! levels]
echo   Mesh size         : !MESH_SIZE_VALUES! [!MESH_SIZE_COUNT! levels]
echo   Force magnitude   : !FORCE_MAGNITUDE_VALUES! [!FORCE_MAGNITUDE_COUNT! levels]
echo   Force angle       : !FORCE_ANGLE_VALUES! [!FORCE_ANGLE_COUNT! levels]
echo.
echo Full grid: !TOTAL! combinations
echo.
echo Starting output ID: !count!
echo ============================================================
echo.

rem ============================================================
rem Initial tqdm-style progress display
rem ============================================================
call :ShowProgress 0 !TARGET_TOTAL!

rem ============================================================
rem Generate all parameter combinations
rem ============================================================
for %%A in (!HORIZONTAL_VALUES!) do (

    for %%B in (!VERTICAL_VALUES!) do (

        for %%C in (!BEND_RADIUS_VALUES!) do (

            for %%D in (!ARM_WIDTH_VALUES!) do (

                for %%E in (!HOLE_RADIUS_VALUES!) do (

                    for %%F in (!THICKNESS_VALUES!) do (

                        for %%G in (!MESH_SIZE_VALUES!) do (

                            for %%H in (!FORCE_MAGNITUDE_VALUES!) do (

                                for %%I in (!FORCE_ANGLE_VALUES!) do (

                                    call :RunOne ^
                                        "%%A" ^
                                        "%%B" ^
                                        "%%C" ^
                                        "%%D" ^
                                        "%%E" ^
                                        "%%F" ^
                                        "%%G" ^
                                        "%%H" ^
                                        "%%I"

                                    if "!FATAL!"=="1" (

                                        goto :Finished
                                    )

                                    if !attempted! GEQ !TARGET_TOTAL! (

                                        goto :Finished
                                    )
                                )
                            )
                        )
                    )
                )
            )
        )
    )
)

rem ============================================================
rem Finished
rem ============================================================
:Finished

if "!FATAL!"=="1" (

    set "EXIT_CODE=1"

    echo.
    echo ============================================================
    echo STOPPED
    echo ============================================================
    echo A fatal file operation error occurred.
    echo Current files were kept for inspection.
    echo.
    echo Attempted : !attempted!
    echo Succeeded : !succeeded!
    echo Failed    : !failed!
    echo Next ID   : !count!
    echo ============================================================

) else (

    rem --------------------------------------------------------
    rem Final 100%% progress display
    rem --------------------------------------------------------
    call :ShowProgress !attempted! !TARGET_TOTAL!

    set "EXIT_CODE=0"

    call :FormatSecondsFromCS !ELAPSED_TOTAL_CS! FINAL_TIME

    echo.
    echo ============================================================
    echo Completed
    echo ============================================================
    echo Attempted : !attempted!
    echo Succeeded : !succeeded!
    echo Failed    : !failed!
    echo Next ID   : !count!
    echo Elapsed   : !FINAL_TIME!
    echo ============================================================
)

popd

endlocal & exit /b %EXIT_CODE%

rem ============================================================
rem Run one Abaqus simulation
rem
rem %1 = horizontal length
rem %2 = vertical length
rem %3 = bend radius
rem %4 = arm width
rem %5 = hole radius
rem %6 = thickness
rem %7 = mesh size
rem %8 = force magnitude
rem %9 = force angle
rem ============================================================
:RunOne

rem ------------------------------------------------------------
rem Increase number of attempted simulations.
rem ------------------------------------------------------------
set /a "attempted+=1"

echo.
echo ------------------------------------------------------------
echo Run !attempted! / !TARGET_TOTAL!
echo Parameters:
echo   horizontal_length = %~1
echo   vertical_length   = %~2
echo   bend_radius       = %~3
echo   arm_width         = %~4
echo   hole_radius       = %~5
echo   thickness         = %~6
echo   mesh_size         = %~7
echo   force_magnitude   = %~8
echo   force_angle_deg   = %~9
echo ------------------------------------------------------------

rem ============================================================
rem Pass parameters to Abaqus Python
rem ============================================================
set "ELBOW_BRACKET_PARAMS=%~1,%~2,%~3,%~4,%~5,%~6,%~7,%~8,%~9"

rem ============================================================
rem Run Abaqus
rem ============================================================
call abaqus cae noGUI=gsi_elbow_bracket_parametric_analysis.py

set "ABAQUS_RC=!errorlevel!"

rem ============================================================
rem Simulation failure:
rem Abaqus returned non-zero exit code.
rem ============================================================
if not "!ABAQUS_RC!"=="0" (

    echo WARNING: Abaqus returned exit code !ABAQUS_RC!.

    set /a "failed+=1"

    call :CleanupRunFiles

    if errorlevel 1 (

        echo ERROR: Cannot clean files left by the failed simulation.

        set "FATAL=1"

        exit /b 1
    )

    echo Result: FAILED. Moving to the next combination.

    rem --------------------------------------------------------
    rem Update tqdm-style progress after failed run.
    rem --------------------------------------------------------
    call :ShowProgress !attempted! !TARGET_TOTAL!

    exit /b 0
)

rem ============================================================
rem Simulation failure:
rem results directory was not created.
rem ============================================================
if not exist "%~dp0results\" (

    set /a "failed+=1"

    call :CleanupRunFiles

    if errorlevel 1 (

        echo ERROR: Cannot clean files left by the failed simulation.

        set "FATAL=1"

        exit /b 1
    )

    echo Result: FAILED. No results folder was generated.
    echo Moving to the next combination.

    rem --------------------------------------------------------
    rem Update tqdm-style progress after failed run.
    rem --------------------------------------------------------
    call :ShowProgress !attempted! !TARGET_TOTAL!

    exit /b 0
)

rem ============================================================
rem Check required result files
rem ============================================================
set "MISSING_RESULT="

for %%R in (

    elbow_bracket_structure.cae
    elbow_bracket_results.odb
    stress_contour.jpg
    displacement_contour.jpg

) do (

    if not exist "%~dp0results\%%R" (

        if not defined MISSING_RESULT (

            set "MISSING_RESULT=%%R"
        )
    )
)

rem ============================================================
rem Simulation failure:
rem result directory exists but required files are incomplete.
rem ============================================================
if defined MISSING_RESULT (

    echo WARNING: Missing result file "!MISSING_RESULT!".

    set /a "failed+=1"

    call :CleanupRunFiles

    if errorlevel 1 (

        echo ERROR: Cannot clean incomplete result files.

        set "FATAL=1"

        exit /b 1
    )

    echo Result: FAILED. Moving to the next combination.

    rem --------------------------------------------------------
    rem Update tqdm-style progress after failed run.
    rem --------------------------------------------------------
    call :ShowProgress !attempted! !TARGET_TOTAL!

    exit /b 0
)

rem ============================================================
rem Destination directory
rem ============================================================
set "DEST=%RAW_ROOT%\elbow_bracket_!count!"

rem ============================================================
rem Safety check:
rem Never overwrite an existing sample.
rem ============================================================
if exist "!DEST!\" (

    echo ERROR: Destination "!DEST!" already exists.

    set "FATAL=1"

    exit /b 1
)

rem ============================================================
rem Create destination directory
rem ============================================================
md "!DEST!" >nul 2>&1

if errorlevel 1 (

    echo ERROR: Cannot create "!DEST!".

    set "FATAL=1"

    exit /b 1
)

rem ============================================================
rem Copy complete Abaqus result directory
rem ============================================================
robocopy ^
    "%~dp0results" ^
    "!DEST!" ^
    /E ^
    /COPY:DAT ^
    /DCOPY:DAT ^
    /R:2 ^
    /W:1 ^
    /NFL ^
    /NDL ^
    /NJH ^
    /NJS ^
    /NP >nul

set "COPY_RC=!errorlevel!"

rem ------------------------------------------------------------
rem robocopy:
rem codes 0-7 are considered successful.
rem codes >= 8 indicate a real copy error.
rem ------------------------------------------------------------
if !COPY_RC! GEQ 8 (

    echo ERROR: Copying results failed with robocopy exit code !COPY_RC!.

    set "FATAL=1"

    exit /b 1
)

rem ============================================================
rem Write parameters.txt
rem ============================================================
(
    echo horizontal_length=%~1
    echo vertical_length=%~2
    echo bend_radius=%~3
    echo arm_width=%~4
    echo hole_radius=%~5
    echo thickness=%~6
    echo mesh_size=%~7
    echo force_magnitude=%~8
    echo force_angle_deg=%~9
) > "!DEST!\parameters.txt"

rem ------------------------------------------------------------
rem Verify parameter file was successfully written.
rem ------------------------------------------------------------
if not exist "!DEST!\parameters.txt" (

    echo ERROR: Cannot write "!DEST!\parameters.txt".

    set "FATAL=1"

    exit /b 1
)

rem ============================================================
rem Successful simulation
rem ============================================================
set /a "succeeded+=1"
set /a "count+=1"

rem ============================================================
rem Clean temporary files generated by this simulation
rem ============================================================
call :CleanupRunFiles

if errorlevel 1 (

    echo ERROR: Cannot remove files left by the completed simulation.

    set "FATAL=1"

    exit /b 1
)

rem ============================================================
rem Find next available sample ID
rem ============================================================
call :FindNextID

echo Result: SUCCESS. Saved as "!DEST!".

rem ============================================================
rem Update tqdm-style progress after successful run
rem ============================================================
call :ShowProgress !attempted! !TARGET_TOTAL!

exit /b 0

rem ============================================================
rem Clean files generated by one Abaqus run
rem ============================================================
:CleanupRunFiles

rem ------------------------------------------------------------
rem Remove results directory
rem ------------------------------------------------------------
if exist "%~dp0results\" (

    rd /s /q "%~dp0results" >nul 2>&1
)

rem ------------------------------------------------------------
rem Remove Abaqus replay file
rem ------------------------------------------------------------
if exist "%~dp0abaqus.rpy" (

    del /f /q "%~dp0abaqus.rpy" >nul 2>&1
)

rem ------------------------------------------------------------
rem Verify results directory was actually removed
rem ------------------------------------------------------------
if exist "%~dp0results\" (

    echo ERROR: Cannot remove the results folder.

    exit /b 1
)

rem ------------------------------------------------------------
rem Verify abaqus.rpy was actually removed
rem ------------------------------------------------------------
if exist "%~dp0abaqus.rpy" (

    echo ERROR: Cannot remove abaqus.rpy.

    exit /b 1
)

exit /b 0

rem ============================================================
rem Find next unused output ID
rem ============================================================
:FindNextID

:FindNextIDLoop

if exist "%RAW_ROOT%\elbow_bracket_!count!\" (

    set /a "count+=1"

    goto :FindNextIDLoop
)

exit /b 0

rem ============================================================
rem tqdm-style progress display
rem
rem %1 = number of completed/attempted simulations
rem %2 = total target simulations
rem
rem Example:
rem
rem [############----------------------------] 30%% |
rem run=1555/5184 |
rem success=1530 |
rem failed=25 |
rem elapsed=03:21:10 |
rem ETA=07:26:02 |
rem next_id=1531
rem ============================================================
:ShowProgress

set /a "PROGRESS_DONE=%~1"
set /a "PROGRESS_TOTAL=%~2"

rem ------------------------------------------------------------
rem Protect against invalid values
rem ------------------------------------------------------------
if !PROGRESS_DONE! LSS 0 (

    set /a "PROGRESS_DONE=0"
)

if !PROGRESS_DONE! GTR !PROGRESS_TOTAL! (

    set /a "PROGRESS_DONE=PROGRESS_TOTAL"
)

rem ============================================================
rem Update elapsed time
rem ============================================================
call :NowCentiseconds NOW_CS

set /a "ELAPSED_DELTA_CS=NOW_CS-LAST_CS"

rem ------------------------------------------------------------
rem Midnight rollover
rem
rem 24 hours =
rem 24 * 60 * 60 * 100
rem = 8,640,000 centiseconds
rem ------------------------------------------------------------
if !ELAPSED_DELTA_CS! LSS 0 (

    set /a "ELAPSED_DELTA_CS+=8640000"
)

set /a "ELAPSED_TOTAL_CS+=ELAPSED_DELTA_CS"

set /a "LAST_CS=NOW_CS"

rem ============================================================
rem Progress percentage
rem ============================================================
if !PROGRESS_TOTAL! GTR 0 (

    set /a "PROGRESS_PERCENT=PROGRESS_DONE*100/PROGRESS_TOTAL"

) else (

    set /a "PROGRESS_PERCENT=0"
)

rem ============================================================
rem Progress bar
rem ============================================================
set /a "PROGRESS_FILLED=PROGRESS_DONE*BAR_WIDTH/PROGRESS_TOTAL"
set /a "PROGRESS_EMPTY=BAR_WIDTH-PROGRESS_FILLED"

for %%A in (!PROGRESS_FILLED!) do (

    set "PROGRESS_LEFT=!BAR_FULL:~0,%%A!"
)

for %%A in (!PROGRESS_EMPTY!) do (

    set "PROGRESS_RIGHT=!BAR_EMPTY:~0,%%A!"
)

rem ============================================================
rem Remaining simulations
rem ============================================================
set /a "PROGRESS_REMAINING=PROGRESS_TOTAL-PROGRESS_DONE"

rem ============================================================
rem Format elapsed time
rem ============================================================
call :FormatSecondsFromCS !ELAPSED_TOTAL_CS! ELAPSED_TEXT

rem ============================================================
rem Calculate ETA
rem
rem ETA =
rem average time per finished run
rem *
rem remaining runs
rem
rem Integer seconds are deliberately used here to avoid the
rem 32-bit arithmetic overflow that can occur with a very long
rem Abaqus batch.
rem ============================================================
if !PROGRESS_DONE! GTR 0 (

    set /a "ELAPSED_SECONDS=ELAPSED_TOTAL_CS/100"

    set /a "AVG_SECONDS=ELAPSED_SECONDS/PROGRESS_DONE"

    if !AVG_SECONDS! LSS 1 (

        set /a "AVG_SECONDS=1"
    )

    set /a "ETA_SECONDS=AVG_SECONDS*PROGRESS_REMAINING"

    call :FormatSeconds !ETA_SECONDS! ETA_TEXT

) else (

    set "ETA_TEXT=calculating"
)

rem ============================================================
rem Display tqdm-style progress bar
rem ============================================================
echo [!PROGRESS_LEFT!!PROGRESS_RIGHT!] !PROGRESS_PERCENT!%% ^| run=!PROGRESS_DONE!/!PROGRESS_TOTAL! ^| success=!succeeded! ^| failed=!failed! ^| elapsed=!ELAPSED_TEXT! ^| ETA=!ETA_TEXT! ^| next_id=!count!

exit /b 0

rem ============================================================
rem Get current clock time in centiseconds
rem ============================================================
:NowCentiseconds

set "CLOCK_VALUE=!time: =0!"

for /f "tokens=1-4 delims=:., " %%A in ("!CLOCK_VALUE!") do (

    set /a "CLOCK_CS=(1%%A-100)*360000+(1%%B-100)*6000+(1%%C-100)*100+(1%%D-100)"
)

set "%~1=!CLOCK_CS!"

exit /b 0

rem ============================================================
rem Convert centiseconds to HH:MM:SS
rem
rem %1 = centiseconds
rem %2 = output variable
rem ============================================================
:FormatSecondsFromCS

set /a "FORMAT_CS=%~1"

set /a "FORMAT_SECONDS=FORMAT_CS/100"

call :FormatSeconds !FORMAT_SECONDS! %~2

exit /b 0

rem ============================================================
rem Convert seconds to HH:MM:SS
rem
rem %1 = seconds
rem %2 = output variable
rem ============================================================
:FormatSeconds

set /a "FMT_TOTAL_SECONDS=%~1"

set /a "FMT_H=FMT_TOTAL_SECONDS/3600"

set /a "FMT_M=(FMT_TOTAL_SECONDS%%3600)/60"

set /a "FMT_S=FMT_TOTAL_SECONDS%%60"

if !FMT_H! LSS 10 (

    set "FMT_H=0!FMT_H!"
)

if !FMT_M! LSS 10 (

    set "FMT_M=0!FMT_M!"
)

if !FMT_S! LSS 10 (

    set "FMT_S=0!FMT_S!"
)

set "%~2=!FMT_H!:!FMT_M!:!FMT_S!"

exit /b 0
