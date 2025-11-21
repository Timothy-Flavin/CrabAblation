@echo off
REM Run 18 trials: 3 runs each for no ablation (0) and ablating pillars 1..5
REM Usage: run_minigrid_ablation.bat

SETLOCAL ENABLEDELAYEDEXPANSION

REM Optional: set DEVICE_ARG before calling, e.g.:
REM   set DEVICE_ARG=--device cuda
REM If not set, defaults to blank (uses script default device)
IF NOT DEFINED DEVICE_ARG (
  SET DEVICE_ARG=
)

ECHO Ensuring results directory exists
IF NOT EXIST results (
  mkdir results
)

FOR %%A IN (0 1 2 3 4 5) DO (
  FOR %%R IN (1 2 3) DO (
    ECHO Running trial: ablation=%%A, run=%%R
    python minigrid_dqn_runner.py --ablation %%A --run %%R !DEVICE_ARG!
  )
)

ECHO All 18 trials completed.
ENDLOCAL
