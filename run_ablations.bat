@echo off
setlocal enabledelayedexpansion

:: Default parameters
set ALGOS=sac
set ENVS=mujoco cartpole minigrid
set ABLATIONS=0 1 2 3 4 5
set RUNS=3
set DEVICE=cpu
set DEVICE_NAME=%COMPUTERNAME%

:: Parse named arguments
:parse_args
if "%~1"=="" goto end_parse
if "%~1"=="--algos" set "ALGOS=%~2" & shift & shift & goto parse_args
if "%~1"=="--envs" set "ENVS=%~2" & shift & shift & goto parse_args
if "%~1"=="--ablations" set "ABLATIONS=%~2" & shift & shift & goto parse_args
if "%~1"=="--runs" set "RUNS=%~2" & shift & shift & goto parse_args
if "%~1"=="--device" set "DEVICE=%~2" & shift & shift & goto parse_args
if "%~1"=="--device_name" set "DEVICE_NAME=%~2" & shift & shift & goto parse_args
echo Unknown parameter: %1
exit /b 1
:end_parse

:: Ensure results directory exists
if not exist results mkdir results

echo Starting Ablations on device: %DEVICE% (%DEVICE_NAME%)
echo ==================================================

for %%A in (%ALGOS%) do (
    for %%E in (%ENVS%) do (
        for %%B in (%ABLATIONS%) do (
            for /L %%R in (1, 1, %RUNS%) do (
                echo [%DATE% %TIME%] Running Algo: %%A ^| Env: %%E ^| Ablation: %%B ^| Run: %%R
                python runner.py --algo %%A --env_name %%E --ablation %%B --run %%R --device !DEVICE! --device_name !DEVICE_NAME!
            )
        )
    )
)

echo ==================================================
echo All trials completed successfully.
echo ==================================================
endlocal