@echo off
REM Auto-generated ablation runner for Windows Batch
REM Usage: Edit ALGOS, ENVS, ABLATIONS, RUNS, DEVICE_NAME as needed

REM Set parameters (space-separated lists)
set "ALGOS=dqn"
set "ENVS=mujoco"
set "ABLATIONS=0 1 2 3 4 5"
set "RUNS=2"
set "DEVICE_NAME=laptop"

REM Parse named arguments (not supported in batch, edit above)
REM If you want to pass arguments, use a PowerShell script instead

if "%DEVICE_NAME%"=="" (
    echo Error: DEVICE_NAME is required. Set it at the top of this file.
    pause
    exit /b 1
)

REM Ensure results directory exists
if not exist results mkdir results

echo Starting Ablations on device computer: %DEVICE_NAME%
echo ==================================================

for %%A in (%ALGOS%) do (
    for %%E in (%ENVS%) do (
        for %%B in (%ABLATIONS%) do (
            for /L %%R in (1,1,%RUNS%) do (
                set "RESULT_FILE=results/%%A/%%E/train_scores_%%R_%%B.npy"
                echo [RUN ] Algo: %%A ^| Env: %%E ^| Ablation: %%B ^| Run: %%R
                if not exist results\%%A mkdir results\%%A
                if not exist results\%%A\%%E mkdir results\%%A\%%E
                .\.venv\Scripts\python.exe runner.py --algo %%A --env_name %%E --ablation %%B --run %%R --device_name %DEVICE_NAME%
            )
        )
    )
)

echo ==================================================
echo All trials completed.
echo ==================================================
pause
