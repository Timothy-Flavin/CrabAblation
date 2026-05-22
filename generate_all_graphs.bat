@echo off
REM Batch script to generate all graphs for all single agent environments and x-axis options
REM Usage: double-click or run in cmd: generate_all_graphs.bat

setlocal enabledelayedexpansion

REM Define environments and x-axis options
set ENVIRONMENTS=cartpole minigrid mujoco
set XAXES=episodes steps time

REM Default runs and smoothing weight
set RUNS=1 2 3 4 5
set WEIGHT=0.95

for %%E in (%ENVIRONMENTS%) do (
  for %%X in (%XAXES%) do (
    echo Generating: %%E %%X
    python graph.py --env %%E --runs %RUNS% --weight %WEIGHT% --xaxis %%X
  )
)

endlocal
