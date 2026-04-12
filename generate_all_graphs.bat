@echo off
REM Batch script to generate all graphs for all algorithms, environments, and x-axis options
REM Usage: double-click or run in cmd: generate_all_graphs.bat

setlocal enabledelayedexpansion

REM Define algorithms, environments, and x-axis options
set ALGORITHMS=dqn sac ppo
set ENVIRONMENTS=cartpole minigrid mujoco hide-and-seek
set XAXES=episodes steps time

REM Default runs and smoothing weight
set RUNS=1 2 3
set WEIGHT=0.95
REM Set max_steps for 'steps' xaxis
set MAX_STEPS=1000000

for %%A in (%ALGORITHMS%) do (
  for %%E in (%ENVIRONMENTS%) do (
    for %%X in (%XAXES%) do (
      if "%%X"=="steps" (
        echo Generating: %%A %%E %%X [with max_steps]
        python graph.py --env %%E --runs %RUNS% --weight %WEIGHT% --xaxis %%X --max_steps %MAX_STEPS%
      ) else (
        echo Generating: %%A %%E %%X
        python graph.py --env %%E --runs %RUNS% --weight %WEIGHT% --xaxis %%X
      )
    )
  )
)

endlocal
