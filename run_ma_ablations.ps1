# Multi-Agent Ablation Runner for Leduc Poker and Rock-Paper-Scissors (PowerShell).
# Tic-Tac-Toe is excluded for now (too slow on the current machine).
# Mirrors run_ma_ablations.sh so the two stay in sync.
# Ablations 0-5 are the same as runner.py
# Ablation 6 is the "Base" algorithm (Default params, high entropy for PPO)

# Force UTF-8 stdout so emoji/unicode prints don't crash on the Windows cp1252 console
# (otherwise every run dies at import with UnicodeEncodeError on a '✅').
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

# Portable python: prefer the Windows venv, fall back to the POSIX venv, then PATH.
if (Test-Path ".\.venv\Scripts\python.exe") {
    $PY = ".\.venv\Scripts\python.exe"
}
elseif (Test-Path ".\.venv\bin\python") {
    $PY = ".\.venv\bin\python"
}
else {
    $PY = "python"
}

$ENVS = @("leduc")
$ALGOS = @("sac", "dqn", "ppo")
$ABLATIONS = 0..6
$RUNS = 1..2
$EVAL_EPISODES = 50
# Discrete-uniform regularizer strength for SAC on the (discrete) MA games. Pulls the
# argmax policy toward uniform; auto-disabled for the entropy ablation and continuous envs.
$SAC_DISCRETE_ENTROPY = 5.0

foreach ($env in $ENVS) {
    foreach ($algo in $ALGOS) {
        foreach ($ablation in $ABLATIONS) {
            foreach ($run in $RUNS) {
                Write-Host "Starting MA Ablation: Env=$env, Algo=$algo, Ablation=$ablation, Run=$run" -ForegroundColor Cyan

                # Default episodes
                $EPISODES = 10000
                if ($env -eq "leduc") {
                    $EPISODES = 20000
                }

                # Prepare arguments
                $ArgList = @(
                    "multiagent_runner.py",
                    "--algo", $algo,
                    "--ma_env", $env,
                    "--ablation", $ablation,
                    "--run", $run,
                    "--total_episodes", $EPISODES,
                    "--eval_episodes", $EVAL_EPISODES
                )

                # Ablation 6 PPO needs high entropy for Nash
                if ($algo -eq "ppo" -and $ablation -eq 6) {
                    $ArgList += "--ent_coef_override", "0.1"
                }
                # SAC uses the discrete-uniform regularizer on these discrete games
                if ($algo -eq "sac") {
                    $ArgList += "--discrete_entropy", $SAC_DISCRETE_ENTROPY
                }

                & $PY $ArgList
            }
        }
    }
}
