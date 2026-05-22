# Multi-Agent Ablation Runner for Tic-Tac-Toe and Leduc Poker (PowerShell)
# Ablations 0-5 are the same as runner.py
# Ablation 6 is the "Base" algorithm (Default params, high entropy for PPO)

$ENVS = @("tictactoe", "leduc", "rps")
$ALGOS = @("dqn", "sac", "ppo")
$ABLATIONS = 0..6
$RUNS = 1..5

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
                if ($env -eq "rps") {
                    $EPISODES = 5000
                }

                # Prepare arguments
                $ArgList = @(
                    "multiagent_runner.py",
                    "--algo", $algo,
                    "--ma_env", $env,
                    "--ablation", $ablation,
                    "--run", $run,
                    "--total_episodes", $EPISODES
                )

                # Extra flags for Ablation 6 PPO to ensure high entropy for Nash
                if ($algo -eq "ppo" -and $ablation -eq 6) {
                    $ArgList += "--ent_coef_override", "0.1"
                }

                # Run using the local venv python
                & ".\.venv\Scripts\python.exe" $ArgList
            }
        }
    }
}
