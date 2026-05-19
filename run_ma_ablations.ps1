# Multi-Agent Ablation Runner for Tic-Tac-Toe and Leduc Poker (PowerShell)
# Ablations 0-5 are the same as runner.py
# Ablation 6 is the "Base" algorithm (Default params, high entropy for PPO)

$ENVS = @("tictactoe", "leduc")
$ALGOS = @("dqn", "ppo", "sac")
$ABLATIONS = 0..6

foreach ($env in $ENVS) {
    foreach ($algo in $ALGOS) {
        foreach ($ablation in $ABLATIONS) {
            Write-Host "Starting MA Ablation: Env=$env, Algo=$algo, Ablation=$ablation" -ForegroundColor Cyan
            
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
