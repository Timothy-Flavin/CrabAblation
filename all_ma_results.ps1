$algos = @("dqn", "ppo", "sac")
$envs = @("rps", "leduc", "tictactoe")#, "tictactoe"

foreach ($algo in $algos) {
    foreach ($env in $envs) {
        Write-Host "Plotting $algo on $env..."
        python plot_ma_results.py --algo $algo --env_name $env --results_dir results
    }
}
