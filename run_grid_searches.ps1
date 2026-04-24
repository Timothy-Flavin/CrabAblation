# Define the configurations
$Algos = @("dqn", "ppo", "sac")
$Envs = @("cartpole", "minigrid", "mujoco")
$DeviceName = "laptop"
$SearchDevices = @("cpu", "cuda")

Write-Host "Starting grid searches for $DeviceName..." -ForegroundColor Cyan

foreach ($env in $Envs) {
    foreach ($algo in $Algos) {
        Write-Host "=====================================================" -ForegroundColor Yellow
        Write-Host "Benchmarking $algo on $env..." -ForegroundColor Yellow
        python benchmark.py --algo $algo --grid_search --device_name "$DeviceName" --env_name $env --search_devices $SearchDevices
    }
}

Write-Host "=====================================================" -ForegroundColor Green
Write-Host "All grid searches for $DeviceName completed!" -ForegroundColor Green