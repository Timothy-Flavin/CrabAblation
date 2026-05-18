$SessionName = "ablation2"

# Function to check status and deploy to a specific machine
function Deploy-ToMachine {
    param (
        [Parameter(Mandatory=$true)]
        [string]$TargetHost,

        [Parameter(Mandatory=$true)]
        [string]$RemoteDir,

        # This captures all subsequent arguments as an array of strings
        [Parameter(ValueFromRemainingArguments=$true)]
        [string[]]$Scripts
    )

    Write-Host "========================================"
    Write-Host "Checking availability for: $TargetHost" -ForegroundColor Cyan

    # 1. Check if the machine is online
    # 2>$null suppresses standard error output in PowerShell
    ssh -q -o BatchMode=yes -o ConnectTimeout=5 $TargetHost "exit" 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[SKIP] Skipping ${TargetHost}: Machine is offline or unreachable via SSH keys." -ForegroundColor DarkYellow
        return
    }

    # 2. Check if the machine is busy
    # ssh -q $TargetHost "export PATH=`$PATH:/opt/homebrew/bin:/usr/local/bin && tmux has-session -t $SessionName 2>/dev/null" 2>$null
    ssh -q $TargetHost "tmux has-session -t $SessionName 2>/dev/null" 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[SKIP] Skipping ${TargetHost}: Tmux session '$SessionName' already exists. Job is currently running." -ForegroundColor Yellow
        return
    }

    Write-Host "[OK] ${TargetHost} is available. Starting deployment in $RemoteDir..." -ForegroundColor Green

    # 3. Format the script commands to run in parallel or sequence
    $ScriptCmds = ""
    for ($i = 0; $i -lt $Scripts.Count; $i++) {
        if ($i -lt ($Scripts.Count - 1)) {
            # Background all scripts except the last one
            $ScriptCmds += "./time_files/$($Scripts[$i]) & "
        } else {
            # Last script runs in foreground
            $ScriptCmds += "./time_files/$($Scripts[$i])"
        }
    }

    # 4. Construct the remote command
    # $RemoteCmd = "export PATH=`$PATH:/opt/homebrew/bin:/usr/local/bin && cd $RemoteDir && git pull origin main && rm -rf ./results && tmux new-session -d -s $SessionName 'source .venv/bin/activate; $ScriptCmds; wait; exec bash'"
    $RemoteCmd = "cd $RemoteDir && git pull origin main && rm -rf ./results && tmux new-session -d -s $SessionName 'source .venv/bin/activate; $ScriptCmds; wait; exec bash'"

    # 5. Execute the command on the remote host
    ssh $TargetHost $RemoteCmd
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[SUCCESS] Launched jobs on ${TargetHost}." -ForegroundColor Green
    } else {
        Write-Host "[FAIL] Failed to launch jobs on ${TargetHost}." -ForegroundColor Red
    }
}

# --- Define the machines, their specific directories, and their corresponding scripts ---
# Replace the placeholder directories with your actual paths on the Linux machines

# Deploy-ToMachine "alienware" "~/Desktop/AAMAS/CrabAblation" `
#     "run_alienware_gpu_0_experiments.sh" "run_alienware_gpu_1_experiments.sh"

Deploy-ToMachine "white-machine" "~/Desktop/CrabAblation" `
    "run_white-machine_gpu0_experiments.sh" "run_white-machine_gpu1_experiments.sh"

# Deploy-ToMachine "lab-comp" "~/CrabAblation" `
#     "run_lab-comp_cpu_experiments.sh" "run_lab-comp_gpu_experiments.sh"

# Deploy-ToMachine "timpc" "~/Desktop/CrabAblation" `
#     "run_timpc_experiments.sh"

Deploy-ToMachine "mac" "~/Desktop/CrabAblation" `
    "run_mac_experiments.sh"

# On laptop
# Deploy-ToMachine "laptop" "~/laptop_workspace" `
#     "run_laptop_experiments.sh"

Write-Host "========================================"
Write-Host "Deployment orchestration complete." -ForegroundColor Cyan