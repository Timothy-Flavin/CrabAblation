import os
import json
import yaml
from ortools.linear_solver import pywraplp

def solve_scheduling_problem():
    # ---------------------------------------------------
    # 1. Data Setup & Hardware Topology
    # ---------------------------------------------------
    devices = [
        "timpc", 
        "mac", 
        "laptop", 
        "white-machine_gpu0", 
        "white-machine_gpu1", 
        #"alienware_gpu_0", 
        #"alienware_gpu_1", 
        "lab-comp_cpu", 
        "lab-comp_gpu"
    ]
    
    command_pre_appends = {
        "timpc": "",
        "mac": "",
        "laptop": "", 
        "white-machine_gpu0": "OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 taskset -c 0-3,8-11 ", 
        "white-machine_gpu1": "OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 taskset -c 4-7,12-15 ", 
        "alienware_gpu_0": "OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 taskset -c 0,1,4,5 ", 
        "alienware_gpu_1": "OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 taskset -c 2,3,6,7 ", 
        "lab-comp_cpu": "OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=\"\" numactl --cpunodebind=0 --membind=0 ", 
        "lab-comp_gpu": "OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=1 --membind=1 ", 
    }

    envs = ["minigrid", "cartpole", "mujoco"]
    models = ["dqn", "ppo", "sac"]
    ablations = [0, 1, 2, 3, 4, 5]
    runs = [1, 2, 3, 4, 5]

    # Generate the list of experiments
    experiments = []
    for env in envs:
        for model in models:
            for abl in ablations:
                for run in runs:
                    experiments.append(
                        {"env": env, "model": model, "ablation": abl, "run": run}
                    )

    num_experiments = len(experiments)
    num_devices = len(devices)

    # Load env_config.yaml for max_steps
    try:
        with open(os.path.join(os.path.dirname(__file__), "../env_config.yaml"), "r") as f:
            ENV_CONFIG = yaml.safe_load(f)
    except FileNotFoundError:
        print("Warning: env_config.yaml not found. Using fallback step counts.")
        ENV_CONFIG = {}

    # Function to get steps per sec handling missing data gracefully
    def get_runtime_minutes(device, experiment):
        env = experiment["env"]
        model = experiment["model"]
        abl = experiment["ablation"]

        json_path = os.path.join("time_files", device, f"{env}_{model}_best.json")
        steps_per_sec = None

        # Default to a generic value if benchmark file is missing
        default_sps = 30.0
        try:
            with open(json_path, "r") as f:
                data = json.load(f)
                abl_key = f"ablation_{abl}"
                if abl_key in data:
                    steps_per_sec = data[abl_key].get("steps_per_sec", default_sps)
                else:
                    steps_per_sec = default_sps
        except Exception:
            steps_per_sec = default_sps
            
        # Prevent division by zero
        if steps_per_sec <= 0:
            steps_per_sec = 1.0

        # Safely parse max_steps from dict or fallback to 300000
        env_dict = ENV_CONFIG.get(env, {})
        max_steps_val = env_dict.get("max_steps", 300000)
        if isinstance(max_steps_val, dict):
            total_steps = int(max_steps_val.get(model, 300000))
        else:
            total_steps = int(max_steps_val)
            
        # return runtime in minutes
        return (total_steps / steps_per_sec) / 60.0

    runtime_matrix = [
        [get_runtime_minutes(device, exp) for device in devices] for exp in experiments
    ]

    # ---------------------------------------------------
    # 2. Solver Setup
    # ---------------------------------------------------
    # Create the linear solver using the SCIP backend (great for Mixed Integer Programming)
    solver = pywraplp.Solver.CreateSolver("SCIP")
    if not solver:
        print("SCIP solver not available.")
        return

    # ENABLE SOLVER LOGS: Prints periodic progress and bounds
    solver.EnableOutput()
    
    # Optional: Set a time limit (e.g., 5 minutes = 300,000 milliseconds)
    # The solver will return the best schedule found within this time frame.
    solver.SetTimeLimit(300000) 

    # ---------------------------------------------------
    # 3. Variables
    # ---------------------------------------------------
    # x[i, j] is a boolean variable: 1 if experiment i is assigned to device j, 0 otherwise.
    x = {}
    for i in range(num_experiments):
        for j in range(num_devices):
            x[i, j] = solver.IntVar(0, 1, f"x_{i}_{j}")

    # The makespan is the maximum total runtime across all devices. We want to minimize this.
    makespan = solver.NumVar(0, solver.infinity(), "makespan")

    # ---------------------------------------------------
    # 4. Constraints
    # ---------------------------------------------------
    # Constraint A: Every experiment must be assigned to exactly ONE device.
    for i in range(num_experiments):
        solver.Add(sum(x[i, j] for j in range(num_devices)) == 1)

    # Constraint B: The total runtime on ANY device cannot exceed the makespan.
    for j in range(num_devices):
        total_time_on_device_j = sum(
            runtime_matrix[i][j] * x[i, j] for i in range(num_experiments)
        )
        solver.Add(total_time_on_device_j <= makespan)

    # ---------------------------------------------------
    # 5. Objective
    # ---------------------------------------------------
    solver.Minimize(makespan)

    # ---------------------------------------------------
    # 6. Solve & Print Results
    # ---------------------------------------------------
    print(f"Assigning {num_experiments} experiments across {num_devices} workers.")
    print("Solving... streaming SCIP optimization logs below:\n")
    print("-" * 50)
    
    status = solver.Solve()
    
    print("-" * 50)

    if status in [pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE]:
        if status == pywraplp.Solver.OPTIMAL:
            print("Status: Optimal Schedule Found!")
        else:
            print("Status: Feasible Schedule Found (Time Limit Reached)!")
            
        print(
            f"Total time to finish all experiments: {makespan.solution_value():.2f} minutes\n"
        )

        for j, device in enumerate(devices):
            print(f"--- Device: {device} ---")
            device_time = 0
            dev_experiments = []

            sh_lines = [
                "#!/usr/bin/env bash\n",
                f"# Auto-generated schedule for {device}\n",
                "set -euo pipefail\n\n",
            ]

            preamble = command_pre_appends.get(device, "")

            for i, exp in enumerate(experiments):
                if x[i, j].solution_value() > 0.5: # Floating point safe check for 1
                    time_taken = runtime_matrix[i][j]
                    device_time += time_taken
                    dev_experiments.append(
                        f"  Env: {exp['env']:<8} | Model: {exp['model']:<3} | Ablation: {exp['ablation']} | Run: {exp['run']} (takes {time_taken:.2f} mins)"
                    )

                    sh_lines.append(
                        f"echo \"[{device}] Running {exp['model']} on {exp['env']} | Ablation {exp['ablation']} | Run {exp['run']}\"\n"
                    )
                    
                    # Command with the hardware preamble injected and device_name passed explicitly
                    cmd = f"{preamble}python runner.py --algo {exp['model']} --env_name {exp['env']} --ablation {exp['ablation']} --run {exp['run']} --device_name {device}\n\n"
                    sh_lines.append(cmd)

            # Write the .sh file for this device
            os.makedirs("time_files", exist_ok=True)
            sh_filename = os.path.join("time_files", f"run_{device}_experiments.sh")
            with open(sh_filename, "w", newline="\n") as sh_file:
                sh_file.writelines(sh_lines)

            for line in dev_experiments:
                print(line)
            print(f"  Total {device} Runtime: {device_time:.2f} mins")
            print(f"  Generated shell script saved to {sh_filename}\n")
            print("-" * 50)
    else:
        print("The solver could not find a feasible solution.")

if __name__ == "__main__":
    solve_scheduling_problem()