import os
import json
from ortools.linear_solver import pywraplp


def solve_scheduling_problem():
    # ---------------------------------------------------
    # 1. Data Setup
    # ---------------------------------------------------
    devices = ["timpc", "mac", "laptop"]
    envs = ["minigrid", "cartpole", "mujoco"]
    models = ["dqn", "ppo"]
    ablations = [0, 1, 2, 3, 4, 5]
    runs = [1, 2, 3]

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

    # Function to get steps per sec handling missing data gracefully
    def get_runtime_minutes(device, experiment):
        env = experiment["env"]
        model = experiment["model"]
        abl = experiment["ablation"]

        json_path = os.path.join("time_files", device, f"{env}_{model}_best.json")
        steps_per_sec = None

        # Default to a generic value if not found
        default_sps = 300.0
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

        total_steps = 1000000 if env == "mujoco" else 300000
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
    print("Solving... this may take a moment.")
    status = solver.Solve()

    if status == pywraplp.Solver.OPTIMAL:
        print("Optimal Schedule Found!")
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
                "set -euo pipefail\n\n"
            ]

            for i, exp in enumerate(experiments):
                if x[i, j].solution_value() == 1:
                    time_taken = runtime_matrix[i][j]
                    device_time += time_taken
                    dev_experiments.append(
                        f"  Env: {exp['env']:<8} | Model: {exp['model']:<3} | Ablation: {exp['ablation']} | Run: {exp['run']} (takes {time_taken:.2f} mins)"
                    )
                    
                    runner = "dqn_runner.py" if exp['model'] == "dqn" else "pg_runner.py"
                    sh_lines.append(f"echo \"[{device}] Running {exp['model']} on {exp['env']} | Ablation {exp['ablation']} | Run {exp['run']}\"\n")
                    sh_lines.append(f"python {runner} --env_name {exp['env']} --ablation {exp['ablation']} --run {exp['run']}\n\n")

            # Write the .sh file for this device
            sh_filename = os.path.join("time_files", f"run_{device}_experiments.sh")
            with open(sh_filename, "w", newline="\n") as sh_file:
                sh_file.writelines(sh_lines)

            for line in dev_experiments:
                print(line)
            print(f"  Total {device} Runtime: {device_time:.2f} mins\n")
            print(f"  Generated shell script saved to {sh_filename}\n")
            print("-" * 50)
    else:
        print("The solver could not find an optimal solution.")


if __name__ == "__main__":
    solve_scheduling_problem()
