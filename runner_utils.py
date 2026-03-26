from __future__ import annotations

import json
import os
import time

import numpy as np
import torch


def get_device_name():
    try:
        with open("device_name.txt", "r") as f:
            return f.read().strip()
    except Exception:
        try:
            return os.getlogin()
        except Exception:
            import getpass

            return getpass.getuser()


def get_benchmark_devices():
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    return devices


def resolve_torch_device(requested_device: str | None = None):
    if requested_device is None:
        requested_device = "cpu"
    dev = str(requested_device).strip()
    if dev.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA requested but not available; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(dev)


def benchmark_updates_generic(
    agent,
    device,
    batch_sizes,
    iters,
    make_batch_fn,
    update_fn,
    warmup_iters=3,
    header="Benchmarking Updates",
):
    print(f"\n--- {header} on {device.upper()} ---")
    agent.to(torch.device(device))

    for bs in batch_sizes:
        batch = make_batch_fn(bs, device)

        for _ in range(warmup_iters):
            update_fn(batch, 0)

        if device == "cuda":
            torch.cuda.synchronize()

        start_time = time.time()
        for i in range(iters):
            update_fn(batch, i)

        if device == "cuda":
            torch.cuda.synchronize()

        end_time = time.time()
        avg_time = (end_time - start_time) / iters
        print(
            f"Batch Size {bs:4d}: {avg_time*1000:.2f} ms / update | Updates/sec: {1.0/avg_time:.2f}"
        )


def benchmark_action_sampling_generic(
    agent,
    obs_dim,
    device,
    batch_sizes,
    iters,
    sample_fn,
    warmup_iters=5,
    header="Benchmarking Action Sampling",
):
    print(f"\n--- {header} on {device.upper()} ---")
    agent.to(torch.device(device))

    for bs in batch_sizes:
        obs = torch.randn((bs, obs_dim), device=device)

        for _ in range(warmup_iters):
            sample_fn(obs)

        if device == "cuda":
            torch.cuda.synchronize()

        start_time = time.time()
        for _ in range(iters):
            sample_fn(obs)

        if device == "cuda":
            torch.cuda.synchronize()

        end_time = time.time()
        avg_time = (end_time - start_time) / iters
        print(
            f"Batch Size (Num Envs) {bs:4d}: {avg_time*1000:.2f} ms / sample | Batched Samples/sec: {1.0/avg_time:.2f}"
        )


def get_grid_search_paths(args, algo_name):
    if hasattr(args, "device_name") and args.device_name is not None:
        file_prefix = args.device_name
    else:
        file_prefix = get_device_name()

    os.makedirs(f"time_files/{file_prefix}", exist_ok=True)
    best_filename = f"time_files/{file_prefix}/{args.env_name}_{algo_name}_best.json"
    all_filename = f"time_files/{file_prefix}/{args.env_name}_{algo_name}_all.json"
    return best_filename, all_filename


def save_grid_search_results(args, algo_name, best_results, all_results):
    best_filename, all_filename = get_grid_search_paths(args, algo_name)

    with open(best_filename, "w") as f:
        json.dump(best_results, f, indent=4)

    with open(all_filename, "w") as f:
        json.dump(all_results, f, indent=4)

    print(
        f"\nGrid search complete. Saved best configs to {best_filename} and all configs to {all_filename}"
    )


def load_grid_search_results(args, algo_name):
    best_filename, all_filename = get_grid_search_paths(args, algo_name)
    best_results = {}
    all_results = {}

    if os.path.exists(best_filename):
        try:
            with open(best_filename, "r") as f:
                loaded = json.load(f)
                if isinstance(loaded, dict):
                    best_results = loaded
        except Exception:
            pass

    if os.path.exists(all_filename):
        try:
            with open(all_filename, "r") as f:
                loaded = json.load(f)
                if isinstance(loaded, dict):
                    all_results = loaded
        except Exception:
            pass

    return best_results, all_results


def plot_results(results, args, model_name):
    import matplotlib.pyplot as plt

    runner_name = args.env_name
    results_dir = os.path.join("results", model_name, runner_name)
    os.makedirs(results_dir, exist_ok=True)

    np.save(
        os.path.join(results_dir, f"train_scores_{args.run}_{args.ablation}.npy"),
        results["rhist"],
    )
    np.save(
        os.path.join(results_dir, f"eval_scores_{args.run}_{args.ablation}.npy"),
        results["eval_hist"],
    )
    np.save(
        os.path.join(results_dir, f"loss_hist_{args.run}_{args.ablation}.npy"),
        results["lhist"],
    )
    np.save(
        os.path.join(
            results_dir,
            f"smooth_train_scores_{args.run}_{args.ablation}.npy",
        ),
        results["smooth_rhist"],
    )

    plt.plot(results["rhist"])
    plt.plot(results["smooth_rhist"])
    plt.legend(["R hist", "Smooth R hist"])
    plt.xlabel("Episode")
    plt.ylabel("Training Episode Reward")
    plt.grid()
    plt.title(f"Training rewards, run {args.run} ablated: {args.ablation}")
    plt.savefig(os.path.join(results_dir, f"train_scores_{args.run}_{args.ablation}"))
    plt.close()

    plt.plot(results["eval_hist"])
    plt.grid()
    plt.title(f"eval scores, run {args.run} ablated: {args.ablation}")
    plt.savefig(os.path.join(results_dir, f"eval_scores_{args.run}_{args.ablation}"))
    plt.close()

    train_time_seconds = results["train_time"]
    np.save(
        os.path.join(results_dir, f"train_time_{args.run}_{args.ablation}.npy"),
        train_time_seconds,
    )
    print(f"Training wall clock time: {train_time_seconds:.2f} seconds")
