import os
import time
import argparse
import torch
import gymnasium as gym
import numpy as np
from minigrid.wrappers import FlatObsWrapper, OneHotPartialObsWrapper, FullyObsWrapper
from DQN_Rainbow import RainbowDQN, EVRainbowDQN


def make_env_thunk(fully_obs):
    def thunk():
        env = gym.make("MiniGrid-FourRooms-v0")
        if fully_obs:
            env = FullyObsWrapper(env)
        else:
            env = OneHotPartialObsWrapper(env)
        env = FlatObsWrapper(env)
        return env

    return thunk


def setup_config(args, obs_dim):
    cfg = {
        "munchausen": True,
        "soft": True,
        "Beta": 0.5 if args.ablation != 4 else 0.05,
        "dueling": True,
        "distributional": True,
        "ent_reg_coef": 0.01,
        "delayed": True,
        "tau": 0.03,
        "alpha": 0.9,
    }

    if args.ablation == 1:
        cfg["munchausen"] = False
        cfg["soft"] = False
    elif args.ablation == 2:
        cfg["ent_reg_coef"] = 0.0
    elif args.ablation == 3:
        cfg["Beta"] = 0.0
    elif args.ablation == 4:
        cfg["distributional"] = False
        cfg["dueling"] = False
        cfg["ent_reg_coef"] = 0.01
    elif args.ablation == 5:
        cfg["delayed"] = False

    AgentClass = RainbowDQN if cfg["distributional"] else EVRainbowDQN
    common_kwargs = dict(
        soft=cfg["soft"],
        munchausen=cfg["munchausen"],
        Thompson=False,
        dueling=cfg["dueling"],
        Beta=cfg["Beta"],
        ent_reg_coef=cfg["ent_reg_coef"],
        delayed=cfg["delayed"],
        tau=cfg["tau"],
        alpha=cfg["alpha"],
    )

    if AgentClass is RainbowDQN:
        dqn = AgentClass(
            obs_dim,
            1,
            7,
            hidden_layer_sizes=[128, 128],
            Beta_half_life_steps=50000,
            norm_obs=False,
            burn_in_updates=10,  # low for benchmark
            **common_kwargs,
        )
    else:
        dqn = AgentClass(
            obs_dim,
            1,
            7,
            hidden_layer_sizes=[128, 128],
            norm_obs=False,
            burn_in_updates=10,
            **common_kwargs,
        )
    return dqn


def move_agent_to_device(dqn, device):
    main_lr = dqn.optim.param_groups[0]["lr"] if hasattr(dqn, "optim") else 1e-3
    int_lr = dqn.int_optim.param_groups[0]["lr"] if hasattr(dqn, "int_optim") else 1e-3
    rnd_lr = dqn.rnd_optim.param_groups[0]["lr"] if hasattr(dqn, "rnd_optim") else 1e-3

    if hasattr(dqn, "ext_online"):
        dqn.ext_online.to(device)
    if hasattr(dqn, "ext_target"):
        dqn.ext_target.to(device)
    if hasattr(dqn, "int_online"):
        dqn.int_online.to(device)
    if hasattr(dqn, "int_target"):
        dqn.int_target.to(device)
    if hasattr(dqn, "rnd") and hasattr(dqn.rnd, "to"):
        dqn.rnd.to(device)
    if hasattr(dqn, "obs_rms") and hasattr(dqn.obs_rms, "to"):
        dqn.obs_rms.to(device)
    if hasattr(dqn, "int_rms") and hasattr(dqn.int_rms, "to"):
        dqn.int_rms.to(device)
    if hasattr(dqn, "ext_rms") and hasattr(dqn.ext_rms, "to"):
        dqn.ext_rms.to(device)

    # Recreate optimizers since parameters moved to a new device
    if hasattr(dqn, "ext_online"):
        dqn.optim = torch.optim.Adam(dqn.ext_online.parameters(), lr=main_lr)
    if hasattr(dqn, "int_online"):
        dqn.int_optim = torch.optim.Adam(dqn.int_online.parameters(), lr=int_lr)
    if hasattr(dqn, "rnd"):
        dqn.rnd_optim = torch.optim.Adam(dqn.rnd.predictor.parameters(), lr=rnd_lr)


def benchmark_updates(
    dqn, obs_dim, device="cpu", batch_sizes=[64, 256, 1024], iters=50
):
    print(f"\n--- Benchmarking Updates on {device.upper()} ---")
    move_agent_to_device(dqn, torch.device(device))

    for bs in batch_sizes:
        # Create dummy buffer
        obs = torch.randn((bs, obs_dim), device=device)
        next_obs = torch.randn((bs, obs_dim), device=device)
        actions = torch.randint(0, 7, (bs,), device=device)
        rewards = torch.randn((bs,), device=device)
        terms = torch.zeros((bs,), device=device)

        # Warmup
        for _ in range(3):
            dqn.update(obs, actions, rewards, next_obs, terms, batch_size=bs, step=bs)

        if device == "cuda":
            torch.cuda.synchronize()

        start_time = time.time()
        for i in range(iters):
            dqn.update(obs, actions, rewards, next_obs, terms, batch_size=bs, step=bs)

        if device == "cuda":
            torch.cuda.synchronize()

        end_time = time.time()
        avg_time = (end_time - start_time) / iters
        print(
            f"Batch Size {bs:4d}: {avg_time*1000:.2f} ms / update | Updates/sec: {1.0/avg_time:.2f}"
        )


def benchmark_action_sampling(
    dqn, obs_dim, device="cpu", batch_sizes=[1, 4, 16, 64], iters=200
):
    print(f"\n--- Benchmarking Action Sampling on {device.upper()} ---")
    move_agent_to_device(dqn, torch.device(device))

    for bs in batch_sizes:
        obs = torch.randn((bs, obs_dim), device=device)

        # Warmup
        for _ in range(5):
            dqn.sample_action(obs, eps=0.1, step=0)

        if device == "cuda":
            torch.cuda.synchronize()

        start_time = time.time()
        for _ in range(iters):
            dqn.sample_action(obs, eps=0.1, step=0)

        if device == "cuda":
            torch.cuda.synchronize()

        end_time = time.time()
        avg_time = (end_time - start_time) / iters
        print(
            f"Batch Size (Num Envs) {bs:4d}: {avg_time*1000:.2f} ms / sample | Batched Samples/sec: {1.0/avg_time:.2f}"
        )


def benchmark_env_rollouts(args, dqn, obs_dim, total_steps=1000):
    print(f"\n--- Benchmarking Environment Rollouts ---")
    print(f"Num Parallel Envs: {args.num_envs}, Device: {args.device}")
    device = torch.device(args.device)
    move_agent_to_device(dqn, device)

    # Init VecEnv
    print("Initializing Vector Environment...")
    env_fns = [make_env_thunk(args.fully_obs) for _ in range(args.num_envs)]
    vec_env = gym.vector.AsyncVectorEnv(env_fns)

    obs, info = vec_env.reset()

    # Warmup
    actions = np.random.randint(0, 7, size=(args.num_envs,))
    obs, _, _, _, _ = vec_env.step(actions)

    print(f"Starting {total_steps} parallel steps...")
    start_time = time.time()

    steps_taken = 0
    while steps_taken < total_steps:
        # 1. CPU -> GPU for obs
        tobs = torch.from_numpy(obs).to(device).float()

        # 2. Network forward pass
        actions = dqn.sample_action(tobs, eps=0.1, step=steps_taken)

        if isinstance(actions, list) and isinstance(actions[0], list):
            actions = [a[0] for a in actions]  # Flatten if list of lists (for dim=1)

        actions = np.array(actions)

        # 4. Env step (multiprocessing over CPU)
        obs, r, term, trunc, info = vec_env.step(actions)

        steps_taken += args.num_envs

    end_time = time.time()
    duration = end_time - start_time
    sps = steps_taken / duration
    print(f"Total time for {steps_taken} frame steps: {duration:.2f}s")
    print(f"Real Steps/sec: {sps:.2f}")

    vec_env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parallel Environment Benchmarking")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Main device",
    )
    parser.add_argument("--ablation", type=int, default=0, choices=[0, 1, 2, 3, 4, 5])
    parser.add_argument("--fully_obs", action="store_true")
    parser.add_argument(
        "--num_envs", type=int, default=8, help="Number of parallel environments to run"
    )
    args = parser.parse_args()

    # Get obs dim from dummy env
    env = make_env_thunk(args.fully_obs)()
    obs, _ = env.reset()
    obs_dim = int(np.prod(np.asarray(obs).shape))
    env.close()

    print("=== Configuration ===")
    print(f"Ablation Level: {args.ablation}")
    print(f"Obs Dim: {obs_dim}")
    print("=====================")

    dqn = setup_config(args, obs_dim)

    # # 1. Benchmark Updates
    # benchmark_updates(dqn, obs_dim, device="cpu", batch_sizes=[64, 256, 1024])
    # if torch.cuda.is_available():
    #     benchmark_updates(dqn, obs_dim, device="cuda", batch_sizes=[64, 256, 1024])

    # # 2. Benchmark Action Sampling
    # benchmark_action_sampling(
    #     dqn, obs_dim, device="cpu", batch_sizes=[1, 4, 16, 64, 256]
    # )
    # if torch.cuda.is_available():
    #     benchmark_action_sampling(
    #         dqn, obs_dim, device="cuda", batch_sizes=[1, 4, 16, 64, 256]
    #     )

    # 3. Benchmark Parallel Env Execution
    args.device = "cpu"
    for i in [1, 16, 32]:
        args.num_envs = i
        benchmark_env_rollouts(args, dqn, obs_dim, total_steps=5000)
    if torch.cuda.is_available():
        args.device = "cuda"
        for i in [1, 16, 32]:
            args.num_envs = i
            benchmark_env_rollouts(args, dqn, obs_dim, total_steps=5000)
