import argparse
import time

import gymnasium as gym
import numpy as np
import torch

from cleanrl_buffers import ReplayBuffer
from SAC_Rainbow import SACAgent
from runner_utilities import (
    make_env_thunk,
    benchmark_updates_generic,
    benchmark_action_sampling_generic,
    get_benchmark_devices,
    save_grid_search_results,
    load_grid_search_results,
    resolve_torch_device,
)


def _proxy_action_space(action_space):
    if isinstance(action_space, gym.spaces.Box):
        return action_space
    if isinstance(action_space, gym.spaces.Discrete):
        return gym.spaces.Box(
            low=np.zeros((1,), dtype=np.float32),
            high=np.ones((1,), dtype=np.float32),
            dtype=np.float32,
        )
    if isinstance(action_space, gym.spaces.MultiDiscrete):
        n_dims = int(len(action_space.nvec))
        return gym.spaces.Box(
            low=np.zeros((n_dims,), dtype=np.float32),
            high=np.ones((n_dims,), dtype=np.float32),
            dtype=np.float32,
        )
    raise NotImplementedError(f"Unsupported action space for SAC benchmark: {action_space}")


def _agent_spec_from_vec_env(vec_env):
    class _Spec:
        pass

    spec = _Spec()
    spec.single_observation_space = vec_env.single_observation_space
    spec.single_action_space = _proxy_action_space(vec_env.single_action_space)
    return spec


def _continuous_to_env_action(actions_cont, env_action_space):
    actions_cont = np.asarray(actions_cont, dtype=np.float32)

    if isinstance(env_action_space, gym.spaces.Box):
        return np.clip(actions_cont, env_action_space.low, env_action_space.high)

    if actions_cont.ndim == 1:
        actions_cont = actions_cont.reshape(-1, 1)

    if isinstance(env_action_space, gym.spaces.Discrete):
        x = np.clip(actions_cont[:, 0], 0.0, 1.0 - 1e-8)
        bins = (x * env_action_space.n).astype(np.int64)
        return np.clip(bins, 0, env_action_space.n - 1)

    if isinstance(env_action_space, gym.spaces.MultiDiscrete):
        out = np.zeros((actions_cont.shape[0], len(env_action_space.nvec)), dtype=np.int64)
        for dim, n in enumerate(env_action_space.nvec):
            x = np.clip(actions_cont[:, dim], 0.0, 1.0 - 1e-8)
            out[:, dim] = np.clip((x * int(n)).astype(np.int64), 0, int(n) - 1)
        return out

    raise NotImplementedError(f"Unsupported env action space: {env_action_space}")


def _hidden_layer_sizes_for_env(env_name: str):
    if env_name == "mujoco":
        return (128, 128)
    if env_name == "cartpole":
        return (32, 32)
    return (128, 128)


def setup_config(args, envs):
    cfg = {
        "entropy_coef_zero": False,
        "distributional": True,
        "dueling": True,
        "popart": True,
        "delayed_critics": True,
    }

    if args.ablation == 1:
        # Placeholder: SAC does not expose the same mirror-descent switch as DQN.
        pass
    elif args.ablation == 2:
        cfg["entropy_coef_zero"] = True
    elif args.ablation == 3:
        # Placeholder: SAC variant here does not include Beta/RND optimism.
        pass
    elif args.ablation == 4:
        cfg["distributional"] = False
        cfg["dueling"] = False
    elif args.ablation == 5:
        cfg["delayed_critics"] = False

    hidden_layer_sizes = _hidden_layer_sizes_for_env(args.env_name)
    agent = SACAgent(
        _agent_spec_from_vec_env(envs),
        gamma=args.gamma,
        tau=args.tau,
        policy_lr=args.policy_lr,
        q_lr=args.q_lr,
        policy_frequency=args.policy_frequency,
        target_network_frequency=args.target_network_frequency,
        alpha=args.alpha,
        autotune=args.autotune,
        entropy_coef_zero=cfg["entropy_coef_zero"],
        distributional=cfg["distributional"],
        dueling=cfg["dueling"],
        popart=cfg["popart"],
        delayed_critics=cfg["delayed_critics"],
        hidden_layer_sizes=hidden_layer_sizes,
        n_quantiles=args.n_quantiles,
        n_target_quantiles=args.n_target_quantiles,
    )
    return agent


def benchmark_updates(agent, obs_dim, action_dim, device="cpu", batch_sizes=[64, 256], iters=50):
    def make_batch(bs, dev):
        observations = torch.randn((bs, obs_dim), dtype=torch.float32, device=dev)
        next_observations = torch.randn((bs, obs_dim), dtype=torch.float32, device=dev)
        actions = torch.randn((bs, action_dim), dtype=torch.float32, device=dev)
        rewards = torch.randn((bs, 1), dtype=torch.float32, device=dev)
        dones = torch.zeros((bs, 1), dtype=torch.float32, device=dev)
        return {
            "observations": observations,
            "next_observations": next_observations,
            "actions": actions,
            "rewards": rewards,
            "dones": dones,
            "bs": bs,
        }

    class Batch:
        pass

    def run_update(batch, step_idx):
        data = Batch()
        data.observations = batch["observations"]
        data.next_observations = batch["next_observations"]
        data.actions = batch["actions"]
        data.rewards = batch["rewards"]
        data.dones = batch["dones"]
        agent.update(data, global_step=step_idx + batch["bs"])

    benchmark_updates_generic(
        agent,
        device=device,
        batch_sizes=batch_sizes,
        iters=iters,
        make_batch_fn=make_batch,
        update_fn=run_update,
    )


def benchmark_action_sampling(
    agent, obs_dim, device="cpu", batch_sizes=[1, 4, 16, 64], iters=200
):
    benchmark_action_sampling_generic(
        agent,
        obs_dim=obs_dim,
        device=device,
        batch_sizes=batch_sizes,
        iters=iters,
        sample_fn=lambda obs: agent.sample_action(obs, deterministic=False),
    )


def benchmark_env_rollouts(args, agent, total_steps=1000, batch_size=256):
    print(f"\n--- Benchmarking Environment Rollouts ---")
    print(f"Num Parallel Envs: {args.num_envs}, Device: {args.device}")
    device = resolve_torch_device(args.device)
    agent.to(device)

    env_fns = [make_env_thunk(args.fully_obs, args.env_name) for _ in range(args.num_envs)]
    vec_env = gym.vector.AsyncVectorEnv(env_fns)

    vec_env.single_observation_space.dtype = np.float32
    proxy_action_space = _proxy_action_space(vec_env.single_action_space)
    proxy_action_dim = int(np.prod(proxy_action_space.shape))
    rb = ReplayBuffer(
        args.buffer_size,
        vec_env.single_observation_space,
        proxy_action_space,
        device,
        n_envs=args.num_envs,
        handle_timeout_termination=False,
    )

    obs, _ = vec_env.reset()

    print(f"Starting {total_steps} parallel steps...")
    start_time = time.time()

    steps_taken = 0
    updates_performed = 0
    while steps_taken < total_steps:
        if steps_taken < args.learning_starts:
            if isinstance(vec_env.single_action_space, gym.spaces.Box):
                actions_agent = np.array(
                    [vec_env.single_action_space.sample() for _ in range(vec_env.num_envs)],
                    dtype=np.float32,
                )
            else:
                actions_agent = np.random.uniform(
                    low=0.0,
                    high=1.0,
                    size=(vec_env.num_envs, proxy_action_dim),
                ).astype(np.float32)
        else:
            actions_agent = agent.sample_action(obs, deterministic=False)

        actions_env = _continuous_to_env_action(actions_agent, vec_env.single_action_space)

        next_obs, rewards, terminations, truncations, infos = vec_env.step(actions_env)

        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc and "final_observation" in infos:
                real_next_obs[idx] = infos["final_observation"][idx]

        rb.add(obs, real_next_obs, actions_agent, rewards, terminations, infos)
        obs = next_obs
        steps_taken += args.num_envs

        if steps_taken > args.learning_starts:
            target_updates = steps_taken // 8
            while updates_performed < target_updates:
                if rb.size() < batch_size:
                    break
                data = rb.sample(batch_size)
                agent.update(data, global_step=steps_taken)
                updates_performed += 1

    end_time = time.time()
    duration = end_time - start_time
    sps = steps_taken / duration
    ups = updates_performed / duration if duration > 0 else 0.0

    print(f"Total time for {steps_taken} frame steps: {duration:.2f}s")
    print(f"Real Steps/sec: {sps:.2f}")
    print(f"Updates/sec: {ups:.2f}")

    vec_env.close()
    return sps, ups


def run_grid_search(args, fully_obs=False, total_steps=2000):
    print("\n=== Starting SAC Grid Search for Best Parameters ===")

    devices = get_benchmark_devices()
    num_envs_list = [1, 4, 8, 12, 16]
    if args.replace_existing:
        best_results = {}
        all_results = {}
    else:
        best_results, all_results = load_grid_search_results(args, "sac")
    args.fully_obs = fully_obs

    for ablation in range(6):
        print(f"\n--- Grid Search: Ablation {ablation} ---")
        args.ablation = ablation

        best_sps = 0.0
        best_config = None
        all_results.setdefault(f"ablation_{ablation}", [])
        existing_trials = {
            (entry.get("device"), entry.get("num_envs"))
            for entry in all_results[f"ablation_{ablation}"]
            if isinstance(entry, dict)
        }

        for dev in devices:
            for num_envs in num_envs_list:
                args.device = dev
                args.num_envs = num_envs

                if (not args.replace_existing) and ((dev, num_envs) in existing_trials):
                    print(
                        f"Skipping existing trial: ablation={ablation}, device={dev}, num_envs={num_envs}"
                    )
                    continue

                env_fns = [
                    make_env_thunk(args.fully_obs, args.env_name)
                    for _ in range(args.num_envs)
                ]
                vec_env = gym.vector.SyncVectorEnv(env_fns)
                try:
                    agent = setup_config(args, vec_env).to(torch.device(dev))
                    sps, ups = benchmark_env_rollouts(
                        args,
                        agent,
                        total_steps=total_steps,
                        batch_size=args.batch_size,
                    )
                    current_config = {
                        "device": dev,
                        "num_envs": num_envs,
                        "steps_per_sec": sps,
                        "updates_per_sec": ups,
                    }
                    if args.replace_existing and ((dev, num_envs) in existing_trials):
                        for idx, entry in enumerate(all_results[f"ablation_{ablation}"]):
                            if (
                                isinstance(entry, dict)
                                and entry.get("device") == dev
                                and entry.get("num_envs") == num_envs
                            ):
                                all_results[f"ablation_{ablation}"][idx] = current_config
                                break
                    else:
                        all_results[f"ablation_{ablation}"].append(current_config)
                        existing_trials.add((dev, num_envs))

                    ablation_entries = [
                        e
                        for e in all_results[f"ablation_{ablation}"]
                        if isinstance(e, dict) and "steps_per_sec" in e
                    ]
                    if ablation_entries:
                        best_results[f"ablation_{ablation}"] = max(
                            ablation_entries, key=lambda e: e["steps_per_sec"]
                        )

                    save_grid_search_results(args, "sac", best_results, all_results)

                    if sps > best_sps:
                        best_sps = sps
                        best_config = current_config
                except Exception as e:
                    import traceback

                    print(f"Error for device={dev}, num_envs={num_envs}: {e}")
                    traceback.print_exc()
                finally:
                    vec_env.close()

        best_results[f"ablation_{ablation}"] = best_config
        print(f"Best for Ablation {ablation}: {best_config}")

    save_grid_search_results(args, "sac", best_results, all_results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parallel Environment Benchmarking SAC")
    parser.add_argument(
        "--env_name",
        type=str,
        default="mujoco",
        choices=["mujoco", "cartpole", "minigrid"],
        help="Environment to use",
    )
    parser.add_argument(
        "--grid_search",
        action="store_true",
        help="Run parameter grid search and save json",
    )
    from runner_utilities import get_device_name

    parser.add_argument(
        "--device_name",
        type=str,
        default=get_device_name(),
        help="Device name for storing time_files",
    )
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
    parser.add_argument(
        "--replace_existing",
        action="store_true",
        default=False,
        help="Re-run and overwrite existing trials in json files",
    )

    parser.add_argument("--buffer_size", type=int, default=int(1e6))
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--learning_starts", type=int, default=0)
    parser.add_argument("--policy_lr", type=float, default=3e-4)
    parser.add_argument("--q_lr", type=float, default=1e-3)
    parser.add_argument("--policy_frequency", type=int, default=2)
    parser.add_argument("--target_network_frequency", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--autotune", action="store_true", default=True)
    parser.add_argument("--n_quantiles", type=int, default=32)
    parser.add_argument("--n_target_quantiles", type=int, default=32)

    args = parser.parse_args()

    env = make_env_thunk(args.fully_obs, args.env_name)()
    obs, _ = env.reset()
    obs_dim = int(np.prod(np.asarray(obs).shape))
    action_dim = int(np.prod(env.action_space.shape))
    env.close()

    print("=== Configuration ===")
    print(f"Environment: {args.env_name}")
    print(f"Ablation Level: {args.ablation}")
    print(f"Obs Dim: {obs_dim}")
    print(f"Action Dim: {action_dim}")
    print("=====================")

    if args.grid_search:
        run_grid_search(args, fully_obs=args.fully_obs, total_steps=5000)
    else:
        dummy_vec = gym.vector.SyncVectorEnv([make_env_thunk(args.fully_obs, args.env_name)])
        agent = setup_config(args, dummy_vec)
        dummy_vec.close()

        benchmark_updates(agent, obs_dim, action_dim, device="cpu", batch_sizes=[64, 256])
        if torch.cuda.is_available():
            benchmark_updates(
                agent,
                obs_dim,
                action_dim,
                device="cuda",
                batch_sizes=[64, 256],
            )

        benchmark_action_sampling(agent, obs_dim, device="cpu", batch_sizes=[1, 4, 16, 64, 256])
        if torch.cuda.is_available():
            benchmark_action_sampling(
                agent,
                obs_dim,
                device="cuda",
                batch_sizes=[1, 4, 16, 64, 256],
            )

        args.device = "cpu"
        for i in [1, 4, 8]:
            args.num_envs = i
            benchmark_env_rollouts(args, agent, total_steps=5000, batch_size=args.batch_size)
        if torch.cuda.is_available():
            args.device = "cuda"
            for i in [1, 4, 8, 12, 16]:
                args.num_envs = i
                benchmark_env_rollouts(args, agent, total_steps=5000, batch_size=args.batch_size)
