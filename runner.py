from __future__ import annotations

import argparse
import json
import os
import time
from types import SimpleNamespace
from typing import Callable, Optional

import gymnasium as gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from runner_utils import get_device_name, plot_results, resolve_torch_device
from environment_utils import (
    ActionTransformHandler,
    _proxy_action_space,
    extract_mixed_observation_shapes,
    get_env_benchmark_spec,
    make_hide_and_seek_vec_env,
    make_env_thunk,
)
from learning_algorithms.cleanrl_buffers import ReplayBuffer
from learning_algorithms.DQN_Rainbow import EVRainbowDQN, RainbowDQN
from learning_algorithms.MixedObservationEncoder import MixedObservationEncoder
from learning_algorithms.PG_Rainbow import PPOAgent
from learning_algorithms.SAC_Rainbow import SACAgent


def get_args():
    parser = argparse.ArgumentParser(description="Unified RL runner")
    parser.add_argument(
        "--algo", type=str, default="dqn", choices=["dqn", "ppo", "sac"]
    )
    parser.add_argument(
        "--env_name",
        type=str,
        default="minigrid",
        choices=["cartpole", "minigrid", "mujoco", "hide-and-seek-engine"],
    )
    parser.add_argument(
        "--env_id",
        type=str,
        default="",
        help="Optional explicit env id. Defaults to env_name.",
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--ablation", type=int, default=0, choices=[0, 1, 2, 3, 4, 5])
    parser.add_argument("--fully_obs", action="store_true")
    parser.add_argument("--run", type=int, default=1)
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--num_steps", type=int, default=128)
    parser.add_argument("--total_steps", type=int, default=1000000)
    parser.add_argument(
        "--total_steps_override",
        type=int,
        default=0,
        help="If > 0, force this total step budget for rollouts",
    )
    parser.add_argument(
        "--max_wall_time",
        type=float,
        default=0.0,
        help="Maximum wall-clock seconds before early stop",
    )
    parser.add_argument("--device_name", type=str, default=get_device_name())
    parser.add_argument(
        "--skip_best_params",
        action="store_true",
        help="Do not load best grid-search params for device/num_envs",
    )

    # DQN knobs
    parser.add_argument("--dqn_buffer_size", type=int, default=10000)
    parser.add_argument("--dqn_batch_size", type=int, default=64)
    parser.add_argument("--update_every", type=int, default=4)
    parser.add_argument("--rnd_burn_in", type=int, default=1000)

    # SAC knobs
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
    parser.add_argument(
        "--hide_seek_bins_per_dim",
        type=int,
        default=3,
        help="Discretization bins per Box dimension for discrete mixed-action wrapper.",
    )

    args = parser.parse_args()

    if args.algo == "sac" and args.update_every == 4:
        args.update_every = 4

    if args.env_name == "mujoco" and args.total_steps == 1000000:
        args.total_steps = 2000000
    if (
        args.algo == "dqn"
        and args.env_name == "mujoco"
        and args.dqn_buffer_size == 10000
    ):
        args.dqn_buffer_size = 20000

    if not args.skip_best_params:
        _maybe_load_best_params(args)

    if not args.env_id:
        args.env_id = args.env_name

    return args


def _maybe_load_best_params(args):
    best_json_path = (
        f"time_files/{args.device_name}/{args.env_name}_{args.algo}_best.json"
    )
    try:
        with open(best_json_path, "r") as f:
            best_results = json.load(f)
        best_config = best_results.get(f"ablation_{args.ablation}")
        if isinstance(best_config, dict):
            args.num_envs = int(best_config.get("num_envs", args.num_envs))
            args.device = str(best_config.get("device", args.device))
    except Exception:
        pass


def create_vec_env(args, num_envs: int | None = None):
    env_id = getattr(args, "env_id", args.env_name)
    if env_id == "hide-and-seek-engine" or args.env_name == "hide-and-seek-engine":
        action_mode = "continuous" if args.algo == "sac" else "discrete"

        vec_env = make_hide_and_seek_vec_env(
            action_mode=action_mode,
            bins_per_dim=int(getattr(args, "hide_seek_bins_per_dim", 3)),
        )
        args.num_envs = int(vec_env.num_envs)
        return vec_env
    run = getattr(args, "run", 999) if getattr(args, "run", None) is not None else 999
    n_envs = int(num_envs if num_envs is not None else args.num_envs)
    env_fns = [
        make_env_thunk(args.fully_obs, args.env_name, seed=run + i, idx=i)
        for i in range(n_envs)
    ]
    return gym.vector.SyncVectorEnv(env_fns)


def _encoder_factory_from_vec_env(
    args, vec_env
) -> Optional[Callable[[], torch.nn.Module]]:
    env_id = getattr(args, "env_id", args.env_name)
    if env_id != "hide-and-seek-engine" and args.env_name != "hide-and-seek-engine":
        return None

    spatial_shape = getattr(vec_env, "spatial_shape", None)
    vector_dim = getattr(vec_env, "vector_dim", None)
    if spatial_shape is None or vector_dim is None:
        raw_obs_space = getattr(vec_env, "raw_observation_space", None)
        if raw_obs_space is None:
            return None
        spatial_shape, vector_dim = extract_mixed_observation_shapes(raw_obs_space)

    spatial_shape = tuple(int(v) for v in spatial_shape)
    vector_dim = int(vector_dim)
    return lambda: MixedObservationEncoder(spatial_shape, vector_dim)


def _agent_spec_from_vec_env(vec_env):
    return SimpleNamespace(
        single_observation_space=vec_env.single_observation_space,
        single_action_space=_proxy_action_space(vec_env.single_action_space),
    )


def _dqn_agent_from_args(args, obs_dim, vec_env, encoder_factory=None):
    cfg = {
        "munchausen": True,
        "soft": True,
        "Beta": 0.7, # if args.ablation != 4 else 0.1
        "dueling": True,
        "distributional": True,
        "ent_reg_coef": 0.05,
        "delayed": True,
        "popart": True,
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
        cfg["ent_reg_coef"] = 0.005
    elif args.ablation == 5:
        cfg["delayed"] = False

    env_spec = get_env_benchmark_spec(args.env_name)
    if isinstance(vec_env.single_action_space, gym.spaces.Discrete):
        n_action_dims = 1
        n_action_bins = int(vec_env.single_action_space.n)
    else:
        n_action_dims = int(env_spec["n_action_dims"])
        n_action_bins = int(env_spec["n_action_bins"])
    hidden_layer_sizes = env_spec["hidden_layer_sizes"]

    AgentClass = RainbowDQN if cfg["distributional"] else EVRainbowDQN
    soft = bool(cfg["soft"])
    munchausen = bool(cfg["munchausen"])
    dueling = bool(cfg["dueling"])
    delayed = bool(cfg["delayed"])
    popart = bool(cfg["popart"])
    beta = float(cfg["Beta"])
    ent_reg_coef = float(cfg["ent_reg_coef"])
    tau = float(cfg["tau"])
    alpha = float(cfg["alpha"])
    if AgentClass is RainbowDQN:
        agent = AgentClass(
            obs_dim,
            n_action_dims,
            n_action_bins,
            hidden_layer_sizes=hidden_layer_sizes,
            soft=soft,
            munchausen=munchausen,
            Thompson=False,
            dueling=dueling,
            Beta=beta,
            ent_reg_coef=ent_reg_coef,
            delayed=delayed,
            popart=popart,
            tau=tau,
            polyak_tau=0.005,
            alpha=alpha,
            Beta_half_life_steps=50000,
            norm_obs=False,
            burn_in_updates=1000,
            encoder_factory=encoder_factory,
        )
    else:
        agent = AgentClass(
            obs_dim,
            n_action_dims,
            n_action_bins,
            hidden_layer_sizes=hidden_layer_sizes,
            soft=soft,
            munchausen=munchausen,
            Thompson=False,
            dueling=dueling,
            Beta=beta,
            ent_reg_coef=ent_reg_coef,
            delayed=delayed,
            popart=popart,
            tau=tau,
            polyak_tau=0.005,
            alpha=alpha,
            norm_obs=False,
            burn_in_updates=1000,
            encoder_factory=encoder_factory,
        )
    return agent, cfg


def _ppo_agent_from_args(args, vec_env, encoder_factory=None):
    cfg = {
        "clip_coef": 0.2,
        "ent_coef": 0.01,
        "Beta": 0.01,
        "distributional": True,
        "use_gae": True,
    }
    if args.ablation == 1:
        cfg["clip_coef"] = 100.0
    elif args.ablation == 2:
        cfg["ent_coef"] = 0.0
    elif args.ablation == 3:
        cfg["Beta"] = 0.0
    elif args.ablation == 4:
        cfg["distributional"] = False
    elif args.ablation == 5:
        cfg["use_gae"] = False

    hidden_layer_sizes = tuple(
        get_env_benchmark_spec(args.env_name)["hidden_layer_sizes"]
    )
    # num_minibatches scales with num_envs so that minibatch_size = num_steps
    # stays constant as num_envs changes. This keeps updates-per-individual-step
    # constant: update_epochs * num_minibatches / (num_steps * num_envs) = update_epochs / num_steps.
    agent = PPOAgent(
        vec_env,
        clip_coef=cfg["clip_coef"],
        ent_coef=cfg["ent_coef"],
        distributional=cfg["distributional"],
        Beta=cfg["Beta"],
        num_envs=int(vec_env.num_envs),
        num_steps=args.num_steps,
        num_minibatches=4,
        hidden_layer_sizes=hidden_layer_sizes,
        use_gae=cfg["use_gae"],
        encoder_factory=encoder_factory,
    )
    return agent, cfg


def _sac_agent_from_args(args, vec_env, encoder_factory=None):
    cfg = {
        "entropy_coef_zero": False,
        "distributional": True,
        "dueling": True,
        "popart": True,
        "delayed_critics": True,
        "munchausen": True,   # ablation 1 removes this
        "Beta": 0.01,         # ablation 3 sets this to 0.0
    }

    if args.ablation == 1:
        cfg["munchausen"] = False
    elif args.ablation == 2:
        cfg["entropy_coef_zero"] = True
    elif args.ablation == 3:
        cfg["Beta"] = 0.0
    elif args.ablation == 4:
        cfg["distributional"] = False
        cfg["dueling"] = False
    elif args.ablation == 5:
        cfg["delayed_critics"] = False

    hidden_layer_sizes = tuple(
        get_env_benchmark_spec(args.env_name)["hidden_layer_sizes"]
    )
    agent = SACAgent(
        _agent_spec_from_vec_env(vec_env),
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
        encoder_factory=encoder_factory,
        munchausen=cfg["munchausen"],
        beta_rnd=cfg["Beta"],
    )
    return agent, cfg


def build_agent(args, vec_env, device):
    obs_shape = vec_env.single_observation_space.shape
    if obs_shape is None:
        raise ValueError("Environment observation space shape is undefined")
    obs_dim = int(np.prod(obs_shape))
    encoder_factory = _encoder_factory_from_vec_env(args, vec_env)
    if args.algo == "dqn":
        agent, cfg = _dqn_agent_from_args(
            args,
            obs_dim,
            vec_env,
            encoder_factory=encoder_factory,
        )
    elif args.algo == "ppo":
        agent, cfg = _ppo_agent_from_args(
            args,
            vec_env,
            encoder_factory=encoder_factory,
        )
    elif args.algo == "sac":
        agent, cfg = _sac_agent_from_args(
            args,
            vec_env,
            encoder_factory=encoder_factory,
        )
    else:
        raise ValueError(f"Unsupported algo: {args.algo}")

    return agent.to(device), cfg


def build_buffer(args, vec_env, device):
    if args.algo != "sac":
        return None
    proxy_action_space = _proxy_action_space(vec_env.single_action_space)
    n_envs = int(getattr(vec_env, "num_envs", args.num_envs))
    return ReplayBuffer(
        args.buffer_size,
        vec_env.single_observation_space,
        proxy_action_space,
        device,
        n_envs=n_envs,
        handle_timeout_termination=False,
    )


def evaluate_agent(agent, args, device, step=0, n_steps=1000000):
    if args.env_name == "hide-and-seek-engine":
        vec_env = create_vec_env(args)
        try:
            env_spec = get_env_benchmark_spec(args.env_name)
            action_transform = ActionTransformHandler(
                args.env_name,
                args.algo,
                vec_env.single_action_space,
                bins_per_dim=int(getattr(args, "hide_seek_bins_per_dim", 3)),
                discrete_bins=int(env_spec["n_action_bins"]),
                batched=True,
            )
            obs, _ = vec_env.reset()
            done = np.zeros(int(vec_env.num_envs), dtype=bool)
            reward = np.zeros(int(vec_env.num_envs), dtype=np.float32)

            step_count = 0
            import time
            start_time = time.time()
            max_eval_time = 300  # 5 minutes, though usually this doesn't hang unless it's very slow

            while not bool(np.all(done)) and step_count < 2000 and (time.time() - start_time) < 120:
                step_count += 1
                with torch.no_grad():
                    if args.algo == "ppo":
                        tobs = torch.from_numpy(np.asarray(obs)).float().to(device)
                        action, _, _, _ = agent.sample_action(tobs)
                        raw_action = action.cpu().numpy()
                    elif args.algo == "dqn":
                        raw_action = agent.sample_action(
                            torch.from_numpy(np.asarray(obs)).float().to(device),
                            eps=0.0,
                            step=step,
                            n_steps=n_steps,
                        )
                    else:
                        raw_action = agent.sample_action(obs, deterministic=True)

                    step_action = action_transform.transform_action(raw_action)

                obs, r, term, trunc, _ = vec_env.step(step_action)
                reward += np.asarray(r, dtype=np.float32)
                done = np.logical_or(done, np.logical_or(term, trunc))

            return float(np.mean(reward))
        finally:
            vec_env.close()

    env = make_env_thunk(False, args.env_name)()
    obs, _ = env.reset()
    done = False
    reward = 0.0
    env_spec = get_env_benchmark_spec(args.env_name)
    action_transform = ActionTransformHandler(
        args.env_name,
        args.algo,
        env.action_space,
        bins_per_dim=int(getattr(args, "hide_seek_bins_per_dim", 3)),
        discrete_bins=int(env_spec["n_action_bins"]),
        batched=False,
    )

    import time
    start_time = time.time()
    step_count = 0

    while not done and step_count < 2000 and (time.time() - start_time) < 120:
        step_count += 1
        with torch.no_grad():
            if args.algo == "ppo":
                tobs = torch.from_numpy(np.asarray(obs)).float().unsqueeze(0).to(device)
                action, _, _, _ = agent.sample_action(tobs)
                raw_action = action.cpu().numpy()[0]
            elif args.algo == "dqn":
                raw_action = agent.sample_action(
                    torch.from_numpy(np.asarray(obs)).float().to(device),
                    eps=0.0,
                    step=step,
                    n_steps=n_steps,
                )
            else:
                raw_action = agent.sample_action(np.asarray(obs), deterministic=True)

            env_action = action_transform.transform_action(raw_action)

        obs, r, term, trunc, _ = env.step(env_action)
        reward += float(r)
        done = term or trunc

    env.close()
    return reward


def _compute_rollout_budget(args, total_steps_override):
    if total_steps_override is not None and total_steps_override > 0:
        return int(total_steps_override)
    return int(args.total_steps)


def rollout_online_rl(
    vec_env,
    agent,
    args,
    device,
    max_wall_time_seconds=None,
    total_steps_override=None,
):
    obs_raw, _ = vec_env.reset()
    env_spec = get_env_benchmark_spec(args.env_name)
    action_transform = ActionTransformHandler(
        args.env_name,
        "ppo",
        vec_env.single_action_space,
        bins_per_dim=int(getattr(args, "hide_seek_bins_per_dim", 3)),
        discrete_bins=int(env_spec["n_action_bins"]),
        batched=True,
    )

    total_step_budget = _compute_rollout_budget(args, total_steps_override)
    num_iterations = max(1, total_step_budget // (args.num_envs * args.num_steps))

    rhist = []
    smooth_rhist = []
    lhist = []
    eval_hist = []

    r_ep = np.zeros(args.num_envs)
    ep_len = np.zeros(args.num_envs, dtype=int)
    smooth_r = 0.0

    start_time = time.time()

    writer = None
    if getattr(args, "run", None) is not None:
        results_dir = os.path.join("results", args.algo, args.env_name)
        os.makedirs(results_dir, exist_ok=True)
        tb_dir = os.path.join(
            results_dir, f"tensorboard_run{args.run}_abl{args.ablation}"
        )
        writer = SummaryWriter(log_dir=tb_dir)
        writer.add_scalar("run/started", 1, 0)

        if hasattr(agent, "attach_tensorboard"):
            agent.attach_tensorboard(writer, prefix="agent")

    obs_shape = vec_env.single_observation_space.shape
    act_shape = vec_env.single_action_space.shape

    agent_obs = torch.zeros((args.num_steps, args.num_envs) + obs_shape).to(device)
    agent_actions = torch.zeros((args.num_steps, args.num_envs) + act_shape).to(device)
    agent_logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    agent_rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    agent_dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    agent_ext_values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    agent_int_values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    global_step = 0
    next_obs = torch.as_tensor(obs_raw, dtype=torch.float32, device=device)
    next_done = torch.zeros(args.num_envs, device=device)
    updates_performed = 0
    timed_out = False

    if max_wall_time_seconds is None:
        max_wall_time_seconds = getattr(args, "max_wall_time", 0.0)

    for iteration in range(1, num_iterations + 1):
        for step in range(args.num_steps):
            if global_step > 0 and global_step % 10000 == 0:
                print(f"[PPO] Step {global_step}/{total_step_budget} (Iteration {iteration}/{num_iterations})")
            if (
                max_wall_time_seconds is not None
                and max_wall_time_seconds > 0
                and (time.time() - start_time) >= max_wall_time_seconds
            ):
                timed_out = True
                break

            if global_step >= total_step_budget:
                timed_out = True
                break

            global_step += args.num_envs
            agent_obs[step] = next_obs
            agent_dones[step] = next_done

            with torch.no_grad():
                action, logprob, ext_v, int_v = agent.sample_action(next_obs)

            agent_ext_values[step] = ext_v
            agent_int_values[step] = int_v
            agent_actions[step] = action
            agent_logprobs[step] = logprob

            np_action = action.cpu().numpy()
            step_action = action_transform.transform_action(np_action)

            next_obs_np, reward, terminations, truncations, _ = vec_env.step(
                step_action
            )
            ep_len += 1
            truncations = np.logical_or(truncations, ep_len >= 2000)
            next_done_np = np.logical_or(terminations, truncations)
            agent_rewards[step] = torch.as_tensor(reward, device=device).view(-1)

            next_obs = torch.as_tensor(next_obs_np, dtype=torch.float32, device=device)
            next_done = torch.as_tensor(
                next_done_np, dtype=torch.float32, device=device
            )

            for env_i in range(args.num_envs):
                r_ep[env_i] += float(reward[env_i])
                if next_done_np[env_i]:
                    smooth_r = (
                        r_ep[env_i]
                        if smooth_r == 0.0
                        else 0.99 * smooth_r + 0.01 * r_ep[env_i]
                    )
                    rhist.append(float(r_ep[env_i]))
                    smooth_rhist.append(float(smooth_r))
                    if writer:
                        writer.add_scalar(
                            "charts/episodic_return", float(r_ep[env_i]), global_step
                        )
                    r_ep[env_i] = 0.0
                    ep_len[env_i] = 0

        if timed_out:
            break

        loss = agent.update(
            agent_obs,
            agent_actions,
            agent_logprobs,
            agent_rewards,
            agent_dones,
            agent_ext_values,
            agent_int_values,
            next_obs,
            next_done,
        )
        updates_performed += int(agent.update_epochs * agent.num_minibatches)
        lhist.append(float(loss))

        if iteration % 5 == 0:
            eval_r = evaluate_agent(
                agent,
                args,
                device,
                step=global_step,
                n_steps=total_step_budget,
            )
            eval_hist.append(float(eval_r))
            if writer:
                writer.add_scalar("charts/eval_return", float(eval_r), global_step)

    train_time = time.time() - start_time
    steps_per_sec = (global_step / train_time) if train_time > 0 else 0.0
    updates_per_sec = (updates_performed / train_time) if train_time > 0 else 0.0

    if writer:
        writer.flush()
        writer.close()

    return {
        "rhist": rhist,
        "smooth_rhist": smooth_rhist,
        "lhist": lhist,
        "eval_hist": eval_hist,
        "steps_run": int(global_step),
        "updates_performed": int(updates_performed),
        "steps_per_sec": float(steps_per_sec),
        "updates_per_sec": float(updates_per_sec),
        "timed_out": bool(timed_out),
        "train_time": float(train_time),
    }


def rollout_offline_rl(
    vec_env,
    agent,
    args,
    device,
    buffer,
    max_wall_time_seconds=None,
    total_steps_override=None,
):
    obs, _ = vec_env.reset()
    total_step_budget = _compute_rollout_budget(args, total_steps_override)

    time_taken_modular = {
        "action_sample": 0.0,
        "env_step": 0.0,
        "update_agent": 0.0,
        "eval_agent": 0.0,
    }
    rhist = []
    smooth_rhist = []
    lhist = []
    eval_hist = []

    r_ep = np.zeros(args.num_envs)
    ep_len = np.zeros(args.num_envs, dtype=int)
    smooth_r = 0.0
    ep = 0

    start_time = time.time()

    writer = None
    if getattr(args, "run", None) is not None:
        results_dir = os.path.join("results", args.algo, args.env_name)
        os.makedirs(results_dir, exist_ok=True)
        tb_dir = os.path.join(
            results_dir, f"tensorboard_run{args.run}_abl{args.ablation}"
        )
        writer = SummaryWriter(log_dir=tb_dir)
        writer.add_scalar("run/started", 1, 0)

        if hasattr(agent, "attach_tensorboard"):
            agent.attach_tensorboard(writer, prefix="agent")

    total_samples = 0
    updates_performed = 0
    steps_since_update = 0
    timed_out = False

    if max_wall_time_seconds is None:
        max_wall_time_seconds = getattr(args, "max_wall_time", 0.0)

    if args.algo == "dqn":
        obs_dim = int(np.prod(vec_env.single_observation_space.shape))
        env_spec = get_env_benchmark_spec(args.env_name)
        n_action_dims = int(env_spec["n_action_dims"])
        dqn_action_transform = ActionTransformHandler(
            args.env_name,
            "dqn",
            vec_env.single_action_space,
            bins_per_dim=int(getattr(args, "hide_seek_bins_per_dim", 3)),
            discrete_bins=int(env_spec["n_action_bins"]),
            batched=True,
        )

        blen = int(args.dqn_buffer_size)
        batch_size = int(args.dqn_batch_size)
        rnd_burn_in = int(args.rnd_burn_in)

        if n_action_dims > 1:
            buff_actions = torch.zeros(
                (blen, n_action_dims), dtype=torch.long, device=device
            )
        else:
            buff_actions = torch.zeros((blen,), dtype=torch.long, device=device)
        buff_obs = torch.zeros((blen, obs_dim), dtype=torch.float32, device=device)
        buff_next_obs = torch.zeros((blen, obs_dim), dtype=torch.float32, device=device)
        buff_term = torch.zeros((blen,), dtype=torch.float32, device=device)
        buff_r = torch.zeros((blen,), dtype=torch.float32, device=device)

        buffer_ptr = 0

        while total_samples < total_step_budget:
            if total_samples > 0 and total_samples % 10000 == 0:
                print(f"[SAC] Step {total_samples}/{total_step_budget} (Episodes: {ep})")
            if (
                max_wall_time_seconds is not None
                and max_wall_time_seconds > 0
                and (time.time() - start_time) >= max_wall_time_seconds
            ):
                timed_out = True
                break

            eps_current = max(
                1.0 - 2.0 * (total_samples / max(1, total_step_budget)), 0.05
            )
            tobs = torch.from_numpy(obs).to(device).float()
            t_ = time.time()
            actions = agent.sample_action(
                tobs,
                eps=eps_current,
                step=total_samples,
                n_steps=total_step_budget,
            )
            time_taken_modular["action_sample"] += time.time() - t_
            step_action = dqn_action_transform.transform_action(actions)

            t_ = time.time()
            next_obs, r, term, trunc, _ = vec_env.step(step_action)
            time_taken_modular["env_step"] += time.time() - t_

            for env_i in range(args.num_envs):
                if ep_len[env_i] + 1 >= 2000:
                    trunc[env_i] = True

            r_mult = r * 10.0

            if hasattr(agent, "update_running_stats"):
                agent.update_running_stats(
                    torch.from_numpy(next_obs).to(device).float(),
                    torch.as_tensor(r_mult, dtype=torch.float32, device=device),
                )

            actions_arr = np.asarray(actions)
            for env_i in range(args.num_envs):
                idx = buffer_ptr % blen
                if n_action_dims > 1:
                    buff_actions[idx] = torch.as_tensor(
                        actions_arr[env_i], dtype=torch.long, device=device
                    )
                else:
                    buff_actions[idx] = int(
                        np.asarray(actions_arr[env_i]).reshape(-1)[0]
                    )

                buff_obs[idx].copy_(torch.from_numpy(obs[env_i]).to(device).float())
                buff_next_obs[idx].copy_(
                    torch.from_numpy(next_obs[env_i]).to(device).float()
                )
                buff_term[idx] = float(term[env_i] or trunc[env_i])
                buff_r[idx] = float(r_mult[env_i])

                buffer_ptr += 1
                total_samples += 1
                r_ep[env_i] += float(r_mult[env_i])
                ep_len[env_i] += 1

                if term[env_i] or trunc[env_i]:
                    rhist.append(float(r_ep[env_i]))
                    if len(rhist) < 20:
                        smooth_r = float(sum(rhist) / len(rhist))
                    else:
                        smooth_r = float(0.05 * rhist[-1] + 0.95 * smooth_r)
                    smooth_rhist.append(float(smooth_r))
                    ep += 1

                    if writer:
                        writer.add_scalar(
                            "episode/reward", float(rhist[-1]), total_samples
                        )
                        writer.add_scalar(
                            "episode/smooth_reward", float(smooth_r), total_samples
                        )

                    t_ = time.time()
                    if ep % 50 == 0 and total_samples > total_step_budget // 2:
                        evalr = 0.0
                        for _ in range(5):
                            evalr += evaluate_agent(
                                agent,
                                args,
                                device,
                                step=total_samples,
                                n_steps=total_step_budget,
                            )
                        eval_hist.append(float(evalr / 5.0))
                        if writer:
                            writer.add_scalar(
                                "eval/reward", float(eval_hist[-1]), total_samples
                            )
                    time_taken_modular["eval_agent"] += time.time() - t_

                    ep_len[env_i] = 0
                    r_ep[env_i] = 0.0

            obs = next_obs
            steps_since_update += args.num_envs

            t_ = time.time()
            while steps_since_update >= args.update_every:
                if total_samples > batch_size:
                    if total_samples < rnd_burn_in:
                        agent.update(
                            buff_obs,
                            buff_actions,
                            buff_r,
                            buff_next_obs,
                            buff_term,
                            batch_size=batch_size,
                            step=min(total_samples, blen),
                        )
                        updates_performed += 1
                    else:
                        loss_val = agent.update(
                            buff_obs,
                            buff_actions,
                            buff_r,
                            buff_next_obs,
                            buff_term,
                            batch_size=batch_size,
                            step=min(total_samples, blen),
                        )
                        lhist.append(float(loss_val))
                        updates_performed += 1
                        agent.update_target()
                steps_since_update -= args.update_every
            time_taken_modular["update_agent"] += time.time() - t_

    else:
        if buffer is None:
            raise ValueError("rollout_offline_rl requires a replay buffer for SAC")

        proxy_action_space = _proxy_action_space(vec_env.single_action_space)
        proxy_action_dim = int(np.prod(proxy_action_space.shape))
        env_spec = get_env_benchmark_spec(args.env_name)
        sac_action_transform = ActionTransformHandler(
            args.env_name,
            "sac",
            vec_env.single_action_space,
            bins_per_dim=int(getattr(args, "hide_seek_bins_per_dim", 3)),
            discrete_bins=int(env_spec["n_action_bins"]),
            batched=True,
        )
        eval_every_episodes = 25

        while total_samples < total_step_budget:
            if (
                max_wall_time_seconds is not None
                and max_wall_time_seconds > 0
                and (time.time() - start_time) >= max_wall_time_seconds
            ):
                timed_out = True
                break

            if total_samples < args.learning_starts:
                if isinstance(vec_env.single_action_space, gym.spaces.Box):
                    actions_agent = np.array(
                        [
                            vec_env.single_action_space.sample()
                            for _ in range(vec_env.num_envs)
                        ],
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

            actions_env = sac_action_transform.transform_action(actions_agent)
            next_obs, rewards, terminations, truncations, infos = vec_env.step(
                actions_env
            )

            for env_i in range(args.num_envs):
                if ep_len[env_i] + 1 >= 2000:
                    truncations[env_i] = True

            real_next_obs = next_obs.copy()
            for idx, trunc in enumerate(truncations):
                if trunc and "final_observation" in infos:
                    real_next_obs[idx] = infos["final_observation"][idx]
            buffer.add(obs, real_next_obs, actions_agent, rewards, terminations, infos)

            for env_i in range(args.num_envs):
                r_ep[env_i] += float(rewards[env_i])
                ep_len[env_i] += 1
                if terminations[env_i] or truncations[env_i]:
                    rhist.append(float(r_ep[env_i]))
                    if len(rhist) < 20:
                        smooth_r = float(sum(rhist) / len(rhist))
                    else:
                        smooth_r = float(0.05 * rhist[-1] + 0.95 * smooth_r)
                    smooth_rhist.append(float(smooth_r))
                    ep += 1
                    if writer:
                        writer.add_scalar(
                            "episode/reward", float(rhist[-1]), total_samples
                        )
                        writer.add_scalar(
                            "episode/smooth_reward", float(smooth_r), total_samples
                        )

                    if (
                        ep % eval_every_episodes == 0
                        and total_samples > total_step_budget // 3
                    ):
                        evalr = evaluate_agent(
                            agent,
                            args,
                            device,
                            step=total_samples,
                            n_steps=total_step_budget,
                        )
                        eval_hist.append(float(evalr))
                        if writer:
                            writer.add_scalar(
                                "eval/reward", float(evalr), total_samples
                            )

                    ep_len[env_i] = 0
                    r_ep[env_i] = 0.0

            if hasattr(agent, "update_running_stats"):
                agent.update_running_stats(
                    torch.from_numpy(next_obs).to(device).float(),
                    torch.as_tensor(rewards, dtype=torch.float32, device=device),
                )

            obs = next_obs
            total_samples += args.num_envs
            steps_since_update += args.num_envs

            # buffer.size() returns timestep positions (not individual transitions).
            # We need batch_size individual transitions, which requires
            # ceil(batch_size / n_envs) timesteps.
            _n_envs_buf = int(getattr(buffer, "n_envs", args.num_envs))
            _min_buf_timesteps = max(1, args.batch_size // _n_envs_buf)
            while steps_since_update >= args.update_every:
                if (
                    total_samples > args.learning_starts
                    and buffer.size() >= _min_buf_timesteps
                ):
                    data = buffer.sample(args.batch_size)
                    loss = agent.update(data, global_step=total_samples)
                    lhist.append(float(loss))
                    updates_performed += 1
                steps_since_update -= args.update_every

    if writer:
        writer.flush()
        writer.close()

    train_time = time.time() - start_time
    steps_per_sec = (total_samples / train_time) if train_time > 0 else 0.0
    updates_per_sec = (updates_performed / train_time) if train_time > 0 else 0.0

    return {
        "rhist": rhist,
        "smooth_rhist": smooth_rhist,
        "lhist": lhist,
        "eval_hist": eval_hist,
        "steps_run": int(total_samples),
        "updates_performed": int(updates_performed),
        "steps_per_sec": float(steps_per_sec),
        "updates_per_sec": float(updates_per_sec),
        "timed_out": bool(timed_out),
        "train_time": float(train_time),
        "time_taken_modular": time_taken_modular,
    }


def main():
    args = get_args()
    device = resolve_torch_device(args.device)
    print(f"Args: {args}")

    vec_env = create_vec_env(args)
    if args.algo == "ppo":
        args.num_steps = max(1, args.num_steps // args.num_envs)
    try:
        agent, _ = build_agent(args, vec_env, device)
        buffer = build_buffer(args, vec_env, device)

        max_wall = args.max_wall_time if args.max_wall_time > 0 else None
        step_override = (
            args.total_steps_override if args.total_steps_override > 0 else None
        )

        if args.algo == "ppo":
            results = rollout_online_rl(
                vec_env,
                agent,
                args,
                device,
                max_wall_time_seconds=max_wall,
                total_steps_override=step_override,
            )
        else:
            results = rollout_offline_rl(
                vec_env,
                agent,
                args,
                device,
                buffer,
                max_wall_time_seconds=max_wall,
                total_steps_override=step_override,
            )
            if "time_taken_modular" in results:
                print(f"Time taken modular {results['time_taken_modular']}")

        plot_results(results, args, args.algo)
    finally:
        vec_env.close()


if __name__ == "__main__":
    main()
