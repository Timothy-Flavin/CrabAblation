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
from runner_utils import get_device_name, plot_results, resolve_torch_device
from environment_utils import (
    ActionTransformHandler,
    _proxy_action_space,
    extract_mixed_observation_shapes,
    make_hide_and_seek_vec_env,
    make_env_thunk,
)
from learning_algorithms.cleanrl_buffers import ReplayBuffer
from learning_algorithms.DQN_Rainbow import EVRainbowDQN, IQNRainbowDQN
from learning_algorithms.PG_Rainbow import StandardPPOAgent, DistributionalPPOAgent
from learning_algorithms.SAC_Rainbow import DistSAC, EVSAC
import yaml

# Load environment config from YAML
with open("env_config.yaml", "r") as f:
    ENV_CONFIG = yaml.safe_load(f)


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
    parser.add_argument(
        "--ablation", type=int, default=0, choices=[0, 1, 2, 3, 4, 5, 6]
    )
    parser.add_argument("--fully_obs", action="store_true")
    parser.add_argument("--run", type=int, default=1)
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--num_steps", type=int, default=2048)
    parser.add_argument("--total_steps", type=int, default=None)
    parser.add_argument(
        "--total_steps_override",
        type=int,
        default=0,
        help="If > 0, force this total step budget for rollouts",
    )
    
    # Updated default to 86400 seconds (24 hours)
    parser.add_argument(
        "--max_wall_time",
        type=float,
        default=86400.0,
        help="Maximum wall-clock seconds before early stop",
    )
    parser.add_argument("--device_name", type=str, default=get_device_name())
    parser.add_argument(
        "--skip_best_params",
        action="store_true",
        help="Do not load best grid-search params for device/num_envs",
    )

    # DQN knobs
    parser.add_argument("--dqn_buffer_size", type=int, default=2e4)
    parser.add_argument("--dqn_batch_size", type=int, default=64)
    parser.add_argument("--update_every", type=int, default=4)
    parser.add_argument("--rnd_burn_in", type=int, default=1000)

    # SAC knobs
    parser.add_argument("--buffer_size", type=int, default=2e4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.001 * 0.003)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--learning_starts", type=int, default=10000)
    parser.add_argument("--policy_lr", type=float, default=3e-4)
    parser.add_argument("--q_lr", type=float, default=1e-3)
    parser.add_argument("--policy_frequency", type=int, default=4)
    parser.add_argument("--target_network_frequency", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=0.001)
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

    # Load config for this environment
    env_cfg = ENV_CONFIG.get(args.env_name, {})
    # Set max_steps/total_steps from config if not overridden
    max_steps_cfg = env_cfg.get("max_steps", 1000000)
    if args.total_steps is None:
        if isinstance(max_steps_cfg, dict):
            args.total_steps = int(max_steps_cfg.get(args.algo, 1000000))
        else:
            args.total_steps = int(max_steps_cfg)
            
    # Set buffer sizes from config if not overridden
    if args.dqn_buffer_size is None:
        args.dqn_buffer_size = int(env_cfg.get("buffer_size", 100000))
    if args.buffer_size is None:
        args.buffer_size = int(env_cfg.get("buffer_size", 200000))

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
        vec_env = make_hide_and_seek_vec_env(
            num_envs=int(num_envs if num_envs is not None else args.num_envs),
            device="cpu",
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
        num_envs=vec_env.num_envs,
    )


def _dqn_agent_from_args(args, obs_dim, vec_env, encoder_factory=None):
    env_cfg = ENV_CONFIG.get(args.env_name, {})
    if isinstance(vec_env.single_action_space, gym.spaces.Discrete):
        n_action_dims = int(env_cfg.get("n_action_dims", 1))
        n_action_bins = int(env_cfg.get("n_action_bins", vec_env.single_action_space.n))
    else:
        n_action_dims = int(env_cfg.get("n_action_dims", 1))
        n_action_bins = int(env_cfg.get("n_action_bins", 2))
    hidden_layer_sizes = env_cfg.get("hidden_layer_sizes", [256, 256])
    # n_action_bins = 5
    # input(
    #     f"n action dims: {n_action_dims}, n action bins: {n_action_bins}. Press Enter to continue..."
    # )
    # Calculate beta decay schedule
    total_steps = int(getattr(args, "total_steps", 1000000))
    update_every = int(getattr(args, "update_every", 2))
    beta_half_life_steps = max(1, (total_steps // update_every) // 5)
    cfg = {
        "munchausen_constant": 0.9,
        "soft": True,
        "Beta": 1.0,  # Start fully intrinsic
        "dueling": True,
        "distributional": True,
        "delayed": True,
        "popart": True,
        "tau": 0.05,
        "alpha": 0.01,
        "beta_half_life_steps": beta_half_life_steps,
    }

    if args.ablation == 1:
        cfg["munchausen_constant"] = 0.0
        cfg["soft"] = False
    elif args.ablation == 2:
        cfg["soft"] = False
    elif args.ablation == 3:
        cfg["Beta"] = 0.0
    elif args.ablation == 4:
        cfg["distributional"] = False
        cfg["dueling"] = False
    elif args.ablation == 5:
        cfg["delayed"] = False
    elif args.ablation == 6:
        cfg["munchausen_constant"] = 0.9
        cfg["soft"] = True
        cfg["Beta"] = 0.0
        cfg["distributional"] = False
        cfg["delayed"] = True
        cfg["dueling"] = False

    AgentClass = IQNRainbowDQN if cfg["distributional"] else EVRainbowDQN
    soft = bool(cfg["soft"])
    dueling = bool(cfg["dueling"])
    delayed = bool(cfg["delayed"])
    beta = float(cfg["Beta"])
    ent_reg_coef = float(cfg["ent_reg_coef"])
    alpha = float(cfg["alpha"])
    if AgentClass is IQNRainbowDQN:
        agent = AgentClass(
            obs_dim,
            n_action_dims,
            n_action_bins,
            n_envs=int(vec_env.num_envs),
            buffer_size=int(args.dqn_buffer_size),
            hidden_layer_sizes=hidden_layer_sizes,
            soft=soft,
            munchausen_constant=cfg["munchausen_constant"],
            Thompson=False,
            dueling=dueling,
            Beta=beta,
            ent_reg_coef=ent_reg_coef,
            delayed=delayed,
            polyak_tau=0.005,
            alpha=alpha,
            beta_half_life_steps=cfg["beta_half_life_steps"],
            norm_obs=False,
            burn_in_updates=1000,
            encoder_factory=encoder_factory,
        )
    else:
        agent = AgentClass(
            obs_dim,
            n_action_dims,
            n_action_bins,
            n_envs=int(vec_env.num_envs),
            buffer_size=int(args.dqn_buffer_size),
            hidden_layer_sizes=hidden_layer_sizes,
            soft=soft,
            munchausen_constant=cfg["munchausen_constant"],
            Thompson=False,
            dueling=dueling,
            Beta=beta,
            ent_reg_coef=ent_reg_coef,
            delayed=delayed,
            polyak_tau=0.005,
            alpha=alpha,
            beta_half_life_steps=cfg["beta_half_life_steps"],
            norm_obs=False,
            burn_in_updates=1000,
            encoder_factory=encoder_factory,
        )
    return agent, cfg


def _ppo_agent_from_args(args, vec_env, encoder_factory=None):
    env_cfg = ENV_CONFIG.get(args.env_name, {})
    hidden_layer_sizes = tuple(env_cfg.get("hidden_layer_sizes", [128, 128]))
    # Calculate beta decay schedule
    total_steps = int(getattr(args, "total_steps", 1000000))
    update_every = 1
    beta_half_life_steps = max(1, (total_steps) // 5)
    cfg = {
        "clip_coef": 0.2,
        "ent_coef": 0.01,
        "Beta": 1.0,  # Start fully intrinsic
        "distributional": True,
        "use_gae": True,
        "beta_half_life_steps": beta_half_life_steps,
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
    elif args.ablation == 6:
        cfg["clip_coef"] = 0.2
        cfg["ent_coef"] = 0.001
        cfg["Beta"] = 0.0
        cfg["distributional"] = False
        cfg["use_gae"] = True

    rollout_steps = max(1, args.num_steps // int(vec_env.num_envs))
    AgentClass = DistributionalPPOAgent if cfg["distributional"] else StandardPPOAgent
    agent = AgentClass(
        vec_env,
        clip_coef=cfg["clip_coef"],
        ent_coef=cfg["ent_coef"],
        Beta=cfg["Beta"],
        num_envs=int(vec_env.num_envs),
        num_steps=rollout_steps,
        num_minibatches=4,
        hidden_layer_sizes=hidden_layer_sizes,
        use_gae=cfg["use_gae"],
        encoder_factory=encoder_factory,
        beta_half_life_steps=cfg["beta_half_life_steps"],
    )
    return agent, cfg


def _sac_agent_from_args(args, vec_env, encoder_factory=None):
    env_cfg = ENV_CONFIG.get(args.env_name, {})
    hidden_layer_sizes = tuple(env_cfg.get("hidden_layer_sizes", [128, 128]))
    # Calculate beta decay schedule
    total_steps = int(getattr(args, "total_steps", 1000000))
    update_every = int(getattr(args, "update_every", 4))
    beta_half_life_steps = max(1, (total_steps // update_every) // 5)
    cfg = {
        "entropy_coef_zero": False,
        "distributional": True,
        "popart": True,
        "delayed_critics": True,
        "munchausen": True,  # ablation 1 removes this
        "munchausen_constant": 0.5,  # ablation 1 removes this
        "Beta": 1.0,  # Start fully intrinsic
        "beta_half_life_steps": beta_half_life_steps,
    }

    if args.ablation == 1:
        cfg["munchausen"] = False
        cfg["munchausen_constant"] = 0.0
    elif args.ablation == 2:
        cfg["munchausen"] = False
        cfg["munchausen_constant"] = 0.0
        cfg["entropy_coef_zero"] = True
    elif args.ablation == 3:
        cfg["Beta"] = 0.0
    elif args.ablation == 4:
        cfg["distributional"] = False
    elif args.ablation == 5:
        cfg["delayed_critics"] = False
    elif args.ablation == 6:
        cfg["munchausen"] = False
        cfg["entropy_coef_zero"] = False
        cfg["Beta"] = 0.0
        cfg["distributional"] = False
        cfg["delayed_critics"] = True

    AgentClass = DistSAC if cfg["distributional"] else EVSAC
    agent = AgentClass(
        _agent_spec_from_vec_env(vec_env),
        gamma=args.gamma,
        tau=0.005,  # args.tau,
        policy_lr=args.policy_lr,
        q_lr=args.q_lr,
        policy_frequency=args.policy_frequency,
        target_network_frequency=args.target_network_frequency,
        alpha=args.alpha,
        autotune=args.autotune,
        entropy_coef_zero=cfg["entropy_coef_zero"],
        delayed_critics=cfg["delayed_critics"],
        hidden_layer_sizes=hidden_layer_sizes,
        n_quantiles=args.n_quantiles,
        n_target_quantiles=args.n_target_quantiles,
        encoder_factory=encoder_factory,
        munchausen=cfg["munchausen"],
        munchausen_constant=0.5,
        beta_rnd=cfg["Beta"],
        beta_half_life_steps=cfg["beta_half_life_steps"],
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
    env_cfg = ENV_CONFIG.get(args.env_name, {})
    if args.env_name == "hide-and-seek-engine":
        vec_env = create_vec_env(args)
        try:
            action_transform = ActionTransformHandler(
                args.env_name,
                args.algo,
                vec_env.single_action_space,
                bins_per_dim=int(getattr(args, "hide_seek_bins_per_dim", 3)),
                discrete_bins=int(env_cfg.get("n_action_bins", 2)),
                batched=True,
            )
            obs, _ = vec_env.reset()
            done = np.zeros(int(vec_env.num_envs), dtype=bool)
            reward = np.zeros(int(vec_env.num_envs), dtype=np.float32)

            step_count = 0
            import time

            start_time = time.time()
            max_eval_time = (
                300  # 5 minutes, though usually this doesn't hang unless it's very slow
            )

            while (
                not bool(np.all(done))
                and step_count < 2000
                and (time.time() - start_time) < 120
            ):
                step_count += 1
                with torch.no_grad():
                    if args.algo == "ppo":
                        tobs = torch.from_numpy(np.asarray(obs)).float().to(device)
                        action, _ = agent.sample_action(tobs)
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
    action_transform = ActionTransformHandler(
        args.env_name,
        args.algo,
        env.action_space,
        bins_per_dim=int(getattr(args, "hide_seek_bins_per_dim", 3)),
        discrete_bins=int(env_cfg.get("n_action_bins", 2)),
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
                action, _ = agent.sample_action(tobs)
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
    obs, _ = vec_env.reset()
    env_cfg = ENV_CONFIG.get(args.env_name, {})
    action_transform = ActionTransformHandler(
        args.env_name,
        "ppo",
        vec_env.single_action_space,
        bins_per_dim=int(getattr(args, "hide_seek_bins_per_dim", 3)),
        discrete_bins=int(env_cfg.get("n_action_bins", 2)),
        batched=True,
    )

    total_step_budget = _compute_rollout_budget(args, total_steps_override)

    rhist = []
    smooth_rhist = []
    lhist = []
    eval_hist = []

    r_ep = np.zeros(args.num_envs)
    ep_len = np.zeros(args.num_envs, dtype=int)
    smooth_r = 0.0
    ep = 0

    start_time = time.time()

    global_step = 0
    updates_performed = 0
    timed_out = False

    eval_every_episodes = getattr(args, "eval_every_episodes", 25)

    if max_wall_time_seconds is None:
        max_wall_time_seconds = getattr(args, "max_wall_time", 0.0)
    max_time = max_wall_time_seconds if (max_wall_time_seconds is not None and max_wall_time_seconds > 0) else float('inf')

    while global_step < total_step_budget:
        time_elapsed = time.time() - start_time
        if time_elapsed >= max_time:
            timed_out = True
            break
        if global_step > 0 and global_step % 10000 == 0:
            print(
                f"[PPO] Step {global_step}/{total_step_budget} beta {agent.Beta} smooth r {smooth_r:.2f}"
            )

        if (
            max_wall_time_seconds is not None
            and max_wall_time_seconds > 0
            and (time.time() - start_time) >= max_wall_time_seconds
        ):
            timed_out = True
            break

        with torch.no_grad():
            tobs = torch.as_tensor(obs, dtype=torch.float32, device=device)
            # Ext/Int values are no longer needed here since we batch them in update()
            action, logprob = agent.sample_action(tobs)

        np_action = action.cpu().numpy()
        step_action = action_transform.transform_action(np_action)

        next_obs, reward, terminations, truncations, infos = vec_env.step(step_action)
        ep_len += 1

        # Funnel strictly into observe. Observe now handles the final_observation extraction.
        agent.observe(
            obs,
            action,
            logprob,
            reward,
            next_obs,
            terminations,
            truncations,
            infos,  # Pass infos so the agent can catch truncations
        )

        for env_i in range(args.num_envs):
            r_ep[env_i] += float(reward[env_i])
            if terminations[env_i] or truncations[env_i]:
                smooth_r = (
                    r_ep[env_i]
                    if smooth_r == 0.0
                    else 0.99 * smooth_r + 0.01 * r_ep[env_i]
                )
                rhist.append(float(r_ep[env_i]))
                smooth_rhist.append(float(smooth_r))
                ep += 1

                if ep % eval_every_episodes == 0 and ep > 0:
                    eval_res = evaluate_agent(
                        agent, args, device, step=global_step, n_steps=total_step_budget
                    )
                    eval_hist.append(eval_res)

                r_ep[env_i] = 0.0
                ep_len[env_i] = 0

        obs = next_obs
        global_step += args.num_envs

        if agent.step_idx >= agent.num_steps:
            loss = agent.update(global_step=global_step)
            if loss is not None:
                lhist.append(float(loss))
                updates_performed += int(agent.update_epochs * agent.num_minibatches)

    train_time = time.time() - start_time
    steps_per_sec = (global_step / train_time) if train_time > 0 else 0.0
    updates_per_sec = (updates_performed / train_time) if train_time > 0 else 0.0

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
    max_wall_time_seconds=None,
    total_steps_override=None,
):
    import time

    start_time = time.time()
    total_samples = 0

    if args.algo == "dqn":
        batch_size = int(args.dqn_batch_size)
    else:
        batch_size = int(args.batch_size)

    total_step_budget = _compute_rollout_budget(args, total_steps_override)
    
    # Resolve the hard time cap
    max_time = max_wall_time_seconds if (max_wall_time_seconds is not None and max_wall_time_seconds > 0) else float('inf')

    rhist = []
    smooth_rhist = []
    lhist = []
    eval_hist = []

    r_ep = np.zeros(args.num_envs)
    ep_len = np.zeros(args.num_envs, dtype=int)
    smooth_r = 0.0
    ep = 0

    time_taken_modular = {
        "action_sample": 0.0,
        "env_step": 0.0,
        "update_agent": 0.0,
        "eval_agent": 0.0,
    }

    obs, _ = vec_env.reset()
    steps_since_update = 0
    updates_performed = 0

    env_cfg = ENV_CONFIG.get(args.env_name, {})
    action_transform = ActionTransformHandler(
        args.env_name,
        args.algo,
        vec_env.single_action_space,
        bins_per_dim=int(getattr(args, "hide_seek_bins_per_dim", 3)),
        discrete_bins=int(env_cfg.get("n_action_bins", 2)),
        batched=True,
    )

    proxy_action_space = _proxy_action_space(vec_env.single_action_space)
    proxy_action_dim = int(np.prod(proxy_action_space.shape))

    eval_every_episodes = 25

    while total_samples < total_step_budget:
        # Check time bounds and progress
        time_elapsed = time.time() - start_time
        time_progress = time_elapsed / max_time if max_time != float('inf') else 0.0
        step_progress = total_samples / max(1, total_step_budget)
        
        # Ensures parameters scale accurately if time finishes before steps
        progress = min(1.0, max(time_progress, step_progress))

        if time_elapsed >= max_time:
            break

        if total_samples > 0 and total_samples % 10000 == 0:
            print(
                f"[{args.algo.upper()}] Step {total_samples}/{total_step_budget} (Episodes: {ep}) smooth r {smooth_r}"
            )

        t_ = time.time()

        # Sample action using PROGRESS fraction instead of raw steps
        if args.algo == "dqn":
            eps_current = max(0.5 - 2.0 * progress, 0.05)
            tobs = torch.as_tensor(obs, dtype=torch.float32, device=device)
            # Pass the effective progression step artificially so agent scales properly internally
            effective_step = int(progress * total_step_budget)
            actions = agent.sample_action(
                tobs, eps=eps_current, step=effective_step, n_steps=total_step_budget
            )
        else:  # SAC
            learning_starts_val = getattr(args, "learning_starts", 5000)
            ls_fraction = learning_starts_val / max(1, total_step_budget)
            
            # Use progress fraction to evaluate learning start boundaries
            if progress < ls_fraction:
                if isinstance(vec_env.single_action_space, gym.spaces.Box):
                    actions = np.array(
                        [
                            vec_env.single_action_space.sample()
                            for _ in range(vec_env.num_envs)
                        ],
                        dtype=np.float32,
                    )
                else:
                    actions = np.random.uniform(
                        low=0.0, high=1.0, size=(vec_env.num_envs, proxy_action_dim)
                    ).astype(np.float32)
            else:
                actions = agent.sample_action(
                    torch.as_tensor(obs, dtype=torch.float32, device=device)
                )

        time_taken_modular["action_sample"] += time.time() - t_

        step_action = action_transform.transform_action(actions)

        t_ = time.time()
        next_obs, rewards, terminations, truncations, infos = vec_env.step(step_action)
        time_taken_modular["env_step"] += time.time() - t_

        real_next_obs = (
            next_obs.clone() if isinstance(next_obs, torch.Tensor) else next_obs.copy()
        )

        if "final_observation" in infos:
            _final_masks = infos.get(
                "_final_observation", np.logical_or(terminations, truncations)
            )
            for idx, is_final in enumerate(_final_masks):
                if is_final:
                    real_next_obs[idx] = infos["final_observation"][idx]

        for env_i in range(args.num_envs):
            if ep_len[env_i] + 1 >= getattr(args, "max_frames_per_ep", 2000):
                truncations[env_i] = True

        actions_arr = np.asarray(actions)
        if args.algo == "dqn" and actions_arr.ndim == 1:
            actions_arr = actions_arr.reshape(-1, 1)

        agent.observe(
            obs, actions_arr, rewards, real_next_obs, terminations, truncations, infos
        )

        for env_i in range(args.num_envs):
            total_samples += 1
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

                if ep % eval_every_episodes == 0 and ep > 0:
                    effective_step = int(progress * total_step_budget)
                    eval_res = evaluate_agent(agent, args, device, step=effective_step, n_steps=total_step_budget)
                    eval_hist.append(eval_res)

                r_ep[env_i] = 0.0
                ep_len[env_i] = 0

        obs = next_obs

        # Update Agent
        steps_since_update += args.num_envs
        t_ = time.time()
        effective_step = int(progress * total_step_budget)
        if args.algo == "dqn":
            rnd_burn_in = getattr(args, "rnd_burn_in", 0)
            rnd_fraction = rnd_burn_in / max(1, total_step_budget)
            
            while steps_since_update >= args.update_every:
                if total_samples > batch_size:
                    loss_val = agent.update(batch_size=batch_size, step=effective_step)
                    if progress >= rnd_fraction:
                        lhist.append(float(loss_val))
                    updates_performed += 1
                steps_since_update -= args.update_every
        else:  # SAC
            while steps_since_update >= args.update_every:
                loss_val = agent.update(
                    batch_size=batch_size, global_step=effective_step
                )
                if loss_val is not None:
                    try:
                        lhist.append(
                            float(
                                loss_val[0] if isinstance(loss_val, tuple) else loss_val
                            )
                        )
                    except:
                        pass
                updates_performed += 1
                steps_since_update -= args.update_every
        time_taken_modular["update_agent"] += time.time() - t_

    train_time = time.time() - start_time
    steps_per_sec = total_samples / (train_time if train_time > 0 else 1)
    updates_per_sec = updates_performed / (train_time if train_time > 0 else 1)

    return {
        "final_model": agent,
        "rhist": rhist,
        "smooth_rhist": smooth_rhist,
        "lhist": lhist,
        "eval_hist": eval_hist,
        "steps_run": int(total_samples),
        "updates_performed": int(updates_performed),
        "steps_per_sec": float(steps_per_sec),
        "updates_per_sec": float(updates_per_sec),
        "train_time": float(train_time),
        "timed_out": (time_elapsed >= max_time),
        "time_taken_modular": time_taken_modular,
    }

def main():
    args = get_args()
    device = resolve_torch_device(args.device)
    print(f"Args: {args}")

    vec_env = create_vec_env(args)
    if args.algo == "ppo":
        pass
    try:
        agent, _ = build_agent(args, vec_env, device)

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
