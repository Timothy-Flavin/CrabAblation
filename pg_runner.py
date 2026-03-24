import os
import argparse
import random
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from PG_Rainbow import PPOAgent
from runner_utilities import bins_to_continuous
from runner_utilities import obs_transformer, FastObsWrapper, make_env_thunk, plot_results
from minigrid.wrappers import FlatObsWrapper


def get_args():
    parser = argparse.ArgumentParser(description="PPO/PG Rainbow Runner")
    parser.add_argument(
        "--env_name",
        type=str,
        default="minigrid",
        choices=["cartpole", "minigrid", "mujoco"],
        help="Which environment to train",
    )
    parser.add_argument("--device", type=str, default="cpu", help="Torch device to use")
    parser.add_argument(
        "--ablation",
        type=int,
        default=0,
        choices=[0, 1, 2, 3, 4, 5],
        help="0=None, 1=Disable KL clip, 2=0 entropy reg, 3=No RND, 4=Non-distributional, 5=Placeholder (replace GAE with MC)",
    )
    parser.add_argument(
        "--fully_obs",
        action="store_true",
        help="Use FullyObsWrapper instead of Partial OneHot wrapper",
    )
    parser.add_argument("--run", dest="run", type=int, default=1, help="Run id index")
    from runner_utilities import get_device_name
    parser.add_argument(
        "--device_name", type=str, default=get_device_name(), help="Device name for loading best.json params"
    )
    args = parser.parse_args()

    args.num_envs = 4
    args.num_steps = 128

    if args.device_name:
        best_json_path = f"time_files/{args.device_name}/{args.env_name}_ppo_best.json"
        try:
            with open(best_json_path, "r") as f:
                params = json.load(f)
            
            abl_key = f"ablation_{args.ablation}"
            if abl_key in params:
                abl_params = params[abl_key]
                if "device" in abl_params:
                    args.device = abl_params["device"]
                if "num_envs" in abl_params:
                    args.num_envs = abl_params["num_envs"]
            else:
                print(f"Could not find {abl_key} in {best_json_path}")
                
        except Exception as e:
            print(f"Failed to load params {best_json_path}: {e}")

    return args


def setup_config(args):
    requested_device = args.device.strip()
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(requested_device)

    # Defaults
    cfg = {
        "clip_coef": 0.2,  # Surrogate clip for PPO
        "ent_coef": 0.01,
        "Beta": 0.01,  # RND
        "distributional": True,
        "use_gae": True,  # Placeholder
    }

    if args.ablation == 1:
        cfg["clip_coef"] = 100.0  # Effectively disables KL/surrogate clip
    elif args.ablation == 2:
        cfg["ent_coef"] = 0.0
    elif args.ablation == 3:
        cfg["Beta"] = 0.0
    elif args.ablation == 4:
        cfg["distributional"] = False
    elif args.ablation == 5:
        # Placeholder for replacing GAE with Monte Carlo
        cfg["use_gae"] = False

    return cfg, device


def eval(agent, device, step=0, n_steps=300000, env_eval=None, env_name="minigrid"):
    if env_eval is None:
        env_eval = make_env_thunk(False, env_name)()
    obs, info = env_eval.reset()
    done = False
    reward = 0.0
    while not done:
        with torch.no_grad():
            tobs = torch.from_numpy(obs).float().unsqueeze(0).to(device)
            # deterministic action (modes, max prob, etc.)
            action, logprob, ext_v, int_v = agent.sample_action(tobs)

        action = action.cpu().numpy()[0]
        if env_name == "mujoco":
            step_action = bins_to_continuous(action)
        else:
            step_action = action
            
        obs, r, term, trunc, info = env_eval.step(step_action)
        reward += float(r)
        done = term or trunc
    return reward


def train_pg(vec_env, agent, cfg, args, device):
    obs_raw, info = vec_env.reset()
    obs = obs_raw

    n_steps_total = getattr(args, "total_steps", 300000) // args.num_envs
    if args.env_name == "mujoco":
        n_steps_total = getattr(args, "total_steps", 1000000) // args.num_envs

    num_iterations = n_steps_total // args.num_steps

    rhist = []
    smooth_rhist = []
    lhist = []
    eval_hist = []

    r_ep = np.zeros(args.num_envs)
    smooth_r = 0.0

    start_time = time.time()
    runner_name = args.env_name
    results_dir = os.path.join("results", "ppo", runner_name)
    os.makedirs(results_dir, exist_ok=True)
    tb_dir = os.path.join(results_dir, f"tensorboard_run{args.run}_abl{args.ablation}")
    writer = SummaryWriter(log_dir=tb_dir)
    writer.add_scalar("run/started", 1, 0)

    if hasattr(agent, "attach_tensorboard"):
        agent.attach_tensorboard(writer, prefix="agent")

    # Storage setup
    agent_obs = torch.zeros(
        (args.num_steps, args.num_envs) + vec_env.single_observation_space.shape
    ).to(device)
    agent_actions = torch.zeros(
        (args.num_steps, args.num_envs) + vec_env.single_action_space.shape
    ).to(device)
    agent_logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    agent_rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    agent_dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    agent_ext_values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    agent_int_values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    global_step = 0
    next_obs = torch.Tensor(obs_raw).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    total_samples = 0

    for iteration in range(1, num_iterations + 1):
        for step in range(0, args.num_steps):
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
            if args.env_name == "mujoco":
                step_action = np.array([bins_to_continuous(a) for a in np_action])
            else:
                step_action = np_action

            next_obs_np, reward, terminations, truncations, infos = vec_env.step(
                step_action
            )

            next_done_np = np.logical_or(terminations, truncations)
            agent_rewards[step] = torch.tensor(reward).to(device).view(-1)

            next_obs = torch.Tensor(next_obs_np).to(device)
            next_done = torch.Tensor(next_done_np).to(device)

            for ne in range(args.num_envs):
                r_ep[ne] += reward[ne]
                if next_done_np[ne]:
                    if smooth_r == 0.0:
                        smooth_r = r_ep[ne]
                    else:
                        smooth_r = 0.99 * smooth_r + 0.01 * r_ep[ne]
                    rhist.append(r_ep[ne])
                    smooth_rhist.append(smooth_r)
                    writer.add_scalar("charts/episodic_return", r_ep[ne], global_step)
                    r_ep[ne] = 0

        # Perform PPO update
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
        lhist.append(loss)

        # Eval
        if iteration % 5 == 0:
            e_rew = eval(agent, device, env_name=args.env_name)
            print(f"Ran eval r = {e_rew}")
            eval_hist.append(e_rew)
            writer.add_scalar("charts/eval_return", e_rew, global_step)

    writer.flush()
    writer.close()
    return {
        "rhist": rhist,
        "smooth_rhist": smooth_rhist,
        "lhist": lhist,
        "eval_hist": eval_hist,
        "train_time": time.time() - start_time,
    }



if __name__ == "__main__":
    args = get_args()
    print(f"Args: {args}")

    cfg, device = setup_config(args)

    env_fns = [
        make_env_thunk(args.fully_obs, args.env_name) for _ in range(args.num_envs)
    ]
    vec_env = gym.vector.SyncVectorEnv(env_fns)

    agent = PPOAgent(
        vec_env,
        clip_coef=cfg["clip_coef"],
        ent_coef=cfg["ent_coef"],
        distributional=cfg["distributional"],
        Beta=cfg["Beta"],
        num_envs=args.num_envs,
        num_steps=args.num_steps,
        use_gae=cfg["use_gae"],
    ).to(device)

    results = train_pg(vec_env, agent, cfg, args, device)
    plot_results(results, args, "ppo")
    vec_env.close()
