from time import time
import os
import torch
import gymnasium as gym

try:
    from minigrid.wrappers import (
        FlatObsWrapper,
        OneHotPartialObsWrapper,
        FullyObsWrapper,
    )
except ModuleNotFoundError:
    pass
import matplotlib.pyplot as plt
import numpy as np
from DQN_Rainbow import RainbowDQN, EVRainbowDQN
import argparse
from runner_utilities import obs_transformer, FastObsWrapper, make_env_thunk, plot_results, bins_to_continuous
from torch.utils.tensorboard import SummaryWriter
import json



def get_args():
    # Parse CLI arguments
    parser = argparse.ArgumentParser(
        description="MiniGrid DQN Runner (Rainbow/EV/ IQN-ready)"
    )
    parser.add_argument(
        "--env_name",
        type=str,
        default="minigrid",
        choices=["cartpole", "minigrid", "mujoco"],
        help="Which environment to train",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device to use (e.g., cpu, cuda, cuda:0)",
    )
    parser.add_argument(
        "--ablation",
        type=int,
        default=0,
        choices=[0, 1, 2, 3, 4, 5],
        help=(
            "Which pillar to ablate: 0=None, 1=Mirror Descent (munchausen), "
            "2=Magnet Regularization (entropy reg), 3=Optimism (Beta), "
            "4=Distributional (use EV agent), 5=Delayed targets"
        ),
    )
    parser.add_argument(
        "--fully_obs",
        action="store_true",
        help="Use FullyObsWrapper instead of Partial OneHot wrapper (mutually exclusive).",
    )
    parser.add_argument(
        "--run",
        "--run_id",
        dest="run",
        type=int,
        default=1,
        help="Run id index for distinguishing multiple trials",
    )
    parser.add_argument(
        "--best_params",
        type=str,
        default="timpc",
        help="Prefix to the <prefix>_best.json file storing parallel hyper-parameters.",
    )
    args = parser.parse_args()

    if args.best_params:
        best_json_path = f"{args.best_params}_best.json"

        try:
            with open(best_json_path, "r") as f:
                best_results = json.load(f)
                best_config = best_results.get(f"ablation_{args.ablation}")
                if best_config:
                    args.num_envs = best_config.get("num_envs", 1)
                    args.device = best_config.get("device", args.device)
        except Exception as e:
            print(f"Failed to load best params from {best_json_path}: {e}")
            args.num_envs = 1
    else:
        args.num_envs = 1
    return args


def setup_config(obs_dim):
    args = get_args()
    # Resolve torch device; allow cuda or cuda:0 while safely falling back to cpu
    requested_device = args.device.strip()
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA requested but not available. Falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(requested_device)

    # Five pillars of RL to be tested
    # (1) Mirror Descent / Bregman Proximal Optimization
    # (2) Magnet Policy Regularization
    # (3) Optimism in the face of Uncertainty
    # (4) Dual/Dueling/Distributional Value Estimates
    # (5) Delayed / Two-Timescale Optimization
    configurations = {
        "all": {
            "munchausen": True,  # pillar (1)
            "soft": True,  # pillar (3) always enabled if munchausen on
            "Beta": (
                0.7 if args.ablation != 4 else 0.1
            ),  # pillar (3) optimism scale (match mujoco)
            "dueling": True,  # pillar (4)
            "distributional": True,  # pillar (4)
            "ent_reg_coef": 0.05,  # pillar (2)
            "delayed": True,  # pillar (5)
            "popart": True,
            "tau": 0.03,  # exploration / policy temperature
            "alpha": 0.9,  # munchausen log-policy scaling
        }
    }

    cfg = dict(configurations["all"])  # copy base config

    # Apply single pillar ablation based on CLI
    if args.ablation == 1:
        # Mirror Descent / Munchausen off
        cfg["munchausen"] = False
        cfg["soft"] = False
    elif args.ablation == 2:
        # Magnet policy regularization off
        cfg["ent_reg_coef"] = 0.0
    elif args.ablation == 3:
        # Optimism off (no intrinsic Beta)
        cfg["Beta"] = 0.0
    elif args.ablation == 4:
        # Distributional off (use EV agent)
        cfg["distributional"] = False
        cfg["dueling"] = False
        # Slightly reduce entropy reg to keep losses comparable (mirrors mujoco tweak)
        cfg["ent_reg_coef"] = 0.005
    elif args.ablation == 5:
        # Delayed target off
        cfg["delayed"] = False

    AgentClass = RainbowDQN if cfg.get("distributional", True) else EVRainbowDQN
    # Common args
    if args.env_name == "minigrid":
        n_action_dims = 1
        n_action_bins = 3
        hidden_layer_sizes = [128, 128]
        norm_obs = False
    elif args.env_name == "cartpole":
        n_action_dims = 1
        n_action_bins = 2
        hidden_layer_sizes = [64, 64]
        norm_obs = False
    elif args.env_name == "mujoco":
        n_action_dims = 6
        n_action_bins = 3
        hidden_layer_sizes = [256, 256]
        norm_obs = False

    common_kwargs = dict(
        soft=cfg.get("soft", True),
        munchausen=cfg.get("munchausen", True),
        Thompson=False,
        dueling=cfg.get("dueling", True),
        Beta=cfg.get("Beta", 0.0),
        ent_reg_coef=cfg.get("ent_reg_coef", 0.0),
        delayed=cfg.get("delayed", True),
        popart=cfg.get("popart", True),
        tau=cfg.get("tau", 0.03),
        polyak_tau=0.005,
        alpha=cfg.get("alpha", 0.7),
    )
    if AgentClass is RainbowDQN:
        dqn = AgentClass(
            obs_dim,
            n_action_dims,
            n_action_bins,
            hidden_layer_sizes=hidden_layer_sizes,
            Beta_half_life_steps=50000,
            norm_obs=norm_obs,
            burn_in_updates=1000,
            **common_kwargs,
        )
    else:
        dqn = AgentClass(
            obs_dim,
            n_action_dims,
            n_action_bins,
            hidden_layer_sizes=hidden_layer_sizes,
            norm_obs=norm_obs,
            burn_in_updates=1000,
            **common_kwargs,
        )

    return dqn, device, configurations, args




def eval(agent, device, step=0, n_steps=300000, env_eval=None, env_name="minigrid"):
    """Greedy evaluation using agent.sample_action with eps=0 for IQN/Rainbow/EV parity."""
    if env_eval is None:
        if env_name == "minigrid":
            env_eval = gym.make("MiniGrid-FourRooms-v0")
            env_eval = FastObsWrapper(env_eval)
        elif env_name == "cartpole":
            env_eval = gym.make("CartPole-v1")
        elif env_name == "mujoco":
            env_eval = gym.make("HalfCheetah-v5")
    obs, info = env_eval.reset()
    done = False
    reward = 0.0
    while not done:
        with torch.no_grad():
            action_bins = agent.sample_action(
                torch.from_numpy(np.asarray(obs)).to(device).float(),
                eps=0.0,
                step=step,
                n_steps=n_steps,
            )

        if env_name == "mujoco":
            action = bins_to_continuous(action_bins)
        else:
            if isinstance(action_bins, (list, tuple, np.ndarray)):
                action = int(action_bins[0])
            elif torch.is_tensor(action_bins):
                action = int(action_bins.view(-1)[0].item())
            else:
                action = int(action_bins)

        obs, r, term, trunc, info = env_eval.step(action)

        reward += float(r)
        done = term or trunc
    return reward


def train_dqn(vec_env, dqn, configurations, args, device, obs_dim, n_action_dims):
    obs_raw, info = vec_env.reset()
    obs = obs_raw

    n_steps = 300000 // args.num_envs
    if args.env_name == "mujoco":
        n_steps = 1000000 // args.num_envs
    rhist = []
    smooth_rhist = []
    lhist = []
    eval_hist = []

    # We maintain r_ep and ep_len for each parallel environment
    r_ep = np.zeros(args.num_envs)
    ep_len = np.zeros(args.num_envs, dtype=int)
    smooth_r = 0.0
    ep = 0

    blen = 10000
    if args.env_name == "mujoco":
        blen = 20000
    update_every = 4

    # Memory buffer
    if n_action_dims > 1:
        buff_actions = torch.zeros(
            (blen, n_action_dims), dtype=torch.long, device=device
        )
    else:
        buff_actions = torch.zeros((blen,), dtype=torch.long, device=device)
    buff_obs = torch.zeros(
        size=(blen, int(obs_dim)), dtype=torch.float32, device=device
    )
    buff_next_obs = torch.zeros(
        size=(blen, int(obs_dim)), dtype=torch.float32, device=device
    )
    buff_term = torch.zeros((blen,), dtype=torch.float32, device=device)
    buff_r = torch.zeros((blen,), dtype=torch.float32, device=device)

    # Priority experience replay buffers
    if n_action_dims > 1:
        priority_actions = torch.zeros(
            (blen // 4, n_action_dims), dtype=torch.long, device=device
        )
    else:
        priority_actions = torch.zeros((blen // 4,), dtype=torch.long, device=device)

    priority_obs = torch.zeros(
        size=(blen // 4, int(obs_dim)), dtype=torch.float32, device=device
    )
    priority_next_obs = torch.zeros(
        size=(blen // 4, int(obs_dim)), dtype=torch.float32, device=device
    )
    priority_term = torch.zeros((blen // 4,), dtype=torch.float32, device=device)
    priority_r = torch.zeros((blen // 4,), dtype=torch.float32, device=device)

    from time import time

    start_time = time()
    runner_name = args.env_name
    results_dir = os.path.join("results", "dqn", runner_name)
    os.makedirs(results_dir, exist_ok=True)
    tb_dir = os.path.join(results_dir, f"tensorboard_run{args.run}_abl{args.ablation}")
    writer = SummaryWriter(log_dir=tb_dir)
    writer.add_scalar("run/started", 1, 0)

    if hasattr(dqn, "attach_tensorboard"):
        dqn.attach_tensorboard(writer, prefix="agent")

    n_updates = 0
    steps_since_update = 0
    RND_BURN_IN = 1000
    BATCH_SIZE = 64
    priority_idx = 0
    priority_filled = 0

    total_samples = 0
    buffer_ptr = 0

    for i in range(n_steps):
        eps_current = 1 - (i * args.num_envs) * 2 / (n_steps * args.num_envs)
        eps_current = max(eps_current, 0.05)

        tobs = torch.from_numpy(obs).to(device).float()
        action = dqn.sample_action(
            tobs,
            eps=eps_current,
            step=total_samples,
            n_steps=n_steps * args.num_envs,
        )

        if isinstance(action, list) and isinstance(action[0], list):
            action = [a[0] for a in action]

        action = np.array(action)
        if args.env_name == "mujoco":
            step_action = np.array([bins_to_continuous(a) for a in action])
        else:
            step_action = action
        next_obs, r, term, trunc, info = vec_env.step(step_action)

        r_mult = r * 10.0

        if hasattr(dqn, "update_running_stats"):
            dqn.update_running_stats(
                torch.from_numpy(next_obs).to(device).float(),
                torch.tensor(r_mult, device=device).float(),
            )

        for env_i in range(args.num_envs):
            idx = buffer_ptr % blen

            if n_action_dims > 1:
                buff_actions[idx] = torch.tensor(action[env_i], device=device)
            else:
                buff_actions[idx] = int(action[env_i])
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
                rhist.append(r_ep[env_i])
                if len(rhist) < 20:
                    smooth_r = sum(rhist) / len(rhist)
                else:
                    smooth_r = 0.05 * rhist[-1] + 0.95 * smooth_r

                smooth_rhist.append(smooth_r)
                ep += 1

                if ep % 10 == 0:
                    dt = time() - start_time
                    print(
                        f"reward for episode: {ep}: {r_ep[env_i]:.2f} at {total_samples/dt:.2f} steps/sec {n_updates/dt:.2f} updates/s {100*total_samples/(n_steps*args.num_envs):.2f}%"
                    )

                writer.add_scalar("episode/reward", float(rhist[-1]), total_samples)
                writer.add_scalar(
                    "episode/smooth_reward", float(smooth_r), total_samples
                )

                if ep % 50 == 0 and total_samples > (n_steps * args.num_envs) // 2:
                    evalr = 0.0
                    for k in range(5):
                        evalr += eval(
                            dqn,
                            device,
                            step=total_samples,
                            n_steps=n_steps * args.num_envs,
                            env_name=args.env_name,
                        )
                    print(f"eval mean: {evalr/5}")
                    eval_hist.append(evalr / 5)
                    writer.add_scalar(
                        "eval/reward", float(eval_hist[-1]), total_samples
                    )

                if r_ep[env_i] > smooth_r:
                    store_len = min(ep_len[env_i], blen // 4)
                    for k in range(store_len, 0, -1):
                        # The transitions for env_i are spaced apart by args.num_envs in the flat buffer
                        # Since buffer_ptr was just incremented by 1 for this env_i, its current index is buffer_ptr - 1
                        # Wait, we need to go back (k-1) more steps. Each previous step is back args.num_envs indices.
                        steps_back = store_len - k
                        buff_idx = (buffer_ptr - 1 - steps_back * args.num_envs) % blen

                        pidx = (priority_idx + store_len - k) % (blen // 4)
                        priority_obs[pidx].copy_(buff_obs[buff_idx])
                        priority_next_obs[pidx].copy_(buff_next_obs[buff_idx])
                        priority_term[pidx] = buff_term[buff_idx]
                        priority_r[pidx] = buff_r[buff_idx]
                        priority_actions[pidx] = buff_actions[buff_idx]
                    priority_idx = (priority_idx + store_len) % (blen // 4)
                    priority_filled = min(priority_filled + store_len, blen // 4)

                ep_len[env_i] = 0
                r_ep[env_i] = 0.0

        obs = next_obs

        steps_since_update += args.num_envs
        while steps_since_update >= update_every:
            if total_samples > BATCH_SIZE:
                if total_samples < RND_BURN_IN:
                    dqn.update(
                        buff_obs,
                        buff_actions,
                        buff_r,
                        buff_next_obs,
                        buff_term,
                        batch_size=BATCH_SIZE,
                        step=min(total_samples, blen),
                    )
                    n_updates += 1
                else:
                    loss_val = dqn.update(
                        buff_obs,
                        buff_actions,
                        buff_r,
                        buff_next_obs,
                        buff_term,
                        batch_size=BATCH_SIZE,
                        step=min(total_samples, blen),
                    )
                    lhist.append(loss_val)
                    n_updates += 1
                    if priority_filled >= BATCH_SIZE:
                        loss_priority = dqn.update(
                            priority_obs,
                            priority_actions,
                            priority_r,
                            priority_next_obs,
                            priority_term,
                            batch_size=BATCH_SIZE,
                            step=min(priority_filled, blen // 4),
                            extrinsic_only=True,
                        )
                        n_updates += 1

                    dqn.update_target()
            steps_since_update -= update_every

    writer.flush()
    writer.close()
    return {
        "rhist": rhist,
        "smooth_rhist": smooth_rhist,
        "lhist": lhist,
        "eval_hist": eval_hist,
        "train_time": time() - start_time,
    }



if __name__ == "__main__":

    args = get_args()
    print(f"Args: {args}")

    # Make a dummy ENV to get obs_dim
    if args.env_name == "minigrid":
        dummy_env = gym.make("MiniGrid-FourRooms-v0")
        dummy_env = FastObsWrapper(dummy_env)
        obs_raw, _ = dummy_env.reset()
        obs_dim = len(obs_raw)
        n_action_dims_arg = 1
    elif args.env_name == "cartpole":
        dummy_env = gym.make("CartPole-v1")
        obs_raw, _ = dummy_env.reset()
        obs_dim = len(obs_raw)
        n_action_dims_arg = 1
    elif args.env_name == "mujoco":
        dummy_env = gym.make("HalfCheetah-v5")
        obs_raw, _ = dummy_env.reset()
        obs_dim = dummy_env.observation_space.shape[0]
        n_action_dims_arg = 6
    dummy_env.close()

    # Setup config gets args inside
    dqn, device, configurations, _ = setup_config(obs_dim)
    dqn.to(device)

    # Make real ENV
    env_fns = [
        make_env_thunk(args.fully_obs, args.env_name) for _ in range(args.num_envs)
    ]
    vec_env = gym.vector.AsyncVectorEnv(env_fns)
    obs_raw, _ = vec_env.reset()
    obs_dim = obs_raw.shape[1] if len(obs_raw.shape) > 1 else len(obs_raw[0])

    # Train Model with intermittent eval
    results = train_dqn(
        vec_env, dqn, configurations, args, device, obs_dim, n_action_dims_arg
    )

    # Save artifacts under results/{runner_name}/
    plot_results(results, args, "dqn")
    vec_env.close()
