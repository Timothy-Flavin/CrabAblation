from time import time
import os
import torch
import gymnasium as gym
from minigrid.wrappers import FlatObsWrapper, OneHotPartialObsWrapper, FullyObsWrapper
import matplotlib.pyplot as plt
import numpy as np
from DQN_Rainbow import RainbowDQN, EVRainbowDQN
import argparse
from torch.utils.tensorboard import SummaryWriter


def setup_config():

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
            "munchausen": False,  # pillar (1)
            "soft": False,  # pillar (3) always enabled if munchausen on
            "Beta": (
                0.7 if args.ablation != 4 else 0.1
            ),  # pillar (3) optimism scale (match mujoco)
            "dueling": False,  # pillar (4)
            "distributional": True,  # pillar (4)
            "ent_reg_coef": 0.01,  # pillar (2)
            "delayed": True,  # pillar (5)
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
        cfg["ent_reg_coef"] = 0.01
    elif args.ablation == 5:
        # Delayed target off
        cfg["delayed"] = False

    AgentClass = RainbowDQN if cfg.get("distributional", True) else EVRainbowDQN
    # Common args
    n_action_dims = 1
    n_action_bins = 3
    common_kwargs = dict(
        soft=cfg.get("soft", True),
        munchausen=cfg.get("munchausen", True),
        Thompson=False,
        dueling=cfg.get("dueling", False),
        Beta=cfg.get("Beta", 0.0),
        ent_reg_coef=cfg.get("ent_reg_coef", 0.0),
        delayed=cfg.get("delayed", True),
        tau=cfg.get("tau", 0.03),
        polyak_tau=0.005,
        alpha=cfg.get("alpha", 0.7),
    )
    if AgentClass is RainbowDQN:
        dqn = AgentClass(
            obs_dim,
            n_action_dims,
            n_action_bins,
            hidden_layer_sizes=[128, 128],
            Beta_half_life_steps=50000,
            norm_obs=False,
            burn_in_updates=1000,
            **common_kwargs,
        )
    else:
        dqn = AgentClass(
            obs_dim,
            n_action_dims,
            n_action_bins,
            hidden_layer_sizes=[128, 128],
            norm_obs=False,
            burn_in_updates=1000,
            **common_kwargs,
        )

    return dqn, device, configurations, args


class obs_transformer:
    def __init__(self):
        # 7x7 image, 3 channels (one-hot for IDs 1, 2, 8)
        self.image_flat_size = 7 * 7 * 3
        # Direction is one-hot encoded (4 values)
        self.direction_size = 4
        self.frame_size = self.image_flat_size + self.direction_size

        # Last obs includes image and direction
        self.last_obs = np.zeros(self.frame_size)

    def transform(self, obs):
        # Extract object ID channel
        img = obs["image"][:, :, 0]
        direction = obs["direction"]

        # One-hot encode IDs: 1=Empty, 2=Wall, 8=Goal
        one_hot_img = np.zeros((7, 7, 3), dtype=np.float32)
        one_hot_img[:, :, 0] = img == 1
        one_hot_img[:, :, 1] = img == 2
        one_hot_img[:, :, 2] = img == 8

        # One-hot encode direction
        one_hot_dir = np.zeros(4, dtype=np.float32)
        one_hot_dir[direction] = 1.0

        current_obs = one_hot_img.flatten()

        # Combine current image and direction
        current_full = np.concatenate([current_obs, one_hot_dir])

        # Stack current and last frame
        transformed_obs = np.concatenate([current_full, self.last_obs])

        self.last_obs = current_full
        return transformed_obs


if __name__ == "__main__":
    # Parse CLI arguments
    parser = argparse.ArgumentParser(
        description="MiniGrid DQN Runner (Rainbow/EV/ IQN-ready)"
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
    args = parser.parse_args()

    def eval(agent, device, env_eval=None, step=0, n_steps=200000):
        """Greedy evaluation using agent.sample_action with eps=0 for IQN/Rainbow/EV parity."""
        if env_eval is None:
            env_eval = gym.make("MiniGrid-FourRooms-v0")
            env_eval = FlatObsWrapper(env_eval)
        et = obs_transformer()
        obs, info = env_eval.reset()
        obs = et.transform(obs)
        done = False
        reward = 0.0
        eps_current = 1 - step * 2 / n_steps
        eps_current = max(eps_current, 0.05)
        while not done:
            with torch.no_grad():
                action_bins = agent.sample_action(
                    torch.from_numpy(np.asarray(obs)).to(device).float(),
                    eps=eps_current,
                    step=step,
                    n_steps=n_steps,
                    verbose=True,
                )
            # action_bins may be list/tensor length 1 for minigrid
            if isinstance(action_bins, (list, tuple, np.ndarray)):
                action = int(action_bins[0])
            elif torch.is_tensor(action_bins):
                action = int(action_bins.view(-1)[0].item())
            else:
                action = int(action_bins)
            print(action)
            obs, r, term, trunc, info = env_eval.step(action)
            obs = et.transform(obs)

            reward += float(r)
            done = term or trunc
        return reward

    env = gym.make("MiniGrid-FourRooms-v0")
    obs, _ = env.reset()
    env1_transformer = obs_transformer()
    obs = env1_transformer.transform(obs)
    # print(obs)
    # print(obs["image"].shape)
    # print(obs["direction"])
    # env = FlatObsWrapper(env)
    # print(obs.shape)
    # print(obs)
    # exit()
    # if args.fully_obs:
    #     # Fully observed grid; DO NOT stack partial one-hot wrapper (they conflict)
    #     env = FullyObsWrapper(env)
    # else:
    #     # Partial observation -> one-hot encode
    #     env = OneHotPartialObsWrapper(env)
    # env = FlatObsWrapper(env)

    env2 = gym.make("MiniGrid-FourRooms-v0", render_mode="human")
    env2_transformer = obs_transformer()
    # if args.fully_obs:
    #     env2 = FullyObsWrapper(env2)
    # else:
    #     env2 = OneHotPartialObsWrapper(env2)
    # env2 = FlatObsWrapper(env2)
    # obs2, _ = env2.reset()  # This now produces an RGB tensor only
    # obs2, info2 = env2.reset()
    # obs2 = obs2["image"].flatten()
    obs_dim = len(obs)
    # Determine action and observation dimensions
    # if (
    #     isinstance(env.observation_space, gym.spaces.Box)
    #     and env.observation_space.shape is not None
    # ):
    #     obs_dim = len(obs)  # int(np.prod(env.observation_space.shape))
    # else:
    #     # for dict observation spaces, infer from a sample obs
    #     sample = obs
    #     if isinstance(sample, dict):
    #         img = sample.get("image") or sample.get("obs")
    #         obs_dim = int(np.prod(np.asarray(img).shape))
    #     else:
    #         obs_dim = int(np.prod(np.asarray(sample).shape))

    action_dim = 3
    dqn, device, configurations, args = setup_config()
    # Move all agent submodules to the selected device and reinitialize optimizers
    # Note: Optimizers need to be recreated after moving parameters across devices.
    # Capture learning rates prior to reinitialization
    main_lr = dqn.optim.param_groups[0]["lr"] if hasattr(dqn, "optim") else 1e-3
    int_lr = dqn.int_optim.param_groups[0]["lr"] if hasattr(dqn, "int_optim") else 1e-3
    rnd_lr = dqn.rnd_optim.param_groups[0]["lr"] if hasattr(dqn, "rnd_optim") else 1e-3

    # Move networks/RND and normalizers to device
    if hasattr(dqn, "ext_online"):
        dqn.ext_online.to(device)
    if hasattr(dqn, "ext_target"):
        dqn.ext_target.to(device)
        dqn.ext_target.requires_grad_(False)
    if hasattr(dqn, "int_online"):
        dqn.int_online.to(device)
    if hasattr(dqn, "int_target"):
        dqn.int_target.to(device)
        dqn.int_target.requires_grad_(False)
    if hasattr(dqn, "rnd") and hasattr(dqn.rnd, "to"):
        dqn.rnd.to(device)
    # Running stats modules
    if hasattr(dqn, "obs_rms") and hasattr(dqn.obs_rms, "to"):
        dqn.obs_rms.to(device)
    if hasattr(dqn, "int_rms") and hasattr(dqn.int_rms, "to"):
        dqn.int_rms.to(device)

    # Recreate optimizers on moved parameters
    if hasattr(dqn, "ext_online"):
        dqn.optim = torch.optim.Adam(dqn.ext_online.parameters(), lr=1e-3)
    if hasattr(dqn, "int_online"):
        dqn.int_optim = torch.optim.Adam(dqn.int_online.parameters(), lr=1e-3)
    if hasattr(dqn, "rnd"):
        dqn.rnd_optim = torch.optim.Adam(dqn.rnd.predictor.parameters(), lr=1e-3)

    n_steps = 200000
    rhist = []
    smooth_rhist = []
    lhist = []
    eval_hist = []
    r_ep = 0.0
    smooth_r = 0.0
    ep = 0
    blen = 50000
    update_every = 1

    # Memory buffer
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
    priority_obs = torch.zeros(
        size=(blen // 4, int(obs_dim)), dtype=torch.float32, device=device
    )
    priority_next_obs = torch.zeros(
        size=(blen // 4, int(obs_dim)), dtype=torch.float32, device=device
    )
    priority_term = torch.zeros((blen // 4,), dtype=torch.float32, device=device)
    priority_r = torch.zeros((blen // 4,), dtype=torch.float32, device=device)
    priority_actions = torch.zeros((blen // 4,), dtype=torch.long, device=device)

    start_time = time()
    # Initialize TensorBoard writer
    runner_name = "minigrid"
    results_dir = os.path.join("results", runner_name)
    os.makedirs(results_dir, exist_ok=True)
    tb_dir = os.path.join(results_dir, f"tensorboard_run{args.run}_abl{args.ablation}")
    writer = SummaryWriter(log_dir=tb_dir)
    writer.add_scalar("run/started", 1, 0)
    # Attach writer to agent so internal Rainbow/EV losses (extrinsic, intrinsic, rnd, etc.) are logged.
    if hasattr(dqn, "attach_tensorboard"):
        dqn.attach_tensorboard(writer, prefix="agent")
    n_updates = 0
    RND_BURN_IN = 1000
    BATCH_SIZE = 64
    ep_len = 0
    priority_idx = 0
    priority_filled = 0
    for i in range(n_steps):
        # Decay epsilon faster (over 1/4 of training) to ensure policy stabilizes and fills buffer with successes
        eps_current = 1 - i * 2 / n_steps
        eps_current = max(eps_current, 0.05)
        tobs = torch.from_numpy(np.asarray(obs)).to(device).float()
        action = dqn.sample_action(
            tobs,
            eps=eps_current,
            step=i,
            n_steps=n_steps,
        )
        if isinstance(action, list):
            action = action[0]
        next_obs, r, term, trunc, info = env.step(action)
        next_obs = env1_transformer.transform(next_obs)
        # print(next_obs.shape)
        r = float(r) * 10.0
        # Update running stats with the freshly collected transition (single step)
        if hasattr(dqn, "update_running_stats"):
            dqn.update_running_stats(
                torch.from_numpy(np.asarray(next_obs)).to(device).float()
            )

        # Save transition to memory buffer (flattened obs vectors)
        buff_actions[i % blen] = int(action)
        # copy into preallocated buffers
        buff_obs[i % blen].copy_(torch.from_numpy(np.asarray(obs)).to(device).float())
        buff_next_obs[i % blen].copy_(
            torch.from_numpy(np.asarray(next_obs)).to(device).float()
        )
        buff_term[i % blen] = term
        buff_r[i % blen] = float(r)

        if i > BATCH_SIZE and i % 2 == 0:
            if i + BATCH_SIZE < RND_BURN_IN:
                dqn.update(
                    buff_obs,
                    buff_actions,
                    buff_r,
                    buff_next_obs,
                    buff_term,
                    batch_size=BATCH_SIZE,
                    step=min(i, blen),
                )
            else:
                loss_val = dqn.update(
                    buff_obs,
                    buff_actions,
                    buff_r,
                    buff_next_obs,
                    buff_term,
                    batch_size=BATCH_SIZE,
                    step=min(i, blen),
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

        r_ep += float(r)
        ep_len += 1
        if term or trunc:
            next_obs_raw, info = env.reset()
            next_obs = env1_transformer.transform(next_obs_raw)
            next_obs = env1_transformer.transform(
                next_obs_raw
            )  # ensure last_obs updated
            rhist.append(r_ep)
            if len(rhist) < 20:
                dt = time() - start_time
                print(
                    f"reward for episode: {ep}: {r_ep} at {i/dt:.2f} steps/sec {n_updates/dt:.2f} updates/s {100*i/n_steps:.2f}%"
                )
                smooth_r = sum(rhist) / len(rhist)
            else:
                smooth_r = 0.05 * rhist[-1] + 0.95 * smooth_r
                if ep % 10 == 0:
                    dt = time() - start_time
                    print(
                        f"reward for episode: {ep}: {r_ep} at {i/dt:.2f} steps/sec {n_updates/dt:.2f} updates/s {100*i/n_steps:.2f}%"
                    )
            smooth_rhist.append(smooth_r)

            ep += 1
            # TensorBoard: episode metrics
            writer.add_scalar("episode/reward", float(rhist[-1]), i)
            writer.add_scalar("episode/smooth_reward", float(smooth_r), i)

            if ep % 50 == 0 and i > n_steps // 2:
                evalr = 0.0
                for k in range(5):
                    evalr += eval(dqn, device, env_eval=env2, step=i, n_steps=n_steps)
                print(f"eval mean: {evalr/5}")
                eval_hist.append(evalr / 5)
                writer.add_scalar("eval/reward", float(eval_hist[-1]), i)

            # if this episode was better than the running smooth reward, add to priority buffer
            if r_ep > smooth_r:
                store_len = min(ep_len, blen // 4)
                for idx in range(store_len, 0, -1):
                    buff_idx = (i - idx) % blen
                    pidx = (priority_idx + store_len - idx) % (blen // 4)
                    priority_obs[pidx].copy_(buff_obs[buff_idx])
                    priority_next_obs[pidx].copy_(buff_next_obs[buff_idx])
                    priority_term[pidx] = buff_term[buff_idx]
                    priority_r[pidx] = buff_r[buff_idx]
                    priority_actions[pidx] = buff_actions[buff_idx]
                priority_idx = (priority_idx + store_len) % (blen // 4)
                priority_filled = min(priority_filled + store_len, blen // 4)
            ep_len = 0
            r_ep = 0.0
        obs = next_obs

    # Save artifacts under results/{runner_name}/
    runner_name = "minigrid"
    results_dir = os.path.join("results", runner_name)
    os.makedirs(results_dir, exist_ok=True)

    np.save(
        os.path.join(results_dir, f"train_scores_{args.run}_{args.ablation}.npy"), rhist
    )
    np.save(
        os.path.join(results_dir, f"eval_scores_{args.run}_{args.ablation}.npy"),
        eval_hist,
    )
    np.save(
        os.path.join(results_dir, f"loss_hist_{args.run}_{args.ablation}.npy"), lhist
    )
    np.save(
        os.path.join(
            results_dir, f"smooth_train_scores_{args.run}_{args.ablation}.npy"
        ),
        smooth_rhist,
    )

    plt.plot(rhist)
    plt.plot(smooth_rhist)
    plt.legend(["R hist", "Smooth R hist"])
    plt.xlabel("Episode")
    plt.ylabel("Training Episode Reward")
    plt.grid()
    plt.title(f"Training rewards, run {args.run} ablated: {args.ablation}")
    plt.savefig(os.path.join(results_dir, f"train_scores_{args.run}_{args.ablation}"))
    plt.close()
    plt.plot(eval_hist)
    plt.grid()
    plt.title(f"eval scores, run {args.run} ablated: {args.ablation}")
    plt.savefig(os.path.join(results_dir, f"eval_scores_{args.run}_{args.ablation}"))
    plt.close()

    # Save total wall clock training time
    end_time = time()
    train_time_seconds = end_time - start_time
    np.save(
        os.path.join(results_dir, f"train_time_{args.run}_{args.ablation}.npy"),
        train_time_seconds,
    )
    print(f"Training wall clock time: {train_time_seconds:.2f} seconds")
    # Close TensorBoard writer
    writer.flush()
    writer.close()
