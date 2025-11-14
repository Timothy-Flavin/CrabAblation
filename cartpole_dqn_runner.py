from time import time
import os
import torch
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from DQN_Rainbow import RainbowDQN, EVRainbowDQN
import argparse


def setup_config():
    # Parse CLI arguments
    parser = argparse.ArgumentParser(description="MuJoCo DQN Runner")
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
        "--run",
        "--run_id",
        dest="run",
        type=int,
        default=1,
        help="Run id index for distinguishing multiple trials",
    )
    args = parser.parse_args()

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
            "Beta": 0.1,  # pillar (3)
            "dueling": False,  # pillar (4)
            "distributional": True,  # pillar (4)
            "ent_reg_coef": 0.01,  # pillar (2)
            "delayed": True,  # pillar (5)
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
    elif args.ablation == 5:
        # Delayed target off
        cfg["delayed"] = False

    AgentClass = RainbowDQN if cfg.get("distributional", True) else EVRainbowDQN
    # Common args
    common_kwargs = dict(
        soft=cfg.get("soft", False),
        munchausen=cfg.get("munchausen", False),
        Thompson=False,
        dueling=cfg.get("dueling", False),
        Beta=cfg.get("Beta", 0.0),
        ent_reg_coef=cfg.get("ent_reg_coef", 0.0),
        delayed=cfg.get("delayed", True),
    )
    if AgentClass is RainbowDQN:
        dqn = AgentClass(
            obs_dim,
            action_dim,
            zmin=0,
            zmax=200,
            n_atoms=51,
            **common_kwargs,
        )
    else:
        dqn = AgentClass(
            obs_dim,
            action_dim,
            **common_kwargs,
        )

    return dqn, device, configurations, args


if __name__ == "__main__":

    def eval(agent):
        lenv = gym.make("CartPole-v1")
        obs, info = lenv.reset()
        done = False
        reward = 0.0
        while not done:
            with torch.no_grad():
                logits = agent.online(torch.from_numpy(obs).to(device).float())
                ev = agent.online.expected_value(logits)
                action = torch.argmax(ev, dim=-1).item()
            obs, r, term, trunc, info = lenv.step(action)
            reward += float(r)
            done = term or trunc
        return reward

    env = gym.make("CartPole-v1")
    obs, info = env.reset()
    action_dim = 2  # 3^6 because 6 actions with bang-0-bang control
    assert env.observation_space.shape is not None
    obs_dim = env.observation_space.shape[0]
    assert obs_dim is not None

    dqn, device, configurations, args = setup_config()
    # Move all agent submodules to the selected device and reinitialize optimizers
    # Note: Optimizers need to be recreated after moving parameters across devices.
    # Capture learning rates prior to reinitialization
    main_lr = dqn.optim.param_groups[0]["lr"] if hasattr(dqn, "optim") else 1e-3
    int_lr = dqn.int_optim.param_groups[0]["lr"] if hasattr(dqn, "int_optim") else 1e-3
    rnd_lr = dqn.rnd_optim.param_groups[0]["lr"] if hasattr(dqn, "rnd_optim") else 1e-3

    # Move networks/RND and normalizers to device
    if hasattr(dqn, "online"):
        dqn.online.to(device)
    if hasattr(dqn, "target"):
        dqn.target.to(device)
        dqn.target.requires_grad_(False)
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
    if hasattr(dqn, "online"):
        dqn.optim = torch.optim.Adam(dqn.online.parameters(), lr=main_lr)
    if hasattr(dqn, "int_online"):
        dqn.int_optim = torch.optim.Adam(dqn.int_online.parameters(), lr=int_lr)
    if hasattr(dqn, "rnd"):
        dqn.rnd_optim = torch.optim.Adam(dqn.rnd.predictor.parameters(), lr=rnd_lr)

    n_steps = 50000
    rhist = []
    smooth_rhist = []
    lhist = []
    eval_hist = []
    r_ep = 0.0
    smooth_r = 0.0
    ep = 0
    blen = 10000
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

    start_time = time()
    n_updates = 0
    for i in range(n_steps):
        eps_current = 1 - i * 2 / n_steps
        eps_current = max(eps_current, 0.05)
        action = dqn.sample_action(
            torch.from_numpy(obs).to(device).float(),
            eps=eps_current,
            step=i,
            n_steps=n_steps,
        )
        next_obs, r, term, trunc, info = env.step(action)
        # Update running stats with the freshly collected transition (single step)
        dqn.update_running_stats(torch.from_numpy(next_obs).to(device).float())

        # Save transition to memory buffer
        buff_actions[i % blen] = action
        buff_obs[i % blen] = torch.from_numpy(obs).to(device).float()
        buff_next_obs[i % blen] = torch.from_numpy(next_obs).to(device).float()
        buff_term[i % blen] = term
        buff_r[i % blen] = float(r)

        if i > 512 and i % 2 == 0:
            lhist.append(
                dqn.update(
                    buff_obs,
                    buff_actions,
                    buff_r,
                    buff_next_obs,
                    buff_term,
                    batch_size=64,
                    step=min(i, blen),
                )
            )
            n_updates += 1
            dqn.update_target()

        r_ep += float(r)

        if term or trunc:
            next_obs, info = env.reset()
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
            r_ep = 0.0
            ep += 1

            if ep % 5 == 0:
                evalr = 0.0
                for k in range(5):
                    evalr += eval(dqn)
                print(f"eval mean: {evalr/5}")
                eval_hist.append(evalr / 5)

        obs = next_obs

    end_time = time()
    print(f"Training completed in {end_time - start_time:.2f} seconds.")
    # Save artifacts under results/{runner_name}/
    runner_name = "cartpole"
    results_dir = os.path.join("results", runner_name)
    os.makedirs(results_dir, exist_ok=True)

    np.save(
        os.path.join(results_dir, f"train_time_{args.run}_{args.ablation}.npy"),
        end_time - start_time,
    )
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
