from time import time
import os
import argparse
from typing import Dict, Optional

import numpy as np
import torch
import gymnasium as gym
from pettingzoo.classic import leduc_holdem_v4

from DQN_Rainbow import RainbowDQN, EVRainbowDQN


def setup_config():
    parser = argparse.ArgumentParser(
        description="PettingZoo Leduc Hold'em DQN Runner (self-play)"
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="Torch device (cpu, cuda, cuda:0)"
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
        help="Run id for trial disambiguation",
    )
    parser.add_argument("--episodes", type=int, default=5000, help="Training episodes")
    args = parser.parse_args()

    req_dev = args.device.strip()
    if req_dev.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA requested but not available. Falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(req_dev)

    configurations = {
        "all": {
            "munchausen": True,
            "soft": True,
            "Beta": 0.05,
            "dueling": True,
            "distributional": True,
            "ent_reg_coef": 0.02,
            "delayed": True,
        }
    }
    cfg = dict(configurations["all"])
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
    elif args.ablation == 5:
        cfg["delayed"] = False

    AgentClass = RainbowDQN if cfg.get("distributional", True) else EVRainbowDQN
    common_kwargs = dict(
        soft=cfg.get("soft", False),
        munchausen=cfg.get("munchausen", False),
        Thompson=False,
        dueling=cfg.get("dueling", False),
        Beta=cfg.get("Beta", 0.0),
        ent_reg_coef=cfg.get("ent_reg_coef", 0.0),
        delayed=cfg.get("delayed", True),
    )

    # Probe env to get shapes
    penv = leduc_holdem_v4.env()
    penv.reset()
    first_agent = penv.agents[0]
    obs_space = penv.observation_space(first_agent)
    # observation space is a Dict with keys 'observation' and 'action_mask'
    if isinstance(obs_space, gym.spaces.Dict):
        obs_box = obs_space["observation"]
        sample = obs_box.sample()
        obs_dim = int(np.prod(np.asarray(sample).shape))
    else:
        # Fallback: sample an observation
        obs = penv.state() if hasattr(penv, "state") else None
        obs_dim = int(np.prod(np.asarray(obs).shape)) if obs is not None else 36
    action_space = penv.action_space(first_agent)
    action_dim = getattr(action_space, "n", None) or 3
    penv.close()

    if AgentClass is RainbowDQN:
        dqn = AgentClass(
            obs_dim, action_dim, zmin=-10, zmax=10, n_atoms=51, **common_kwargs
        )
    else:
        dqn = AgentClass(obs_dim, action_dim, **common_kwargs)

    return dqn, device, args, action_dim, obs_dim


def masked_action(
    dqn, device, obs_vec: np.ndarray, action_mask: np.ndarray, eps: float
) -> int:
    mask = torch.tensor(action_mask.astype(bool), device=device)
    if mask.sum() == 0:
        return 0
    if np.random.rand() < eps:
        valid_indices = torch.nonzero(mask, as_tuple=False).view(-1)
        return int(valid_indices[np.random.randint(len(valid_indices))].item())
    with torch.no_grad():
        x = torch.from_numpy(obs_vec).to(device).float().unsqueeze(0)
        logits = dqn.online(x)
        ev = dqn.online.expected_value(logits).squeeze(0)  # [A]
        ev_masked = ev.clone()
        ev_masked[~mask] = -1e9
        return int(torch.argmax(ev_masked).item())


def eval_vs_random(dqn, device, n_episodes: int = 50) -> float:
    """Evaluate controlling player_0 against random player_1.
    Returns mean reward for player_0 across episodes."""
    total = 0.0
    for _ in range(n_episodes):
        env = leduc_holdem_v4.env()
        env.reset()
        # track rewards per agent
        rewards: Dict[str, float] = {agent: 0.0 for agent in env.agents}
        last_obs: Dict[str, Optional[np.ndarray]] = {
            agent: None for agent in env.agents
        }
        last_act: Dict[str, Optional[int]] = {agent: None for agent in env.agents}
        for agent in env.agent_iter():
            obs, rew, term, trunc, info = env.last()
            rewards[agent] += float(rew)
            if term or trunc:
                env.step(None)
                continue
            obs_dict = obs if isinstance(obs, dict) else {}
            obs_vec = np.asarray(
                obs_dict.get("observation", np.zeros(obs_dim, dtype=np.float32)),
                dtype=np.float32,
            )
            action_mask = np.asarray(
                obs_dict.get("action_mask", np.ones(action_dim, dtype=bool)), dtype=bool
            )
            if agent == env.agents[0]:  # player_0 is learning agent
                act = masked_action(dqn, device, obs_vec, action_mask, eps=0.0)
            else:
                valid = np.nonzero(action_mask)[0]
                act = int(np.random.choice(valid)) if len(valid) else 0
            env.step(act)
        env.close()
        total += rewards.get(env.agents[0], 0.0)
    return total / max(1, n_episodes)


if __name__ == "__main__":

    dqn, device, args, action_dim, obs_dim = setup_config()

    # Move to device and reset optimizers like other runners
    main_lr = dqn.optim.param_groups[0]["lr"] if hasattr(dqn, "optim") else 1e-3
    if hasattr(dqn, "online"):
        dqn.online.to(device)
    if hasattr(dqn, "target"):
        dqn.target.to(device)
        dqn.target.requires_grad_(False)
    if hasattr(dqn, "online"):
        dqn.optim = torch.optim.Adam(dqn.online.parameters(), lr=main_lr)

    # Replay buffer
    blen = 20000
    buff_actions = torch.zeros((blen,), dtype=torch.long, device=device)
    buff_obs = torch.zeros((blen, obs_dim), dtype=torch.float32, device=device)
    buff_next_obs = torch.zeros((blen, obs_dim), dtype=torch.float32, device=device)
    buff_term = torch.zeros((blen,), dtype=torch.float32, device=device)
    buff_r = torch.zeros((blen,), dtype=torch.float32, device=device)

    # Training trackers
    rhist = []  # episode reward for player_0
    smooth_rhist = []
    lhist = []
    eval_hist = []

    t = 0  # transition counter
    start_time = time()
    eps_min = 0.05
    for ep in range(args.episodes):
        env = leduc_holdem_v4.env()
        env.reset()
        # trackers per agent
        last_obs: Dict[str, Optional[np.ndarray]] = {
            agent: None for agent in env.agents
        }
        last_act: Dict[str, Optional[int]] = {agent: None for agent in env.agents}
        ep_rewards: Dict[str, float] = {agent: 0.0 for agent in env.agents}

        for agent in env.agent_iter():
            obs, rew, term, trunc, info = env.last()
            ep_rewards[agent] += float(rew)

            # If we have a pending transition for this agent, finalize it now with current obs
            if last_obs[agent] is not None and last_act[agent] is not None:
                if term or trunc:
                    curr_vec = np.zeros(obs_dim, dtype=np.float32)
                else:
                    obs_dict = obs if isinstance(obs, dict) else {}
                    curr_vec = np.asarray(
                        obs_dict.get(
                            "observation", np.zeros(obs_dim, dtype=np.float32)
                        ),
                        dtype=np.float32,
                    )
                idx = t % blen
                buff_obs[idx].copy_(torch.from_numpy(last_obs[agent]).to(device))
                la = last_act[agent]
                if la is None:
                    act_val = 0
                else:
                    act_val = int(la)
                buff_actions[idx] = act_val
                buff_next_obs[idx].copy_(torch.from_numpy(curr_vec).to(device))
                buff_r[idx] = float(rew)
                buff_term[idx] = float(term or trunc)
                # update running stats with next obs when available
                if not (term or trunc):
                    dqn.update_running_stats(torch.from_numpy(curr_vec).to(device))
                t += 1

                # Parameter updates periodically once buffer has data
                if t > 1024 and t % 4 == 0:
                    l = dqn.update(
                        buff_obs,
                        buff_actions,
                        buff_r,
                        buff_next_obs,
                        buff_term,
                        batch_size=64,
                        step=min(t, blen),
                    )
                    lhist.append(l)
                    dqn.update_target()

            if term or trunc:
                env.step(None)
                last_obs[agent] = None
                last_act[agent] = None
                continue

            obs_dict = obs if isinstance(obs, dict) else {}
            obs_vec = np.asarray(
                obs_dict.get("observation", np.zeros(obs_dim, dtype=np.float32)),
                dtype=np.float32,
            )
            action_mask = np.asarray(
                obs_dict.get("action_mask", np.ones(action_dim, dtype=bool)), dtype=bool
            )
            # epsilon schedule by episodes
            eps = max(eps_min, 1.0 - 2.0 * ep / max(1, args.episodes))
            act = masked_action(dqn, device, obs_vec, action_mask, eps)
            last_obs[agent] = obs_vec
            last_act[agent] = act
            env.step(act)

        env.close()

        # Episode end: record player_0 reward
        p0 = env.agents[0] if len(env.agents) > 0 else "player_0"
        rhist.append(ep_rewards.get(p0, 0.0))
        if len(rhist) < 20:
            smooth_r = float(np.mean(rhist))
        else:
            smooth_r = 0.05 * rhist[-1] + 0.95 * smooth_rhist[-1]
        smooth_rhist.append(smooth_r)

        if ep % 20 == 0 and ep > 0:
            score = eval_vs_random(dqn, device, n_episodes=20)
            eval_hist.append(score)
            dt = time() - start_time
            print(
                f"Episode {ep}/{args.episodes} | Eval vs random: {score:.3f} | transitions: {t} | {ep/dt:.2f} eps/s"
            )

    end_time = time()
    print(
        f"Training completed in {end_time - start_time:.2f} seconds with {t} transitions."
    )

    # Save artifacts under results/leduc/
    runner_name = "leduc"
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
