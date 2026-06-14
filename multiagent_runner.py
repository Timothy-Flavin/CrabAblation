import argparse
import os
import time
import numpy as np
import torch
import pyspiel
from open_spiel.python.algorithms.exploitability import exploitability
from open_spiel.python.policy import Policy
from shimmy.openspiel_compatibility import OpenSpielCompatibilityV0
from runner import build_agent, process_args, get_parser
from runner_utils import resolve_torch_device
import gymnasium as gym
from types import SimpleNamespace
import os
import random

def get_ma_args():
    parser = get_parser()
    parser.add_argument("--ma_env", type=str, default="tictactoe", choices=["tictactoe", "leduc", "rps"])
    parser.add_argument("--total_episodes", type=int, default=10000)
    parser.add_argument("--eval_every", type=int, default=500)
    parser.add_argument("--eval_episodes", type=int, default=100)
    parser.add_argument("--ent_coef_override", type=float, default=None)
    # Batch of parallel one-shot RPS games per round (vectorized path; RPS only).
    parser.add_argument("--rps_batch_size", type=int, default=64)
    # Gradient steps per round
    parser.add_argument("--rps_grad_steps", type=int, default=4)
    
    # Override some defaults for MA
    # learning_starts of 10k (single-agent default) is far too large for the short
    # multi-agent runs (e.g. RPS is only ~20k agent_iter steps), so lower it here.
    # rnd_burn_in of 1000 is also a single-agent (1M-step) default: SAC's update() does
    # ONLY RND for the first rnd_burn_in *update calls*, so with the MA update cadence
    # the actor never trains on short games. Lower it so the policy actually learns.
    # num_steps of 2048 (single-agent vectorized rollout) is far too large for MA
    # self-play: with num_envs=1 each PPO agent collects ~1 transition per decision, so
    # it takes ~2048 decisions to fill one rollout -> only a handful of PPO updates over
    # a whole run (exploitability stays frozen). 128 gives frequent updates so PPO
    # actually learns (verified: Leduc Ext 2.2->1.8 vs frozen 2.07 at 2048).
    # dqn_target_entropy_frac: the soft-DQN alpha autotuner defaults to 0.2*max_ent
    # (exploitative single-agent target). Nash on symmetric games (RPS) IS max entropy
    # (uniform), so 0.2 drives the policy to a peaked 20%-entropy distribution and
    # exploitability climbs over training. Target near-max entropy for MA instead.
    # These games are SHORT, so the single-agent (1M-step) defaults are all oversized:
    #  - learning_starts / rnd_burn_in are in ENV-step units; keep them to a few hundred
    #    steps so learning actually starts early instead of after most of the run.
    #  - small batches (32) and a modest buffer are plenty for short horizons and let
    #    off-policy agents track the (non-stationary) self-play opponent with recent data.
    #  - PPO rollout (num_steps) kept <=256; with batches of 32 it needs no extra
    #    stability margin for these short time-horizon games.
    parser.set_defaults(
        num_envs=1,
        env_name="tictactoe",
        learning_starts=256,
        rnd_burn_in=100,
        num_steps=128,
        batch_size=32,
        dqn_batch_size=32,
        buffer_size=20000,
        dqn_buffer_size=20000,
        dqn_target_entropy_frac=0.9,
    )
    
    args, _ = parser.parse_known_args()
    # Sync env_name with ma_env for build_agent
    if args.ma_env == "tictactoe":
        args.env_name = "tictactoe"
    elif args.ma_env == "leduc":
        args.env_name = "leduc"
    else:
        args.env_name = "rps"
        
    return process_args(args)

class MAWrapperPolicy(Policy):
    def __init__(self, game, agent_wrappers):
        super().__init__(game, [0, 1])
        self.agent_wrappers = agent_wrappers
        self.game_type = game.get_type()
        self._cache = {}

    def action_probabilities(self, state, player_id=None):
        if player_id is None:
            player_id = state.current_player()
            
        if state.is_terminal():
            return {}
            
        infostate = state.information_state_string(player_id)
        if (player_id, infostate) in self._cache:
            return self._cache[(player_id, infostate)]
            
        if self.game_type.provides_observation_tensor:
            obs = np.array(state.observation_tensor(player_id), dtype=np.float32)
        else:
            obs = np.array(state.information_state_tensor(player_id), dtype=np.float32)
            
        legal_actions = state.legal_actions(player_id)
        wrapper = self.agent_wrappers[f"player_{player_id}"]
        
        mask_t = torch.zeros(wrapper.args.n_actions, dtype=torch.float32, device=wrapper.device)
        mask_t[legal_actions] = 1.0
        
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=wrapper.device).unsqueeze(0)
        
        with torch.no_grad():
            probs = wrapper._get_probs(obs_t, mask_t)
            probs_np = probs.cpu().numpy()
        
        dict_probs = {}
        total = 0.0
        for act in legal_actions:
            str_p = float(probs_np[act])
            dict_probs[act] = max(1e-8, str_p) # safety
            total += dict_probs[act]
        
        if total > 0:
            for act in legal_actions:
                dict_probs[act] /= total
        else:
            for act in legal_actions:
                dict_probs[act] = 1.0 / len(legal_actions)
        
        self._cache[(player_id, infostate)] = dict_probs
        return dict_probs

def evaluate_vs_random(agent_wrappers, env_name, agent_id_to_eval, num_episodes=50):
    if env_name == "tictactoe":
        game_name = "tic_tac_toe"
    elif env_name == "leduc":
        game_name = "leduc_poker"
    else:
        game_name = "matrix_rps"
        
    eval_env = OpenSpielCompatibilityV0(game_name=game_name)
    returns = []
    
    is_simultaneous = (env_name == "rps")
    
    for _ in range(num_episodes):
        eval_env.reset()
        ep_ret = 0.0
        
        if is_simultaneous:
            # Simultaneous games in AEC usually have a fixed order but results are applied at the end
            # or they are stepped one by one. Shimmy/OpenSpiel AEC for Matrix games:
            # player_0 acts, then player_1 acts, then both get rewards.
            for agent_id in eval_env.agent_iter():
                obs, reward, term, trunc, info = eval_env.last()
                if agent_id == agent_id_to_eval:
                    ep_ret += reward
                if term or trunc:
                    eval_env.step(None)
                    continue
                mask = info["action_mask"]
                if agent_id == agent_id_to_eval:
                    act, _, _ = agent_wrappers[agent_id].get_action(obs.flatten(), mask, deterministic=True)
                else:
                    valid = np.where(mask)[0]
                    act = np.random.choice(valid)
                eval_env.step(act)
        else:
            for agent_id in eval_env.agent_iter():
                obs, reward, term, trunc, info = eval_env.last()
                if agent_id == agent_id_to_eval:
                    ep_ret += reward
                if term or trunc:
                    eval_env.step(None)
                    continue
                mask = info["action_mask"]
                if agent_id == agent_id_to_eval:
                    act, _, _ = agent_wrappers[agent_id].get_action(obs.flatten(), mask, deterministic=True)
                    eval_env.step(act)
                else:
                    valid = np.where(mask)[0]
                    eval_env.step(np.random.choice(valid))
        returns.append(ep_ret)
    eval_env.close()
    return np.mean(returns)

def get_rps_exploitability(agents):
    # RPS is a matrix game. player_0 and player_1 are symmetric.
    # Obs is usually constant. We get probs for a dummy obs.
    dummy_obs = torch.zeros((1, 1), device=agents["player_0"].device)
    dummy_mask = torch.ones((1, 3), device=agents["player_0"].device)
    
    with torch.no_grad():
        probs0 = agents["player_0"]._get_probs(dummy_obs, dummy_mask).detach().cpu().numpy()
        probs1 = agents["player_1"]._get_probs(dummy_obs, dummy_mask).detach().cpu().numpy()
    
    # Payoff matrix for player 0: 0: Rock, 1: Paper, 2: Scissors
    # R vs R: 0, R vs P: -1, R vs S: 1
    # P vs R: 1, P vs P: 0, P vs S: -1
    # S vs R: -1, S vs P: 1, S vs S: 0
    payoffs = np.array([
        [0, -1, 1],
        [1, 0, -1],
        [-1, 1, 0]
    ])
    
    # Expected value for player 0 if they play action i: sum_j payoffs[i, j] * probs1[j]
    ev0 = payoffs @ probs1
    br0_val = np.max(ev0)
    
    # Expected value for player 1 if they play action j: sum_i payoffs_p1[j, i] * probs0[i]
    # payoffs_p1 = -payoffs.T
    ev1 = (-payoffs.T) @ probs0
    br1_val = np.max(ev1)
    
    # Exploitability in zero-sum symmetric is often (BR0 + BR1)/2
    # Current value for P0 is probs0 @ payoffs @ probs1
    v0 = probs0 @ payoffs @ probs1
    v1 = -v0
    
    return (br0_val - v0 + br1_val - v1) / 2.0

class MAAgentWrapper:
    def __init__(self, agent, algo, device, args):
        self.agent = agent
        self.algo = algo
        self.device = device
        self.args = args
        self.action_dist_log = [] # List of policy distributions for Tic-Tac-Toe
        
    def get_action(self, obs, mask, deterministic=False, step=0, total_steps=1000000, log_dist=False):
        # We now use the agent's built-in sample_action
        # Prepare inputs
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        mask_t = torch.as_tensor(mask, dtype=torch.float32, device=self.device)
        
        # Track action distribution for Tic-Tac-Toe ternary plots or RPS strategy tracking
        if log_dist and self.args.ma_env in ["tictactoe", "rps"]:
            with torch.no_grad():
                probs = self._get_probs(obs_t, mask_t)
                self.action_dist_log.append(probs.cpu().numpy())

        if self.algo == "dqn":
            eps = max(0.5 - 2.0 * (step / total_steps), 0.05)
            # Rainbow DQN sample_action expects (obs, eps, step, ...)
            # We added action_mask support to it.
            action = self.agent.sample_action(
                obs_t, eps=eps, step=step, n_steps=total_steps, action_mask=mask_t
            )
            if isinstance(action, (list, np.ndarray)) and len(action) == 1:
                action = action[0]
            val = action.item() if hasattr(action, 'item') else int(action)
            return val, val, None
            
        elif self.algo == "ppo":
            # PPO sample_action expects (obs)
            # We added action_mask support to it.
            action, logprob = self.agent.sample_action(obs_t, action_mask=mask_t)
            return int(action.item()), action.item(), logprob.item()
            
        elif self.algo == "sac":
            # SAC sample_action expects (obs, deterministic)
            # We will handle masking manually so we get the raw continuous outputs back
            action = self.agent.sample_action(obs_t, deterministic=deterministic)
            action_np = action if isinstance(action, np.ndarray) else action.cpu().numpy()
            if action_np.ndim == 2:
                action_np = action_np[0]

            # Argmax over valid actions only
            valid_actions = np.where(mask > 0)[0]
            if len(valid_actions) > 0:
                best_valid_idx = np.argmax(action_np[valid_actions])
                env_act = int(valid_actions[best_valid_idx])
            else:
                env_act = int(np.argmax(action_np))
                
            return env_act, action_np, None
        
        return None, None, None

    def _get_probs(self, obs_t, mask_t):
        # Internal helper to extract policy probabilities for tracking
        if obs_t.ndim == 1:
            obs_t = obs_t.unsqueeze(0)
        
        # Ensure mask_t is at least 1D and matches q shape later
        if mask_t.ndim == 2:
            mask_t_1d = mask_t[0]
        else:
            mask_t_1d = mask_t

        if self.algo == "dqn":
            if hasattr(self.agent, "n_quantiles"):
                # IQN: mean over quantiles
                taus = self.agent._sample_taus(1, self.agent.n_quantiles, self.device)
                q = self.agent.ext_online(obs_t, taus, normalized=True)
                # q shape is [1, 32, 1, 9] for MultiDiscrete or [1, 32, 9] for Discrete
                # We need to mean over quantiles and reduce to [1, n_actions]
                q = q.view(1, self.agent.n_quantiles, -1).mean(dim=1)
            else:
                q = self.agent.ext_online(obs_t, normalized=True)
                q = q.view(1, -1)
            
            # Use the 1D version for indexing q[0]
            q[0][mask_t_1d == 0] = -1e9
            if self.agent.soft or self.agent.munchausen:
                return torch.softmax(q / self.agent.alpha, dim=-1)[0]
            else:
                # Epsilon-greedy approx distribution
                probs = torch.zeros_like(q[0])
                best_act = torch.argmax(q[0])
                eps = getattr(self.agent, "last_eps", 0.05)
                valid_count = mask_t_1d.sum()
                probs[mask_t_1d == 1] = eps / valid_count
                probs[best_act] += (1.0 - eps)
                return probs
                
        elif self.algo == "ppo":
            logits = self.agent.actor(obs_t)
            logits[0][mask_t_1d == 0] = -1e9
            return torch.softmax(logits, dim=-1)[0]
            
        elif self.algo == "sac":
            # SAC acts by sampling a continuous activation per action (Box proxy) and
            # taking the argmax over legal actions. The realized discrete policy is
            # therefore P(argmax of sampled activations == k), which has no closed form.
            # Estimate it via Monte-Carlo: draw many stochastic action vectors from the
            # actor, mask illegal dims, argmax, and histogram. This reflects what SAC
            # actually plays (unlike a softmax of the means, which is arbitrary).
            n_samples = getattr(self, "sac_mc_samples", 256)
            n_actions = mask_t_1d.shape[0]
            with torch.no_grad():
                obs_rep = obs_t.expand(n_samples, -1)
                sampled, _, _ = self.agent.actor.get_action(obs_rep)  # [N, n_actions]
                sampled = sampled.clone()
                mask_1d = mask_t_1d.to(sampled.device)
                sampled[:, mask_1d == 0] = -1e9
                choices = torch.argmax(sampled, dim=-1)  # [N]
                probs = torch.bincount(choices, minlength=n_actions).float()
                total = probs.sum()
                if total > 0:
                    probs = probs / total
                else:
                    probs = mask_t_1d / mask_t_1d.sum()
            return probs

    def observe(self, obs, action, reward, next_obs, term, trunc, logprob=None, action_mask=None):
        # Flattened obs
        if self.algo == "ppo":
            # PPO observe expects (obs, action, logprob, reward, next_obs, term, trunc, infos)
            # all as batches (num_envs, ...). Pass the legal-action mask through infos so
            # the PPO update can recompute log-probs/entropy on the *masked* distribution
            # (otherwise the importance ratio and entropy bonus include illegal actions).
            infos = {}
            if action_mask is not None:
                infos["action_mask"] = np.asarray(action_mask)[np.newaxis, ...]
            self.agent.observe(
                obs[np.newaxis, ...],
                np.array([action]),
                np.array([logprob]),
                np.array([reward]),
                next_obs[np.newaxis, ...],
                np.array([term]),
                np.array([trunc]),
                infos
            )
        else:
            # DQN/SAC observe expects (obs, action, reward, next_obs, term, trunc, info)
            # action should be (num_envs, action_dim)
            if self.algo == "dqn":
                act_to_store = np.array([[action]])
            else:
                # SAC: action is an array of shape (act_dim,)
                act_to_store = np.array([action])
                
            self.agent.observe(
                obs[np.newaxis, ...], 
                act_to_store, 
                np.array([reward]), 
                next_obs[np.newaxis, ...], 
                np.array([term]), 
                np.array([trunc]), 
                {}
            )

    def get_action_batch(self, obs_t, mask_t, step=0, total_steps=1000000, log_dist=False):
        """Batched action selection for the vectorized (simultaneous) RPS path.
        obs_t: (B, obs_dim), mask_t: (B, n_actions), both torch tensors on device.
        Returns (env_actions[B] long tensor, raw_for_buffer, logprob_or_None)."""
        if log_dist:
            with torch.no_grad():
                probs = self._get_probs(obs_t[:1], mask_t[:1])
                self.action_dist_log.append(probs.cpu().numpy())

        B = obs_t.shape[0]
        if self.algo == "dqn":
            eps = max(0.5 - 2.0 * (step / total_steps), 0.05)
            actions = self.agent.sample_action(
                obs_t, eps=eps, step=step, n_steps=total_steps, action_mask=mask_t
            )
            act_t = torch.as_tensor(
                np.asarray(actions), device=obs_t.device
            ).long().reshape(B)
            return act_t, act_t, None
        elif self.algo == "sac":
            action = self.agent.sample_action(obs_t, deterministic=False)
            action_np = action if isinstance(action, np.ndarray) else action.detach().cpu().numpy()
            # Argmax over (all-legal) action dims -> discrete action; store full vector.
            env_act = torch.as_tensor(np.argmax(action_np, axis=1), device=obs_t.device).long()
            return env_act, action_np, None
        elif self.algo == "ppo":
            action, logprob = self.agent.sample_action(obs_t, action_mask=mask_t)
            return action.long().reshape(B), action, logprob
        return None, None, None

    def observe_batch(self, obs, raw, reward, next_obs, term, trunc, logprob=None, mask=None):
        """Batched transition store for the vectorized RPS path. All array-likes have a
        leading batch (num_envs) dimension; tensors are moved to CPU numpy at the boundary."""
        def _np(x):
            if torch.is_tensor(x):
                return x.detach().cpu().numpy()
            return np.asarray(x)

        if self.algo == "ppo":
            infos = {}
            if mask is not None:
                infos["action_mask"] = _np(mask)
            self.agent.observe(
                _np(obs), _np(raw), _np(logprob), _np(reward),
                _np(next_obs), _np(term), _np(trunc), infos,
            )
        else:
            if self.algo == "dqn":
                act_to_store = _np(raw).reshape(-1, 1)
            else:
                act_to_store = _np(raw)
            self.agent.observe(
                _np(obs), act_to_store, _np(reward),
                _np(next_obs), _np(term), _np(trunc), {},
            )

    def update(self, global_step):
        if self.algo == "ppo":
            return self.agent.update(global_step=global_step)
        else:
            batch_size = getattr(self.args, "dqn_batch_size", 64) if self.algo == "dqn" else getattr(self.args, "batch_size", 64)
            learning_starts = getattr(self.args, "learning_starts", 1000)

            # Safety checks for buffer-based agents (DQN, SAC). global_step counts ENV
            # steps (games), so learning_starts is in env-step units. The buffer stores
            # timesteps of n_envs each, so its transition count is size()*n_envs -- gate
            # on that, NOT raw size(): with B parallel games, size() (timesteps) would
            # otherwise need `batch_size` ROUNDS before learning starts.
            if global_step < learning_starts:
                return None
            n_envs = int(getattr(self.agent.buffer, "n_envs", 1))
            if self.agent.buffer.size() * n_envs < batch_size:
                return None
                
            if self.algo == "dqn":
                return self.agent.update(batch_size=batch_size, step=global_step)
            else:
                return self.agent.update(batch_size=batch_size, global_step=global_step)

def flatten_obs(obs):
    if isinstance(obs, dict):
        return obs["observation"].flatten()
    return obs.flatten()

def train_rps_vectorized(args, seed=0):
    """Vectorized self-play for Rock-Paper-Scissors."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    args.seed = seed

    device = resolve_torch_device(args.device)
    B = max(1, int(getattr(args, "rps_batch_size", 64)))
    n_actions = 3
    args.n_actions = n_actions
    obs_dim = 1

    mock_env = SimpleNamespace(
        single_observation_space=gym.spaces.Box(low=0, high=1, shape=(obs_dim,), dtype=np.float32),
        single_action_space=gym.spaces.Discrete(n_actions),
        num_envs=B,
    )
    args.dqn_buffer_size = min(getattr(args, "dqn_buffer_size", 10000), 10000)
    args.buffer_size = min(getattr(args, "buffer_size", 10000), 10000)

    agent_raws = []
    for _ in range(2):
        a_raw, _ = build_agent(args, mock_env, device)
        if args.ent_coef_override is not None and args.algo == "ppo":
            a_raw.ent_coef = args.ent_coef_override
        agent_raws.append(a_raw)
    agents = {f"player_{i}": MAAgentWrapper(agent_raws[i], args.algo, device, args) for i in range(2)}

    # Payoff to player_0: rows = P0 action, cols = P1 action (0=R,1=P,2=S). Zero-sum.
    payoff = torch.tensor(
        [[0.0, -1.0, 1.0], [1.0, 0.0, -1.0], [-1.0, 1.0, 0.0]], device=device
    )
    obs_t = torch.zeros((B, obs_dim), device=device)
    obs_np = obs_t.detach().cpu().numpy()
    mask_t = torch.ones((B, n_actions), device=device)
    term = np.ones(B, dtype=np.float32)   # one-shot game: every step terminates
    trunc = np.zeros(B, dtype=np.float32)

    total_steps = 0
    ep_rewards = {aid: [] for aid in ["player_0", "player_1"]}
    exploitability_hist, rand_scores_0, rand_scores_1 = [], [], []

    rounds = max(1, args.total_episodes // B)
    eval_interval = max(1, rounds // 10)
    log_interval = max(1, rounds // 150)
    # Off-policy agents (DQN/SAC) take several gradient steps per round
    grad_steps = max(1, int(getattr(args, "rps_grad_steps", 8)))
    start_time = time.time()

    for rnd in range(rounds):
        log_this = (rnd % log_interval == 0)
        a0, raw0, lp0 = agents["player_0"].get_action_batch(
            obs_t, mask_t, step=total_steps, total_steps=args.total_steps, log_dist=log_this
        )
        a1, raw1, lp1 = agents["player_1"].get_action_batch(
            obs_t, mask_t, step=total_steps, total_steps=args.total_steps, log_dist=log_this
        )
        r0 = payoff[a0, a1]          # (B,)
        r1 = -r0
        r0_np, r1_np = r0.detach().cpu().numpy(), r1.detach().cpu().numpy()

        agents["player_0"].observe_batch(obs_np, raw0, r0_np, obs_np, term, trunc, lp0, mask_t)
        agents["player_1"].observe_batch(obs_np, raw1, r1_np, obs_np, term, trunc, lp1, mask_t)
        total_steps += B

        # Each round collects a fresh batch of B games. PPO updates when its rollout is full
        for pid in ["player_0", "player_1"]:
            a = agents[pid]
            if args.algo == "ppo":
                if a.agent.step_idx >= a.agent.num_steps:
                    a.update(total_steps)
            else:
                for _ in range(grad_steps):
                    a.update(total_steps)

        ep_rewards["player_0"].append(float(r0_np.mean()))
        ep_rewards["player_1"].append(float(r1_np.mean()))

        if (rnd + 1) % eval_interval == 0 or (rnd + 1) == rounds:
            r0e = evaluate_vs_random(agents, "rps", "player_0", num_episodes=args.eval_episodes)
            r1e = evaluate_vs_random(agents, "rps", "player_1", num_episodes=args.eval_episodes)
            rand_scores_0.append(r0e)
            rand_scores_1.append(r1e)
            expl = get_rps_exploitability(agents)
            exploitability_hist.append(expl)
            fps = total_steps / (time.time() - start_time)
            print(
                f"Round {rnd+1}/{rounds} (games {total_steps}): vsRand(P0)={r0e:.2f}, "
                f"vsRand(P1)={r1e:.2f}, Ext={expl:.4f} | FPS {fps:.0f}"
            )

    results_dir = os.path.join("results", args.algo, args.env_name)
    os.makedirs(results_dir, exist_ok=True)
    for aid, rewards in ep_rewards.items():
        np.save(os.path.join(results_dir, f"train_scores_{aid}_{args.ablation}_seed{seed}.npy"), np.array(rewards))
    np.save(os.path.join(results_dir, f"exploitability_{args.ablation}_seed{seed}.npy"), np.array(exploitability_hist))
    np.save(os.path.join(results_dir, f"evaluate_vs_random_p0_{args.ablation}_seed{seed}.npy"), np.array(rand_scores_0))
    np.save(os.path.join(results_dir, f"evaluate_vs_random_p1_{args.ablation}_seed{seed}.npy"), np.array(rand_scores_1))
    for pid, tag in [("player_0", "p0"), ("player_1", "p1")]:
        dist = agents[pid].action_dist_log
        if len(dist) > 0:
            np.save(os.path.join(results_dir, f"action_dist_{tag}_{args.ablation}_seed{seed}.npy"), np.array(dist, dtype=object))
    return ep_rewards


def train_ma(args, seed=0):
    import random
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    args.seed = seed
    device = resolve_torch_device(args.device)
    
    if args.ma_env == "tictactoe":
        game_name = "tic_tac_toe"
    elif args.ma_env == "leduc":
        game_name = "leduc_poker"
    else:
        game_name = "matrix_rps"
        
    game = pyspiel.load_game(game_name)
    env = OpenSpielCompatibilityV0(game_name=game_name)
    env.reset()
    
    # Standardize obs_dim
    if args.ma_env == "tictactoe":
        obs_dim = 27
    elif args.ma_env == "leduc":
        obs_dim = 16
    else:
        # RPS observation is usually just a constant or dummy in OpenSpiel
        obs_dim = env.observation_space("player_0").shape[0] if len(env.observation_space("player_0").shape) > 0 else 1
        
    n_actions = env.action_space("player_0").n
    args.n_actions = n_actions
        
    mock_env = SimpleNamespace(
        single_observation_space=gym.spaces.Box(low=0, high=1, shape=(obs_dim,), dtype=np.float32),
        single_action_space=gym.spaces.Discrete(n_actions),
        num_envs=1
    )
    
    # Smaller buffers for tiny games to save RAM
    if args.ma_env in ["tictactoe", "rps"]:
        args.dqn_buffer_size = min(getattr(args, "dqn_buffer_size", 10000), 10000)
        args.buffer_size = min(getattr(args, "buffer_size", 10000), 10000)

    # Build two agents (self-play)
    agent_raws = []
    for _ in range(len(env.possible_agents)):
        a_raw, _ = build_agent(args, mock_env, device)
        # Apply entropy override if provided (for the unit test)
        if args.ent_coef_override is not None and args.algo == "ppo":
            a_raw.ent_coef = args.ent_coef_override
        agent_raws.append(a_raw)
    
    agents = {
        agent_id: MAAgentWrapper(agent_raws[i], args.algo, device, args)
        for i, agent_id in enumerate(env.possible_agents)
    }
    
    total_steps = 0
    ep_rewards = {agent_id: [] for agent_id in env.possible_agents}
    exploitability_hist = []
    rand_scores_0 = []
    rand_scores_1 = []
    
    start_time = time.time()
    eval_interval = max(1, args.total_episodes // 10)
    #
    # CRITICAL: the counter must be PER-AGENT or some players never update
    update_every = max(1, int(getattr(args, "update_every", 4)))
    agent_decisions = {agent_id: 0 for agent_id in env.possible_agents}

    for ep in range(args.total_episodes):
        env.reset()
        
        current_ep_rewards = {agent_id: 0.0 for agent_id in env.possible_agents}
        # Reward each agent has earned since its last action
        pending_reward = {agent_id: 0.0 for agent_id in env.possible_agents}
        last_data = {
            agent_id: {"obs": None, "action": None, "logprob": None, "mask": None}
            for agent_id in env.possible_agents
        }
        
        # Only log distribution every 10 episodes to save memory
        log_this_ep = (ep % 10 == 0)
        
        for agent_id in env.agent_iter():
            obs, _last_reward, termination, truncation, info = env.last()
            # Use our own per-agent accrual
            reward = pending_reward[agent_id]
            pending_reward[agent_id] = 0.0
            current_ep_rewards[agent_id] += reward
            if last_data[agent_id]["obs"] is not None:
                agents[agent_id].observe(
                    last_data[agent_id]["obs"],
                    last_data[agent_id]["action"],
                    reward,
                    obs.flatten(),
                    termination,
                    truncation,
                    logprob=last_data[agent_id]["logprob"],
                    action_mask=last_data[agent_id]["mask"]
                )
            
            if termination or truncation:
                env_act = None
            else:
                flat_obs = obs.flatten()
                mask = info["action_mask"]
                
                env_act, raw_act, logprob = agents[agent_id].get_action(
                    flat_obs, mask, step=total_steps, total_steps=args.total_steps, log_dist=log_this_ep
                )
                
                last_data[agent_id]["obs"] = flat_obs
                last_data[agent_id]["action"] = raw_act
                last_data[agent_id]["logprob"] = logprob
                last_data[agent_id]["mask"] = mask

            env.step(env_act)
            total_steps += 1

            # Accrue the instantaneous per-step rewards into each agent's pending total.
            # Only on real action steps: the terminal None step re-emits the same
            # env.rewards, which is a double-count we are avoiding.
            if env_act is not None:
                for aid2, rv in env.rewards.items():
                    pending_reward[aid2] = pending_reward.get(aid2, 0.0) + float(rv)

            # Periodic update: only update the agent who just acted to be more efficient
            if env_act is not None:
                agent_decisions[agent_id] += 1
                a = agents[agent_id]
                if args.algo == "ppo":
                    if a.agent.step_idx >= a.agent.num_steps:
                        a.update(total_steps)
                elif agent_decisions[agent_id] % update_every == 0:
                    a.update(total_steps)
        
        for agent_id in env.possible_agents:
            ep_rewards[agent_id].append(current_ep_rewards[agent_id])
        
        if (ep + 1) % max(1, args.total_episodes // 100) == 0:
            msg = f"Ep {ep+1}/{args.total_episodes} | Steps {total_steps}"
            for agent_id in env.possible_agents:
                avg_r = np.mean(ep_rewards[agent_id][-max(1, args.total_episodes // 10):])
                msg += f" | {agent_id}: {avg_r:.2f}"
            fps = total_steps / (time.time() - start_time)
            msg += f" | FPS {fps:.1f}"
            print(msg)
            
        if (ep + 1) % eval_interval == 0 or (ep + 1) == args.total_episodes:
            # Eval against random
            r0 = evaluate_vs_random(agents, args.ma_env, "player_0", num_episodes=args.eval_episodes)
            r1 = evaluate_vs_random(agents, args.ma_env, "player_1", num_episodes=args.eval_episodes)
            rand_scores_0.append(r0)
            rand_scores_1.append(r1)
            
            # Exploitability
            if args.ma_env == "rps":
                expl = get_rps_exploitability(agents)
            else:
                eval_policy = MAWrapperPolicy(game, agents)
                eval_policy._cache.clear() # Ensure a fresh cache for each eval
                expl = exploitability(game, eval_policy)
            
            exploitability_hist.append(expl)
            
            print(f"Eval Ep {ep+1}: vsRand(P0)={r0:.2f}, vsRand(P1)={r1:.2f}, Ext={expl:.4f}")

    env.close()
    
    results_dir = os.path.join("results", args.algo, args.env_name)
    os.makedirs(results_dir, exist_ok=True)
    
    for agent_id, rewards in ep_rewards.items():
        np.save(os.path.join(results_dir, f"train_scores_{agent_id}_{args.ablation}_seed{seed}.npy"), np.array(rewards))
        
    np.save(os.path.join(results_dir, f"exploitability_{args.ablation}_seed{seed}.npy"), np.array(exploitability_hist))
    np.save(os.path.join(results_dir, f"evaluate_vs_random_p0_{args.ablation}_seed{seed}.npy"), np.array(rand_scores_0))
    np.save(os.path.join(results_dir, f"evaluate_vs_random_p1_{args.ablation}_seed{seed}.npy"), np.array(rand_scores_1))
    
    if args.ma_env in ["tictactoe", "rps"]:
        p0_dist = agents["player_0"].action_dist_log
        p1_dist = agents["player_1"].action_dist_log
        if len(p0_dist) > 0:
            np.save(os.path.join(results_dir, f"action_dist_p0_{args.ablation}_seed{seed}.npy"), np.array(p0_dist, dtype=object))
        if len(p1_dist) > 0:
            np.save(os.path.join(results_dir, f"action_dist_p1_{args.ablation}_seed{seed}.npy"), np.array(p1_dist, dtype=object))
    
    return ep_rewards


if __name__ == "__main__":
    args = get_ma_args()
    seed = args.run - 1
    print(f"--- Running Seed {seed} (Run {args.run}) ---")
    if args.ma_env == "rps":
        train_rps_vectorized(args, seed=seed)
    else:
        train_ma(args, seed=seed)
