import argparse
import os
import time
import numpy as np
import torch
import pyspiel
from collections import deque
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
    # Parallel AEC environments for sequential games
    parser.add_argument("--num_ma_envs", type=int, default=32)
    # Gradient steps per round
    parser.add_argument("--rps_grad_steps", type=int, default=4)
    # Shared model: for PPO, share actor weights between players (each keeps its own
    # rollout buffer / value function); for DQN/SAC, share the full agent.
    # For tictactoe, P1's observations are augmented (planes 1↔2 swapped) so the
    # shared network always receives player-relative [empty|my|opp] inputs.
    parser.add_argument("--shared_model", action="store_true", default=False)

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
        # Higher soft-DQN temperature for MA: at the single-agent default (0.03)
        # softmax(Q/alpha) collapses to a near-deterministic policy (realized entropy
        # ~0.03 vs target ~0.99), so the entropy/Munchausen anchor stops biting and
        # exploitability diverges. Raise it so the played policy stays stochastic.
        dqn_alpha=0.1,
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

    def precompute_cache(self):
        """BFS traversal to find all unique info states and batch their neural network inferences."""
        self._cache = {}
        # player_id -> info_state_string -> (obs_tensor, legal_actions)
        player_states = {0: {}, 1: {}}
        
        # Standard BFS to find all reachable info states
        queue = deque([self.game.new_initial_state()])
        seen_states = set()
        
        while queue:
            state = queue.popleft()
            state_str = state.history_str()
            if state_str in seen_states:
                continue
            seen_states.add(state_str)
            
            if state.is_terminal():
                continue
            
            if state.is_chance_node():
                for outcome, _prob in state.chance_outcomes():
                    queue.append(state.child(outcome))
                continue
            
            player_id = state.current_player()
            infostate = state.information_state_string(player_id)
            
            if infostate not in player_states[player_id]:
                if self.game_type.provides_observation_tensor:
                    obs = np.array(state.observation_tensor(player_id), dtype=np.float32)
                else:
                    obs = np.array(state.information_state_tensor(player_id), dtype=np.float32)
                wrapper = self.agent_wrappers[f"player_{player_id}"]
                obs = wrapper._augment_obs(obs)
                player_states[player_id][infostate] = {
                    "obs": obs,
                    "legal": state.legal_actions(player_id)
                }
            
            for action in state.legal_actions():
                queue.append(state.child(action))

        # Batch inference per player
        for pid, states_dict in player_states.items():
            if not states_dict:
                continue
            
            infostates = list(states_dict.keys())
            obs_list = [states_dict[is_]["obs"] for is_ in infostates]
            legal_list = [states_dict[is_]["legal"] for is_ in infostates]
            
            wrapper = self.agent_wrappers[f"player_{pid}"]
            obs_t = torch.as_tensor(np.stack(obs_list), dtype=torch.float32, device=wrapper.device)
            
            # Create a batched mask
            mask_t = torch.zeros((len(infostates), wrapper.args.n_actions), dtype=torch.float32, device=wrapper.device)
            for i, legal in enumerate(legal_list):
                mask_t[i, legal] = 1.0
            
            with torch.no_grad():
                probs_t = wrapper._get_probs(obs_t, mask_t)
                probs_np = probs_t.cpu().numpy()
            
            for i, infostate in enumerate(infostates):
                legal = legal_list[i]
                dict_probs = {}
                total = 0.0
                for act in legal:
                    p = float(probs_np[i, act])
                    dict_probs[act] = max(1e-8, p)
                    total += dict_probs[act]
                
                if total > 0:
                    for act in legal:
                        dict_probs[act] /= total
                else:
                    for act in legal:
                        dict_probs[act] = 1.0 / len(legal)
                
                self._cache[(pid, infostate)] = dict_probs

    def action_probabilities(self, state, player_id=None):
        if player_id is None:
            player_id = state.current_player()
            
        if state.is_terminal():
            return {}
            
        infostate = state.information_state_string(player_id)
        if (player_id, infostate) in self._cache:
            return self._cache[(player_id, infostate)]
            
        # Fallback to single inference if not cached
        legal_actions = state.legal_actions(player_id)
        wrapper = self.agent_wrappers[f"player_{player_id}"]

        if self.game_type.provides_observation_tensor:
            obs = np.array(state.observation_tensor(player_id), dtype=np.float32)
        else:
            obs = np.array(state.information_state_tensor(player_id), dtype=np.float32)
        obs = wrapper._augment_obs(obs)

        mask_t = torch.zeros(wrapper.args.n_actions, dtype=torch.float32, device=wrapper.device)
        mask_t[legal_actions] = 1.0

        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=wrapper.device).unsqueeze(0)
        mask_t = mask_t.unsqueeze(0)
        
        with torch.no_grad():
            probs = wrapper._get_probs(obs_t, mask_t)
            probs_np = probs.cpu().numpy()[0]
        
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
                obs, _acc_reward, term, trunc, info = eval_env.last()
                # Use eval_env.rewards (instantaneous) not env.last() (Shimmy-accumulated).
                # Shimmy's dead-step env.step(None) re-accumulates the terminal reward for
                # all remaining agents, so the second dead-step agent always gets a doubled
                # value from last(). P1 is always second, so P1's scores would be 2× wrong.
                if agent_id == agent_id_to_eval:
                    ep_ret += eval_env.rewards.get(agent_id, 0.0)
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
                obs, _acc_reward, term, trunc, info = eval_env.last()
                # Same Shimmy dead-step doubling fix: use instantaneous rewards.
                if agent_id == agent_id_to_eval:
                    ep_ret += eval_env.rewards.get(agent_id, 0.0)
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
        probs0 = agents["player_0"]._get_probs(dummy_obs, dummy_mask).detach().cpu().numpy()[0]
        probs1 = agents["player_1"]._get_probs(dummy_obs, dummy_mask).detach().cpu().numpy()[0]
    
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
    def __init__(self, agent, algo, device, args, n_envs=1, player_id="player_0", augment_obs=False, no_learn=False):
        self.agent = agent
        self.algo = algo
        self.device = device
        self.args = args
        self.n_envs = n_envs
        self.player_id = player_id
        # When True, swap tic-tac-toe observation planes 1↔2 before any network call
        # so the shared actor always sees [empty|my_pieces|opp_pieces].
        self.augment_obs = augment_obs
        # When True, observe() and observe_batch() are no-ops: the wrapper acts (using
        # the shared actor) but never stores transitions or triggers gradient updates.
        # Used for P1 in shared-model PPO to eliminate zero-sum gradient rotation.
        self.no_learn = no_learn
        self.last_update_info = None  # latest update() info dict (for MMD diagnostics)
        self.action_dist_log = [] # List of policy distributions for Tic-Tac-Toe
        # One FIFO per parallel env. The underlying agent buffers are vectorized
        # [n_envs, ...] and a single add() writes a full row of n_envs transitions.
        # AEC games are async (each env advances at its own pace and resets
        # independently), so we keep each env's transition stream in its own column
        # and only emit a synchronized [n_envs] row once every column has a pending
        # transition. This preserves per-column temporal order, which PPO's GAE /
        # next_obs shift relies on; off-policy replay (DQN/SAC) samples across both
        # dims so it is correct either way.
        self._env_queues = [[] for _ in range(n_envs)]

    def _augment_obs(self, obs):
        """Player-relative observation augmentation for shared-model P1.
        TTT: swap planes 1↔2 (obs[9:18] ↔ obs[18:27]).
          Converts absolute [empty|P0_pieces|P1_pieces] to relative [empty|my|opp].
        Leduc: swap player indicator (obs[0]↔obs[1]) and chip totals (obs[14]↔obs[15]).
          Converts absolute [P0_id, P1_id, ..., P0_chips, P1_chips] to
          player-relative [my_id=1, opp_id=0, ..., my_chips, opp_chips].
        No-op when augment_obs is False (player_0, non-shared model, or RPS)."""
        if not self.augment_obs:
            return obs
        env = getattr(self.args, "ma_env", "tictactoe")
        if isinstance(obs, torch.Tensor):
            obs = obs.clone()
            if env == "tictactoe":
                if obs.ndim == 1:
                    tmp = obs[9:18].clone()
                    obs[9:18] = obs[18:27]
                    obs[18:27] = tmp
                else:
                    tmp = obs[:, 9:18].clone()
                    obs[:, 9:18] = obs[:, 18:27]
                    obs[:, 18:27] = tmp
            elif env == "leduc":
                if obs.ndim == 1:
                    tmp0 = obs[0].clone()
                    obs[0] = obs[1]; obs[1] = tmp0
                    tmp14 = obs[14].clone()
                    obs[14] = obs[15]; obs[15] = tmp14
                else:
                    tmp0 = obs[:, 0].clone()
                    obs[:, 0] = obs[:, 1]; obs[:, 1] = tmp0
                    tmp14 = obs[:, 14].clone()
                    obs[:, 14] = obs[:, 15]; obs[:, 15] = tmp14
        else:
            obs = np.array(obs, copy=True)
            if env == "tictactoe":
                if obs.ndim == 1:
                    tmp = obs[9:18].copy()
                    obs[9:18] = obs[18:27]
                    obs[18:27] = tmp
                else:
                    tmp = obs[:, 9:18].copy()
                    obs[:, 9:18] = obs[:, 18:27]
                    obs[:, 18:27] = tmp
            elif env == "leduc":
                if obs.ndim == 1:
                    obs[[0, 1]] = obs[[1, 0]]
                    obs[[14, 15]] = obs[[15, 14]]
                else:
                    obs[:, [0, 1]] = obs[:, [1, 0]]
                    obs[:, [14, 15]] = obs[:, [15, 14]]
        return obs

    def get_action(self, obs, mask, deterministic=False, step=0, total_steps=1000000, log_dist=False):
        obs = self._augment_obs(obs)
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
        
        if mask_t.ndim == 1:
            mask_t = mask_t.unsqueeze(0)

        B = obs_t.shape[0]

        if self.algo == "dqn":
            if hasattr(self.agent, "n_quantiles"):
                # IQN: mean over quantiles
                taus = self.agent._sample_taus(B, self.agent.n_quantiles, self.device)
                q = self.agent.ext_online(obs_t, taus, normalized=True)
                # q shape is [B, n_quantiles, 1, n_actions] or [B, n_quantiles, n_actions]
                q = q.view(B, self.agent.n_quantiles, -1).mean(dim=1)
            else:
                q = self.agent.ext_online(obs_t, normalized=True)
                q = q.view(B, -1)
            
            # Mask illegal actions
            q[mask_t == 0] = -1e9
            if self.agent.soft or self.agent.munchausen:
                return torch.softmax(q / self.agent.alpha, dim=-1)
            else:
                # Epsilon-greedy approx distribution
                probs = torch.zeros_like(q)
                best_act = torch.argmax(q, dim=-1)
                eps = getattr(self.agent, "last_eps", 0.05)
                valid_counts = mask_t.sum(dim=-1, keepdim=True)
                
                # Set baseline probability for all valid actions
                probs = (mask_t * eps) / valid_counts
                # Add (1-eps) to the best action
                probs.scatter_add_(1, best_act.unsqueeze(1), torch.ones((B, 1), device=self.device) * (1.0 - eps))
                return probs
                
        elif self.algo == "ppo":
            logits = self.agent.actor(obs_t)
            logits[mask_t == 0] = -1e9
            return torch.softmax(logits, dim=-1)
            
        elif self.algo == "sac":
            # SAC MC estimation
            n_samples = getattr(self, "sac_mc_samples", 256)
            n_actions = mask_t.shape[-1]
            with torch.no_grad():
                # obs_t: [B, D] -> [B*n_samples, D]
                obs_rep = obs_t.repeat_interleave(n_samples, dim=0)
                sampled, _, _ = self.agent.actor.get_action(obs_rep)  # [B*N, n_actions]
                sampled = sampled.clone()
                # mask_t: [B, n_actions] -> [B*n_samples, n_actions]
                mask_rep = mask_t.repeat_interleave(n_samples, dim=0)
                sampled[mask_rep == 0] = -1e9
                choices = torch.argmax(sampled, dim=-1)  # [B*N]
                
                # Reshape and count
                choices = choices.view(B, n_samples)
                probs = torch.zeros((B, n_actions), device=self.device)
                for b in range(B):
                    counts = torch.bincount(choices[b], minlength=n_actions).float()
                    probs[b] = counts / n_samples
            return probs

    def observe(self, env_idx, obs, action, reward, next_obs, term, trunc, logprob=None, action_mask=None):
        if self.no_learn:
            return
        # Queue the transition in its env's column, then emit any full rows.
        self._env_queues[env_idx].append({
            "obs": self._augment_obs(obs),
            "action": action,
            "reward": reward,
            "next_obs": self._augment_obs(next_obs),
            "term": term,
            "trunc": trunc,
            "logprob": logprob,
            "mask": action_mask
        })
        self.flush_transitions()

    def flush_transitions(self):
        # Emit a synchronized [n_envs] row whenever every env column has a pending
        # transition, popping one transition from each column per row.
        is_ppo = (self.algo == "ppo")
        while all(len(q) > 0 for q in self._env_queues):
            # PPO's rollout buffer is fixed-length [num_steps, n_envs]; stop filling
            # once it is full and let the training loop trigger update() (which resets
            # step_idx) before we drain the remaining queued rows.
            if is_ppo and self.agent.step_idx >= self.agent.num_steps:
                break

            batch = [q.pop(0) for q in self._env_queues]

            obs = np.stack([d["obs"] for d in batch])
            raw = np.stack([d["action"] for d in batch])
            reward = np.array([d["reward"] for d in batch])
            next_obs = np.stack([d["next_obs"] for d in batch])
            term = np.array([d["term"] for d in batch])
            trunc = np.array([d["trunc"] for d in batch])
            logprob = np.array([d["logprob"] for d in batch]) if batch[0]["logprob"] is not None else None
            mask = np.stack([d["mask"] for d in batch]) if batch[0]["mask"] is not None else None

            self.observe_batch(obs, raw, reward, next_obs, term, trunc, logprob, mask)

    def get_action_batch(self, obs_t, mask_t, step=0, total_steps=1000000, log_dist=False):
        """Batched action selection for the vectorized (simultaneous) RPS path.
        obs_t: (B, obs_dim), mask_t: (B, n_actions), both torch tensors on device.
        Returns (env_actions[B] long tensor, raw_for_buffer, logprob_or_None)."""
        obs_t = self._augment_obs(obs_t)
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
            # Discrete env mapped to Box(n_actions): pick the highest-valued *legal*
            # dim. Mask illegal actions before argmax (otherwise SAC can emit an
            # illegal move, e.g. raising past the Leduc max). Store the full vector.
            mask_np = mask_t.detach().cpu().numpy()
            masked = np.where(mask_np > 0, action_np, -np.inf)
            env_act = torch.as_tensor(np.argmax(masked, axis=1), device=obs_t.device).long()
            return env_act, action_np, None
        elif self.algo == "ppo":
            action, logprob = self.agent.sample_action(obs_t, action_mask=mask_t)
            return action.long().reshape(B), action, logprob
        return None, None, None

    def observe_batch(self, obs, raw, reward, next_obs, term, trunc, logprob=None, mask=None):
        """Batched transition store for the vectorized RPS path. All array-likes have a
        leading batch (num_envs) dimension; tensors are moved to CPU numpy at the boundary."""
        if self.no_learn:
            return
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
        # Ensure any leftover transitions are flushed before update
        self.flush_transitions()

        if self.algo == "ppo":
            info = self.agent.update(global_step=global_step)
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
                info = self.agent.update(batch_size=batch_size, step=global_step)
            else:
                info = self.agent.update(batch_size=batch_size, global_step=global_step)

        # Keep the latest non-empty update info for MMD diagnostics (alpha / Munchausen
        # magnet magnitude for DQN; entropy coef / approx-KL trust region for PPO).
        # update() returns a bare float on burn-in / loss-only paths -- ignore those.
        if isinstance(info, dict) and info:
            self.last_update_info = info
        return info

def flatten_obs(obs):
    if isinstance(obs, dict):
        return obs["observation"].flatten()
    return obs.flatten()

def _mean_discrete_entropy(eval_policy):
    """Mean entropy (nats) of each player's *played* discrete policy over the game's
    info states, read straight from the precomputed exploitability cache (so no extra
    forward passes). This is the MMD-relevant quantity: entropy of the realized policy
    that actually generates play. If the temperature/magnet is doing its job this stays
    bounded away from 0; collapse toward 0 means the regularizer stopped anchoring."""
    sums = {0: 0.0, 1: 0.0}
    counts = {0: 0, 1: 0}
    for (pid, _infostate), probs in eval_policy._cache.items():
        p = np.array(list(probs.values()), dtype=np.float64)
        p = p[p > 0]
        if p.size == 0:
            continue
        sums[pid] += float(-(p * np.log(p)).sum())
        counts[pid] += 1
    return {pid: (sums[pid] / counts[pid] if counts[pid] else 0.0) for pid in sums}

def _reg_diagnostics(agents, algo):
    """Per-player effective regularization strength from each agent's latest update.
    DQN: alpha (entropy temperature), munchausen coef + realized magnet reward |mr|
    (the proximal/mirror term), Beta. PPO: ent_coef, approx_kl (realized trust-region /
    mirror step size), clipfrac."""
    out = {}
    for pid, tag in [("player_0", "P0"), ("player_1", "P1")]:
        info = agents[pid].last_update_info or {}
        agent = agents[pid].agent
        if algo == "dqn":
            # DQN's update() PRINTS its diagnostic dict but RETURNS a float, so read the
            # live regularization knobs straight off the agent. alpha = entropy
            # temperature (softmax(Q/alpha)); compare target_ent to the realized Hpi to
            # see whether the alpha-autotuner is actually holding the entropy target.
            out[tag] = {
                "alpha": float(getattr(agent, "alpha", 0.0)),
                "munch_c": float(getattr(agent, "munchausen_constant", 0.0)),
                "target_ent": float(getattr(agent, "target_entropy", 0.0)),
                "beta": float(getattr(agent, "Beta", 0.0)),
            }
        elif algo == "ppo":
            kl = info.get("approx_kl", 0.0)
            kl = float(kl.item()) if hasattr(kl, "item") else float(kl)
            out[tag] = {
                "ent_coef": float(getattr(agent, "ent_coef", 0.0)),
                "approx_kl": kl,
                "clipfrac": float(info.get("clipfrac", 0.0)),
            }
    return out

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

    shared_model = getattr(args, "shared_model", False)
    if shared_model and args.algo == "ppo":
        a0, _ = build_agent(args, mock_env, device)
        if args.ent_coef_override is not None:
            a0.ent_coef = args.ent_coef_override
        a1, _ = build_agent(args, mock_env, device)
        if args.ent_coef_override is not None:
            a1.ent_coef = args.ent_coef_override
        lr = a1.optimizer.param_groups[0]["lr"]
        eps_adam = a1.optimizer.param_groups[0]["eps"]
        a1.actor = a0.actor
        a1.optimizer = torch.optim.Adam(
            list(a0.actor.parameters()) + a1._get_ext_critic_params(),
            lr=lr, eps=eps_adam,
        )
        agent_raws = [a0, a1]
    elif shared_model:
        a0, _ = build_agent(args, mock_env, device)
        agent_raws = [a0, a0]
    else:
        agent_raws = []
        for _ in range(2):
            a_raw, _ = build_agent(args, mock_env, device)
            if args.ent_coef_override is not None and args.algo == "ppo":
                a_raw.ent_coef = args.ent_coef_override
            agent_raws.append(a_raw)
    agents = {f"player_{i}": MAAgentWrapper(agent_raws[i], args.algo, device, args, player_id=f"player_{i}") for i in range(2)}

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

        # Each round collects a fresh batch of B games.
        for pid in ["player_0", "player_1"]:
            a = agents[pid]
            if args.algo == "ppo":
                if not shared_model:
                    if a.agent.step_idx >= a.agent.num_steps:
                        a.update(total_steps)
                # Shared-model PPO: synchronized block below.
            else:
                # DQN/SAC with shared model: only P0 drives gradient steps
                if not (shared_model and pid == "player_1"):
                    for _ in range(grad_steps):
                        a.update(total_steps)

        # Synchronized PPO update for shared-model RPS
        if args.algo == "ppo" and shared_model:
            all_full = all(
                agents[pid].agent.step_idx >= agents[pid].agent.num_steps
                for pid in ["player_0", "player_1"]
            )
            if all_full:
                update_order = ["player_0", "player_1"]
                random.shuffle(update_order)
                for pid in update_order:
                    agents[pid].update(total_steps)

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

    results_base = "results_shared" if getattr(args, "shared_model", False) else "results"
    results_dir = os.path.join(results_base, args.algo, args.env_name)
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
    
    # Standardize obs_dim using a temporary env
    temp_env = OpenSpielCompatibilityV0(game_name=game_name)
    if args.ma_env == "tictactoe":
        obs_dim = 27
    elif args.ma_env == "leduc":
        obs_dim = 16
    else:
        obs_dim = temp_env.observation_space("player_0").shape[0] if len(temp_env.observation_space("player_0").shape) > 0 else 1
    n_actions = temp_env.action_space("player_0").n
    possible_agents = temp_env.possible_agents
    temp_env.close()

    args.n_actions = n_actions
    mock_env = SimpleNamespace(
        single_observation_space=gym.spaces.Box(low=0, high=1, shape=(obs_dim,), dtype=np.float32),
        single_action_space=gym.spaces.Discrete(n_actions),
        num_envs=args.num_ma_envs if args.ma_env != "rps" else 1 # Vectorized RPS handles its own
    )
    
    # Smaller buffers for tiny games
    if args.ma_env in ["tictactoe", "rps"]:
        args.dqn_buffer_size = min(getattr(args, "dqn_buffer_size", 10000), 10000)
        args.buffer_size = min(getattr(args, "buffer_size", 10000), 10000)

    # Build agents (optionally sharing weights between players)
    shared_model = getattr(args, "shared_model", False)
    if shared_model and args.algo == "ppo":
        # PPO: two separate agents (separate rollout buffers + critics) but shared actor.
        # Both optimizers reference the same actor parameters so gradient updates from
        # either player propagate through the shared weights.
        a0, _ = build_agent(args, mock_env, device)
        if args.ent_coef_override is not None:
            a0.ent_coef = args.ent_coef_override
        a1, _ = build_agent(args, mock_env, device)
        if args.ent_coef_override is not None:
            a1.ent_coef = args.ent_coef_override
        lr = a1.optimizer.param_groups[0]["lr"]
        eps_adam = a1.optimizer.param_groups[0]["eps"]
        a1.actor = a0.actor
        a1.optimizer = torch.optim.Adam(
            list(a0.actor.parameters()) + a1._get_ext_critic_params(),
            lr=lr, eps=eps_adam,
        )
        agent_raws = [a0, a1]
    elif shared_model:
        # DQN/SAC: share the full agent (one replay buffer, one network).
        a0, _ = build_agent(args, mock_env, device)
        agent_raws = [a0, a0]
    else:
        agent_raws = []
        for _ in range(len(possible_agents)):
            a_raw, _ = build_agent(args, mock_env, device)
            if args.ent_coef_override is not None and args.algo == "ppo":
                a_raw.ent_coef = args.ent_coef_override
            agent_raws.append(a_raw)

    num_envs = args.num_ma_envs
    agents = {
        agent_id: MAAgentWrapper(
            agent_raws[i], args.algo, device, args, n_envs=num_envs,
            player_id=agent_id,
            augment_obs=(shared_model and args.ma_env in ["tictactoe", "leduc"] and agent_id == "player_1"),
        )
        for i, agent_id in enumerate(possible_agents)
    }

    # Parallel Envs
    envs = [OpenSpielCompatibilityV0(game_name=game_name) for _ in range(num_envs)]
    for e in envs: e.reset()
    
    # Iterators for AEC
    iters = [iter(e.agent_iter()) for e in envs]
    # Active agent per env
    active_agents = [next(it) for it in iters]
    
    total_steps = 0
    episodes_done = 0
    ep_rewards = {agent_id: [] for agent_id in possible_agents}
    exploitability_hist = []
    rand_scores_0 = []
    rand_scores_1 = []
    entropy_hist = []   # [H_pi(P0), H_pi(P1)] per eval (realized discrete-policy entropy)
    reg_hist = []       # per-eval dict of effective regularization knobs (dqn/ppo)

    # State tracking per environment
    current_ep_rewards = [{aid: 0.0 for aid in possible_agents} for _ in range(num_envs)]
    pending_reward = [{aid: 0.0 for aid in possible_agents} for _ in range(num_envs)]
    last_data = [
        {aid: {"obs": None, "action": None, "logprob": None, "mask": None} for aid in possible_agents}
        for _ in range(num_envs)
    ]
    
    start_time = time.time()
    update_every = max(1, int(getattr(args, "update_every", 4)))
    agent_decisions = {agent_id: 0 for agent_id in possible_agents}
    # Threshold-based logging/eval: with many parallel envs episodes_done advances in
    # bursts and skips exact multiples, so an `episodes_done % interval == 0` check can
    # silently never fire. Trigger whenever we cross the next threshold instead.
    eval_interval = max(1, args.total_episodes // 10)
    next_eval_at = eval_interval

    primary_player = possible_agents[0]  # P0 drives DQN/SAC updates for shared model

    while episodes_done < args.total_episodes:
        # Group environments by the player who needs to act
        player_to_envs = {aid: [] for aid in possible_agents}
        for env_idx, aid in enumerate(active_agents):
            if aid is not None:
                player_to_envs[aid].append(env_idx)

        # Step each player who has active environments
        for acting_player, env_indices in player_to_envs.items():
            if not env_indices:
                continue
            
            obs_list = []
            mask_list = []
            valid_env_indices = []
            
            # 1. Collect observations and handle transitions from previous steps
            for env_idx in env_indices:
                env = envs[env_idx]
                obs, _last_reward, termination, truncation, info = env.last()
                
                # Accrue rewards
                reward = pending_reward[env_idx][acting_player]
                pending_reward[env_idx][acting_player] = 0.0
                current_ep_rewards[env_idx][acting_player] += reward
                
                # Observe previous transition if it exists
                if last_data[env_idx][acting_player]["obs"] is not None:
                    agents[acting_player].observe(
                        env_idx,
                        last_data[env_idx][acting_player]["obs"],
                        last_data[env_idx][acting_player]["action"],
                        reward,
                        obs.flatten(),
                        termination,
                        truncation,
                        logprob=last_data[env_idx][acting_player]["logprob"],
                        action_mask=last_data[env_idx][acting_player]["mask"]
                    )
                
                if termination or truncation:
                    # Terminal reached. These games pay out only at the end, so EVERY
                    # player holding an outstanding action must still see its final
                    # transition -- not just whichever player agent_iter surfaced first.
                    # We reset immediately below (instead of stepping the AEC dead-steps
                    # for every agent), so deliver the other players' terminal transitions
                    # here. Skipping this drops their terminal reward entirely, which made
                    # the consistently-second player (P1) never learn win/loss.
                    for aid in possible_agents:
                        if aid == acting_player:
                            continue
                        if last_data[env_idx][aid]["obs"] is not None:
                            r_aid = pending_reward[env_idx][aid]
                            pending_reward[env_idx][aid] = 0.0
                            current_ep_rewards[env_idx][aid] += r_aid
                            agents[aid].observe(
                                env_idx,
                                last_data[env_idx][aid]["obs"],
                                last_data[env_idx][aid]["action"],
                                r_aid,
                                last_data[env_idx][aid]["obs"],  # next_obs unused (term=True)
                                True,
                                bool(truncation),
                                logprob=last_data[env_idx][aid]["logprob"],
                                action_mask=last_data[env_idx][aid]["mask"],
                            )

                    # Episode ended for this environment
                    for aid in possible_agents:
                        ep_rewards[aid].append(current_ep_rewards[env_idx][aid])

                    episodes_done += 1
                    # Reset environment
                    env.reset()
                    iters[env_idx] = iter(env.agent_iter())
                    active_agents[env_idx] = next(iters[env_idx])
                    # Reset state tracking for this env
                    current_ep_rewards[env_idx] = {aid: 0.0 for aid in possible_agents}
                    pending_reward[env_idx] = {aid: 0.0 for aid in possible_agents}
                    last_data[env_idx] = {aid: {"obs": None, "action": None, "logprob": None, "mask": None} for aid in possible_agents}
                else:
                    obs_list.append(obs.flatten())
                    mask_list.append(info["action_mask"])
                    valid_env_indices.append(env_idx)

            if not obs_list:
                continue
            
            # 2. Batched Action Selection
            B_real = len(obs_list)
            obs_t = torch.as_tensor(np.stack(obs_list), dtype=torch.float32, device=device)
            mask_t = torch.as_tensor(np.stack(mask_list), dtype=torch.float32, device=device)
            
            log_this = (episodes_done % 100 == 0)
            env_acts, raw_acts, logprobs = agents[acting_player].get_action_batch(
                obs_t, mask_t, step=total_steps, total_steps=args.total_steps, log_dist=log_this
            )
            
            env_acts_np = env_acts.cpu().numpy()
            raw_acts_np = raw_acts if isinstance(raw_acts, np.ndarray) else raw_acts.cpu().numpy()
            logprobs_np = logprobs.cpu().numpy() if logprobs is not None else [None] * B_real

            # 3. Step environments
            for i, e_idx in enumerate(valid_env_indices):
                env = envs[e_idx]
                act = int(env_acts_np[i])
                
                last_data[e_idx][acting_player]["obs"] = obs_list[i]
                last_data[e_idx][acting_player]["action"] = raw_acts_np[i]
                last_data[e_idx][acting_player]["logprob"] = logprobs_np[i]
                last_data[e_idx][acting_player]["mask"] = mask_list[i]
                
                env.step(act)
                total_steps += 1
                
                for aid2, rv in env.rewards.items():
                    pending_reward[e_idx][aid2] += float(rv)
                
                try:
                    active_agents[e_idx] = next(iters[e_idx])
                except StopIteration:
                    active_agents[e_idx] = None

            # 4. Periodic Updates
            agent_decisions[acting_player] += B_real
            a = agents[acting_player]
            if args.algo == "ppo":
                if not shared_model:
                    # Independent agents: each player updates its own actor when buffer full.
                    if a.agent.step_idx >= a.agent.num_steps:
                        a.update(total_steps)
                # Shared-model PPO update is handled in the synchronized block below.
            else:
                # Off-policy (DQN/SAC): keep the SAME gradient-step-to-experience ratio
                # as the single-env runner, i.e. one update per `update_every` decisions.
                # A round collects B_real decisions across the parallel envs, so do
                # floor(decisions / update_every) updates and carry the remainder.
                # (Gating on `update_every * num_envs` did only ONE update per ~128
                #  decisions -> num_envs x too few updates, so the agents barely learned.)
                # With a shared model, ONLY the primary player drives gradient steps so the
                # shared network does not get 2× updates (one per player per round).
                if not (shared_model and acting_player != primary_player):
                    n_updates = agent_decisions[acting_player] // update_every
                    if n_updates > 0:
                        agent_decisions[acting_player] -= n_updates * update_every
                        for _ in range(n_updates):
                            a.update(total_steps)

        # Synchronized PPO update for shared model: only update when BOTH players'
        # rollout buffers are full so neither player's logprobs are stale relative to
        # the other's update. Randomize order so neither player is systematically first.
        if args.algo == "ppo" and shared_model:
            all_full = all(
                agents[pid].agent.step_idx >= agents[pid].agent.num_steps
                for pid in possible_agents
            )
            if all_full:
                update_order = list(possible_agents)
                random.shuffle(update_order)
                for pid in update_order:
                    agents[pid].update(total_steps)

        # Periodic Logging & Eval (threshold-based; robust to bursty episodes_done)
        if episodes_done >= next_eval_at:
            # Advance past the current count so the next trigger is one interval later
            # even if this iteration jumped over several thresholds at once.
            next_eval_at = (episodes_done // eval_interval + 1) * eval_interval

            fps = total_steps / (time.time() - start_time)
            print(f"Ep ~{episodes_done}/{args.total_episodes} | Steps {total_steps} | FPS {fps:.1f}")

            r0 = evaluate_vs_random(agents, args.ma_env, "player_0", num_episodes=args.eval_episodes)
            r1 = evaluate_vs_random(agents, args.ma_env, "player_1", num_episodes=args.eval_episodes)
            rand_scores_0.append(r0)
            rand_scores_1.append(r1)

            eval_policy = None
            if args.ma_env == "rps":
                expl = get_rps_exploitability(agents)
            else:
                eval_policy = MAWrapperPolicy(game, agents)
                eval_policy.precompute_cache()
                expl = exploitability(game, eval_policy)
            exploitability_hist.append(expl)
            print(f"Eval: vsRand(P0)={r0:.2f}, vsRand(P1)={r1:.2f}, Ext={expl:.4f}")

            # MMD diagnostics: is the regularizer actually anchoring the policy?
            # H_pi = realized discrete-policy entropy (should stay bounded away from 0
            # while contracting; collapse to 0 = magnet/temperature stopped biting).
            if eval_policy is not None:
                ent = _mean_discrete_entropy(eval_policy)
                entropy_hist.append([ent[0], ent[1]])
                msg = f"  MMD: Hpi(P0)={ent[0]:.3f} Hpi(P1)={ent[1]:.3f}"
                if args.algo in ("dqn", "ppo"):
                    reg = _reg_diagnostics(agents, args.algo)
                    reg_hist.append(reg)
                    d0, d1 = reg["P0"], reg["P1"]
                    if args.algo == "dqn":
                        # alpha = entropy temperature; target_ent = entropy the autotuner
                        # is supposed to hold. Hpi << target_ent with alpha not rising =>
                        # the magnet/temperature is not anchoring (MMD assumption broken).
                        msg += (f" | alpha {d0['alpha']:.3f}/{d1['alpha']:.3f}"
                                f" target_ent {d0['target_ent']:.3f}"
                                f" munch_c {d0['munch_c']:.2f} Beta {d0['beta']:.3f}")
                    else:
                        # approx_kl = realized trust-region / mirror step size.
                        msg += (f" | ent_coef {d0['ent_coef']:.3f}"
                                f" approxKL {d0['approx_kl']:.4f}/{d1['approx_kl']:.4f}"
                                f" clipfrac {d0['clipfrac']:.2f}/{d1['clipfrac']:.2f}")
                print(msg)

    for e in envs: e.close()

    results_base = "results_shared" if getattr(args, "shared_model", False) else "results"
    results_dir = os.path.join(results_base, args.algo, args.env_name)
    os.makedirs(results_dir, exist_ok=True)
    for agent_id, rewards in ep_rewards.items():
        np.save(os.path.join(results_dir, f"train_scores_{agent_id}_{args.ablation}_seed{seed}.npy"), np.array(rewards))
    np.save(os.path.join(results_dir, f"exploitability_{args.ablation}_seed{seed}.npy"), np.array(exploitability_hist))
    np.save(os.path.join(results_dir, f"evaluate_vs_random_p0_{args.ablation}_seed{seed}.npy"), np.array(rand_scores_0))
    np.save(os.path.join(results_dir, f"evaluate_vs_random_p1_{args.ablation}_seed{seed}.npy"), np.array(rand_scores_1))
    # MMD diagnostics history (realized discrete-policy entropy + effective regularization)
    if entropy_hist:
        np.save(os.path.join(results_dir, f"policy_entropy_{args.ablation}_seed{seed}.npy"), np.array(entropy_hist))
    if reg_hist:
        np.save(os.path.join(results_dir, f"reg_diag_{args.ablation}_seed{seed}.npy"), np.array(reg_hist, dtype=object))

    return ep_rewards


if __name__ == "__main__":
    args = get_ma_args()
    seed = args.run - 1
    print(f"--- Running Seed {seed} (Run {args.run}) ---")
    if args.ma_env == "rps":
        train_rps_vectorized(args, seed=seed)
    else:
        train_ma(args, seed=seed)
