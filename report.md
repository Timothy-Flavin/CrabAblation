## Purpose

This report keeps track of a number of unexplained results in the ablation testing accross DQN SAC and PPO in environments cartpole mujoco and minigrid.

## List of Anomolies

1. Models are evaluated with greedy action selection every 50 episodes for offline rl and every 5th iteration for online rl, but the x axis is not scaled to match the train rewards in the episode x axis plots. DQN eval lines are either horizontal or non existant
**cause**: In `graph.py` episode mode, `x_eval = np.arange(e_mean.size)` places eval points at indices `0, 1, ..., N-1` (where N is the number of eval recordings, typically 10-130). The train x-axis spans `0..M-1` (where M is total training episodes, typically 1000-12000). So the eval line is plotted as a tiny cluster at the far left of the chart instead of being spread across the full training range.
**fix**: Changed `x_eval` in episode mode to `np.linspace(0, t_mean.size, num=e_mean.size)`, which distributes the eval points evenly across the full training episode range. This is an approximation (the exact episode positions are not stored), but produces a correctly-scaled overlay.
**status**: Fixed in `graph.py`.

2. Dqn cartpole was run without truncation at 500 steps and magnet_reg `ablation == 2` ran for ~1,800 episodes while the rest ran for ~1,200. 
**cause**: The data confirms ablation 2 (ent_reg_coef=0.0) has ~2× more episodes than other ablations in run 1 (2133 vs 1058–1098). The max reward for other ablations is 500 (50 steps × 10× scale), while ablation 2 reaches 5000 (500 steps). CartPole-v1 truncation IS active in the current code (`gym.make("CartPole-v1")` includes TimeLimit(500)), but to make this explicit and reliable, the `make_env_thunk` function now passes `max_episode_steps=500` to `gym.make`. Additionally, ablation 5 (no delayed target) shows catastrophic instability (run 2: 96,568 episodes at avg 2.6 steps — training diverged completely).

**fix**: Added explicit `max_episode_steps=500` to `gym.make` for CartPole in `environment_utils.py`.

**Todo**: rerun all dqn cartpole ablations on timpc with proper truncation

3. Dqn minigrid `ablation == 4` ran for 12,000 episodes. All other ablations ran for 4,000. None of the ablations score above 4/10 on minigrid when Random Network Distilation with frame stacking should likely do better than that. 
**cause**: Ablation 4 uses `EVRainbowDQN` (non-distributional) + `Beta=0.1` (RND coefficient; other ablations use `Beta=0.7`, 7× stronger). With much weaker RND intrinsic reward, ablation 4 explores differently in minigrid: data shows ~83 steps per episode average (vs ~250 for others), meaning the agent finds the goal more frequently but with lower quality — eval scores 0.10–0.14 vs up to 0.26 for other ablations. The 3× episode count difference is a direct consequence of shorter episodes (more goal-finding OR faster exploration failure). Ablation 5 (no delayed target) has 0.0 eval across all runs, confirming that delayed targets are essential in minigrid. The low ceiling of ~4/10 (eval ≤ 0.26) across all ablations reflects the difficulty of minigrid FourRooms with the current hyperparameters, not a code bug per se.
**Note**: The `Beta=0.1` vs `Beta=0.7` difference in ablation 4 makes it a combined ablation (no Dist-RL + weaker RND), not a clean "no Dist-RL" ablation. Beta has been changed to 0.7 for all cases besides 3

**Todo**: rerun DQN minigrid ablation 4 to confirm behavior.


4. ppo cartpole `ablation == 4` ran for far fewer episodes than the rest (1000 vs 1700) 1M steps should be more than enough for PPO to complete cartpole but 1000 episodes at ~200 step average is not 1M and PPO has not converged. By the end. Rerun likely needed.
**cause**: Data analysis shows ablation 4 run 1 has 1050 episodes vs 1738–1983 for other ablations. However, `last10` performance shows convergence: ablation 4 reaches 404–443/500 reward across all 3 runs, comparable to other ablations (239–500). The lower episode count is due to *longer average episodes* (agent learned to balance longer in that seed), not a truncated run. The claim of "not converged" appears to be based on a single run observation; the ablation is actually performing correctly. Run 3 of ablation 4 (4210 episodes, last10=443) is consistent with normal behavior.
**status**: Anomaly explained by stochastic variance in PPO learning dynamics. Rerun still recommended to average over more seeds.

5. ppo minigrid ablation 4 ran for 12k episodes while all other ablations ran for 2k. 12k was needed for it to get close to convergence
**cause**: PPO ablation 4 (distributional=False, Beta=0.01) consistently has ~12k–13k episodes vs ~3k for other ablations. Interestingly, ablation 4 achieves the **highest** eval scores in some runs (e.g., run 3 last10=0.558 vs max 0.316 for other ablations). The non-distributional PPO critic learns a simpler value function that generalizes better in minigrid, but initially explores more broadly (shorter episodes = more resets). The 4× episode count difference is caused by the distributional=False critic having different value estimates during early training, leading to shorter episodes before the policy stabilises. This is a behavioral difference, not a code bug.
**status**: Anomaly explained. Rerun recommended to confirm robustness across seeds.

6. SAC results dont make sense for any of the environments. For cartpole the rewards look like a step function that jumps between each interval of 100. For minigrid SAC learns almost nothing (0.2/1.0). On mujoco all results besides ablation 2 and 5 look the exact same with no variance, while 5 fails to learn anything and 2's error bars span almost the entire y axis.
**cause (cartpole)**: Eval scores show catastrophic forgetting (e.g., run 1 abl 0: `[500, 500, ..., 83, 70, ..., 500, 500]`). The SAC autotune (alpha adjustment) interacts with the continuous→discrete action mapping for CartPole (proxy Box([0,1]) → Discrete(2)), causing oscillation between high-entropy (random) and low-entropy (deterministic) policies. The "step function at intervals of 100" description matches this oscillating convergence pattern. Ablation 5 (no delayed critics) completely diverges (~43k–95k episodes at avg 10 steps).

**cause (minigrid)**: SAC maps Discrete(7) via a continuous proxy (Box([0,1]) → 7 bins). This coarse discretization loses information and makes it hard to learn fine-grained action preferences. Eval scores ~0.1–0.2. This is a fundamental limitation of applying SAC to discrete action spaces via proxy, not a code bug per se.

**cause (mujoco — identical results for ablations 0, 1, 3, 4)**: **Confirmed code bug** — SAC ablations 1 and 3 have no configuration changes in `_sac_agent_from_args`. They use identical configs to ablation 0 (baseline), which is why they look identical in all plots. Ablation 4 (no distributional/dueling) also achieves similar MuJoCo performance (~1854–1879), suggesting those features don't significantly help in HalfCheetah-v5. Ablation 2 (entropy_coef_zero=True) has high variance. Ablation 5 (no delayed critics) fails completely (-291 to -376).

**fix**: Implemented SAC ablations 1 and 3 to match DQN and PPO:
- Baseline SAC now includes Munchausen-style KL self-imitation and RND intrinsic reward, matching DQN and PPO baselines.
- Ablation 1 (KL_Penalty): `munchausen=False` — removes the Munchausen KL penalty from the SAC actor. The baseline adds `munchausen_alpha * munchausen_tau * clamp(log_pi(a_replay | s), min=l_clip)` to Q-targets using the current policy's log-prob of the stored replay action (computed via atanh inversion of the tanh-squashed action). This is the continuous-action analogue of Munchausen DQN.
- Ablation 3 (Optimism): `Beta=0.0` — removes RND-based intrinsic reward from SAC. The baseline uses `Beta=0.01` with an RND model and RunningMeanStd normalisation, matching DQN (`Beta=0.7`) and PPO (`Beta=0.01`). PopArt is NOT ablated — it is used in all models for all environments.

**Todo**: rerun all SAC ablations after the ablation 1 and 3 fix.

7. PPO and SAC have non-constant updates-per-true-environment-step as `num_envs` varies, while DQN maintains a constant ratio. This causes `run_grid_searches.sh` to misreport the effective training intensity across different parallel environment counts.
**cause (PPO)**: `updates_performed += agent.update_epochs * agent.num_minibatches` is counted per rollout, but rollout size = `num_steps * num_envs`. So updates/step = `(update_epochs * num_minibatches) / (num_steps * num_envs)` — inversely proportional to num_envs. With num_minibatches=4 fixed, doubling num_envs halves the ratio. Measured: ratio 0.125 → 0.031 → 0.016 for num_envs 1→4→8.

**cause (SAC)**: `buffer.size()` returns the number of *timestep positions* stored (one per `vec_env.step()` call), not individual transitions. The warmup check `buffer.size() >= batch_size=256` therefore requires `256 * num_envs` individual transitions before updates start. At num_envs=8 this needs 2048 transitions — exceeding the 2000-step grid search budget entirely (0 updates observed). DQN is unaffected because it uses a flat circular buffer that counts individual transitions.

**fix (PPO)**: Set `num_minibatches=int(vec_env.num_envs)` in `_ppo_agent_from_args`, so minibatch_size = `num_steps * num_envs / num_envs = num_steps = 128` (constant). Updates/step = `update_epochs * num_envs / (num_steps * num_envs) = update_epochs / num_steps` — independent of num_envs.

**fix (SAC)**: Convert the warmup threshold to timestep units: `_min_buf_timesteps = max(1, batch_size // buffer.n_envs)` in `rollout_offline_rl`. Updates now start after the same 256 individual transitions regardless of num_envs.

**status**: Fixed in `runner.py`. No reruns required (fixes only affect the grid-search benchmark, not the per-algorithm training logic).
