## Purpose

This report keeps track of a number of unexplained results in the ablation testing accross DQN SAC and PPO in environments cartpole mujoco and minigrid.

## List of Anomolies

1. Models are evaluated with greedy action selection every 50 episodes for offline rl and every 5th iteration for online rl, but the x axis is not scaled to match the train rewards in the episode x axis plots. DQN eval lines are either horizontal or non existant
**cause**: Error in `graph.py` where `x_eval` is generated using `np.linspace(0, max_steps, num=e_mean.size)` even for episode charts, effectively stretching or squeezing the eval lines incorrectly.
**next steps**: Modify `graph.py` to use `max_episodes` for the x-axis when plotting episodes instead of `max_steps`.

2. Dqn cartpole was run without truncation at 500 steps and magnet_reg `ablation == 2` ran for ~1,800 episodes while the rest ran for ~1,200. 
**cause**: Truncation limit not applied properly in the environment wrapper for DQN cartpole.

**Todo**: rerun all dqn cartpole ablations on timpc with proper truncation
**next steps**: Ensure TimeLimit wrapper or max_episode_steps is properly set in `runner.py` for CartPole and re-run.

3. Dqn minigrid `ablation == 4` ran for 12,000 episodes. All other ablations ran for 4,000. None of the ablations score above 4/10 on minigrid when Random Network Distilation with frame stacking should likely do better than that. 
**cause**: unknown
**next steps**: Investigate RND implementation and frame stacking logic in DQN.

4. ppo cartpole `ablation == 4` ran for far fewer episodes than the rest (1000 vs 1700) 1M steps should be more than enough for PPO to complete cartpole but 1000 episodes at ~200 step average is not 1M and PPO has not converged. By the end. Rerun likely needed.
**cause**: unknown
**next steps**: Re-run the PPO CartPole ablation 4 experiment to see if it converges.

5. ppo minigrid ablation 4 ran for 12k episodes while all other ablations ran for 2k. 12k was needed for it to get close to convergence
**cause**: unknown
**next steps**: Check hyperparameters or exploration schedule for PPO ablation 4 on Minigrid.

6. SAC results dont make sense for any of the environments. For cartpole the rewards look like a step function that jumps between each interval of 100. For minigrid SAC learns almost nothing (0.2/1.0). On mujoco all results besides ablation 2 and 5 look the exact same with no variance, while 5 fails to learn anything and 2's error bars span almost the entire y axis.
**cause**: unknown
**next steps**: Review SAC implementation details, specifically temperature tuning and replay buffer interactions for step function anomalies.