# Investigation Report: SAC Failures on Minigrid Environments

## 1. Do SAC and Other Algorithms Share the Same CNN/Encoder?
**Yes.** 
SAC receives all of the same observation enhancements and encoders that DQN and PPO do. 
- **Environment Processing:** In `environment_utils.py`, `minigrid` bypasses large frame stacking configurations but uses `FastObsWrapper` consistently across all algorithms.
- **Architectural Integration:** In `runner.py`, the `MixedObservationEncoder` is bundled into an `encoder_factory` and securely passed into `_sac_agent_from_args`. Upon reviewing `learning_algorithms/SAC_Rainbow.py`, the `DistSAC` and `EVSAC` construct both their `Actor` and `Critic`/`ObsActEncoder` utilizing this provided `encoder_factory`. Thus, SAC accurately "sees" the structured Minigrid environment identically to DQN/PPO.

## 2. Is the Action Space the Root Problem?
**Yes. The methodology for bridging SAC (a continuous algorithm) to Minigrid (a discrete action space) is fundamentally flawed for unordered categorical choices.**

### How the Conversion Works in the Codebase:
1. `runner.py` forces any standard discrete environment to look continuous to SAC by using the helper `_proxy_action_space` logic (`Box(low=0.0, high=1.0)`).
2. The continuous SAC actor network outputs actions drawn from a squashed Gaussian distribution bounded between `[-1.0, 1.0]`, which is re-scaled to `[0.0, 1.0]`.
3. Before executing the step, `environment_utils.py` applies the `_continuous_to_discrete_classic` transformation. This slices the continuous `[0.0, 1.0]` scalar evenly into `n` integer bins (where Minigrid has 7 choices).

### Why this breaks SAC's learning:
- **Distributional Drag across Unrelated Actions:** SAC uses continuous Gaussians where neighboring values imply similar actions (e.g. steering a wheel 10 degrees vs 11 degrees). In Minigrid's `Discrete(7)` space, the actions are categorically unordered (e.g., $0 \rightarrow$ Turn Left, $1 \rightarrow$ Turn Right, $2 \rightarrow$ Move Forward, $3 \rightarrow$ Pick Up).  
- **Catastrophic Interpolation:** If the SAC agent believes both "Turn Left" (0.1) and "Pick Up" (0.5) are good choices in a similar state, the mean of the Gaussian might center around `0.3`, unintentionally executing "Move Forward." Continuous gradients must physically "sweep" across intermediate categorical actions to update their mean constraints.
- **Unlike PPO/DQN:** PPO (via categorical branches) and DQN (via individual Q-heads) maintain separated, independent logits for each action. They bypass this sliding interpolation problem completely. 

## 3. Potential Solutions
Since SAC works extremely well on continuous control challenges (e.g., MuJoCo) where actions map correctly onto ordinal floats, no core `SAC_Rainbow.py` modifications are strictly necessary unless you want Minigrid functionality. 

If Discrete Environment compatibility via SAC is desired in the future:
- **Implement a pure "Discrete SAC":** Unlike Continuous SAC's Reparameterization Trick, Discrete SAC replaces the continuous squashed Gaussian distributions with a true Categorical distribution. The Actor spits out $N$ distinct probabilities (akin to PPO's discrete policy outputs) and utilizes expected discrete soft Q-values.
- **Gumbel-Softmax Relaxation:** Implement a distinct Actor head when `isinstance(action_space, gym.spaces.Discrete)` is true. This can emulate gradients through categorical bottlenecks.