# Performance Investigation Report: Rainbow DQN in MuJoCo HalfCheetah

## Executive Summary
The observed performance collapse in the MuJoCo HalfCheetah environment for "Ablation 0" (all features active) and the performance gains from ablating Soft DQN and Distributional RL (IQN) point to two primary issues:
1.  **A scale mismatch in the IQN entropy bonus** that creates a divergent feedback loop with the alpha autotuner.
2.  **Over-aggressive Munchausen RL penalties** that occur when a diverging `alpha` interacts with PopArt's `sigma`.

---

## Hypothesis 1: IQN Entropy Bonus Scale Mismatch
In `IQNRainbowDQN.update`, the entropy bonus added to the target values is calculated using `.mean(dim=-1)` across action bins, whereas in `EVRainbowDQN.update` it is summed.

### Technical Detail
*   **IQN Implementation:** `ent_bonus = -(pi_next * logpi_next).mean(dim=-1)`. This makes the bonus $1/N_{bins}$ of the actual entropy.
*   **Autotuner:** `current_entropy = -(pi_fresh * logpi_fresh).sum(dim=-1).mean()`. The autotuner uses the full entropy.
*   **Consequence:** The Q-targets do not "reward" entropy at the same scale the autotuner expects. This forces the autotuner to drive `alpha` to much higher values than necessary to achieve the target entropy. Because Munchausen and Entropy terms are scaled by `alpha * current_sigma`, these terms explode in magnitude when `alpha` diverges.

### Proposed Verification Test
**Unit Test (Target Sanity):**
Create a test case with a fixed `alpha` and a uniform Q-distribution. Calculate the expected target value manually. Compare the output of `EVRainbowDQN.update` and `IQNRainbowDQN.update`.
*   **Pass Condition:** Both agents should produce identical `next_head_vals` (within floating point error) when given the same inputs.

---

## Hypothesis 2: Munchausen Over-Dominance due to Diverging Alpha
While scaling the Munchausen RL penalty by PopArt's `sigma` is **mathematically correct for scale-invariance** (ensuring the penalty remains proportional to the return scale), it becomes a failure point when `alpha` is not properly bounded or stabilized.

### Technical Detail
*   **Scaling Mechanism:** `penalty = current_sigma * alpha * munchausen_constant * logpi`. This correctly ensures that in an environment with 1000x larger rewards, the Munchausen term remains relevant.
*   **The "Drowning" Effect:** When the IQN entropy bug drives `alpha` to extreme values, the Munchausen penalty (and the entropy bonus) becomes significantly larger than the raw environment rewards ($b\_r\_ext$). 
*   **Observation:** In Ablation 2 (Soft DQN off), `alpha` is fixed at a small value (0.001), preventing this explosion. This explains why Ablation 2 performs significantly better—it allows the Munchausen term to provide its scale-invariant benefits without being amplified to an level that ignores the environment signal.

### Proposed Verification Test
**Integration Test (Reward Scale Monitoring):**
Instrument the `update` method to log the ratio of `|Munchausen_term|` to `|b_r_ext|`.
*   **Pass Condition:** The Munchausen term should be of the same order of magnitude as the environment reward. If it exceeds it by 10x+, the agent is likely unstable.

---

## Hypothesis 3: Discretization and Alpha Interaction
MuJoCo is a continuous environment. Discretizing it into many bins (e.g., 11 or more) creates a large action space.

### Technical Detail
*   Soft DQN's `target_entropy` is set to `0.8 * log(bins)`. In a multi-dimensional action space ($D=6$ for HalfCheetah), if the agent averages entropy across dimensions but the autotuner targets the sum (or vice versa), the temperature will be wrong.
*   The current code averages entropy across dimensions: `next_head_vals = next_head_vals.mean(-1)`. This is correct for MultiDiscrete only if the reward is also treated as an average across dimensions, which depends on the environment wrapper.

### Proposed Verification Test
**Unit Test (Multi-Dim Entropy):**
Initialize an agent with `n_action_dims > 1`. Pass a batch where only one dimension has high entropy. Check if the autotuner and the target calculation react consistently.

---

## Summary of Findings & Next Steps

| Observed Symptom | Likely Cause |
| :--- | :--- |
| Ablating DistRL improves performance | `IQNRainbowDQN` entropy bug (mean vs sum). |
| Ablating Soft DQN improves performance | Stops `alpha` from diverging due to the IQN bug. |
| Lowering `l_clip` worsens performance | Munchausen penalty becomes too large; `l_clip` acts as a safety. |
| Ablating Delayed Target causes collapse | Standard DQN instability (overestimation bias). |

### Recommended Action Plan (No code changes yet)
1.  **Fix IQN Entropy:** Change `.mean(dim=-1)` to `.sum(dim=-1)` in `IQNRainbowDQN` entropy bonus.
2.  **Validate Scale Invariance:** Maintain the PopArt `sigma` scaling for Munchausen, but monitor `alpha` stability.
3.  **Validate Alpha autotuning:** Add logging for `alpha` and `current_entropy` to confirm they stabilize.
