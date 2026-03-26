# Agent API

This repository uses a shared `Agent` interface (see `agent.py`) so PPO/PG and DQN-style models can be swapped by runners with minimal glue code.

## Required methods

Future agent implementations should subclass `Agent` and implement:

- `to(device)`
  - Move all model modules and stateful normalizers to the target device.
  - Return `self`.

- `sample_action(...)`
  - Produce action(s) from observation(s).
  - Must support the calling pattern used by the unified runner (`runner.py`).

- `update(...)`
  - Run one optimization/update step.
  - Return a scalar training value (usually loss).
  - Populate `self.last_losses` with numeric metrics when possible.

## Optional hooks

The base class provides optional no-op hooks that should be overridden when relevant:

- `update_target()` for delayed/Polyak target-network updates.
- `update_running_stats(...)` for observation/reward normalizers.
- `attach_tensorboard(writer, prefix="agent")` for in-agent TensorBoard logging.

## Runtime behavior requirements

To stay compatible with current experiments, agents should preserve:

- PopArt-capable value heads when enabled.
- IQN/distributional critic support when enabled.
- Comparable public runtime API (`to`, `sample_action`, `update`, optional `update_target`, optional `update_running_stats`).

## Five ablation pillars

Agents used by the ablation runners should expose toggles/behavior for the same 5 pillars:

1. **Mirror Descent / Munchausen**
2. **Magnet Policy Regularization** (entropy regularization)
3. **Optimism** (intrinsic reward via `Beta` / RND path)
4. **Distributional Value Estimates** (IQN vs expected-value critics)
5. **Delayed Targets** (target-network update path)

The exact implementation details can differ by algorithm (e.g., PPO vs DQN), but these controls should be representable through constructor/runtime configuration and reflected in `update()` behavior.
