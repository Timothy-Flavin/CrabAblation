# Hide-And-Seek Engine (High-Throughput Multi-Agent POMDP Simulator)

![SAR Simulation Replay](replay.gif)

**A high-performance, compute-efficient C++/PyBind11 engine for large-scale Multi-Agent Reinforcement Learning (MARL) in partially observable grid-worlds.**

---

## Key Features

- **Ultra-fast C++ backend**: Data-Oriented Design (DOD), cache-aligned memory, and OpenMP parallelism for 4,000,000+ FPS on consumer CPUs.
- **Zero-copy PyTorch bridge**: Direct Memory Access (DMA) via pinned memory—no serialization, no Python GIL bottleneck.
- **Flexible POMDP/MARL support**: Decentralized, centralized, and omniscient observability modes; hybrid continuous/discrete action spaces.
- **Highly configurable**: Data-driven world generation from PNG/JSON; supports arbitrary agent and tile types.
- **PettingZoo parallel API**: Drop-in compatibility for multi-agent RL research.
- **Rich rendering**: Global and agent POV visualizations, radio event tracing, and replay recording.

---

## Installation

```bash
pip install -e .
```
Optional rendering/input dependencies:
```bash
pip install pygame pillow pettingzoo imageio
```

---

## Engine Architecture & Performance

- **C++ core**: Aggressively cache-aligned, bit-packed memory slabs for all environment state; 256-byte stride padding eliminates false sharing and maximizes parallel throughput.
- **OpenMP parallelism**: Seamless scaling to 128+ environments per batch; NUMA-aware memory allocation for high-core CPUs.
- **Zero-copy PyTorch bridge**: Pinned memory tensors are mutated natively in C++ and instantly available to PyTorch/GPUs—no Python-side copying or serialization.
- **Empirical throughput**: 4M+ FPS (desktop), 200K+ FPS (laptop), 12M+ FPS (server); 20–50x faster than unpacked Python tensor engines.

---

## Level File Formats

### `test_level/tiles.json`
Each tile (list or name→object map):
- `rgb`: `[r, g, b]` (for PNG mapping)
- `altitude`: float
- `supports_walking`, `supports_flying`, `supports_aquatic`: bool
- `blocking`: bool

### `test_level/agents.json`
Each agent (list or name→object map):
- `flying`, `aqueous`, `walking`: bool
- `altitude_min`, `altitude_max`, `base_speed`, `base_view`, `battery`, `deployment_delay`
- `rgb`
- `terrain_speed` (dict by tile name)
- `start`: `[y, x]` (map coords or normalized)

### `test_level/survivors.json`
Each survivor (list or name→object map):
- `allowed_savers`: list of agent names
- `moves`: bool
- `rgb` (optional)
- `start` (optional)

### `test_level/level.png`
PNG map; each pixel matched to nearest tile `rgb` in `tiles.json`.

---

## Core Environment Usage

```python
from hide_and_seek_engine.env_wrapper import SARBatchedGridEnv

env = SARBatchedGridEnv(
    num_envs=8,
    map_png="test_level/level.png",
    tiles_json="test_level/tiles.json",
    agents_json="test_level/agents.json",
    survivors_json="test_level/survivors.json",
    map_size=32,
    seed=42,
)

obs, info = env.reset()
actions = env.action_space.sample()
obs, rewards, terminated, truncated, info = env.step(actions)
state = env.state()  # global CTDE state
```

### Observation Space (Local Actor Input)
`obs` is a dict:
- `obs["spatial"]`: `[Env, Agent, C_local, H, W]` (terrain, altitude, survivors, obs mask, agent layers)
- `obs["internal"]`: `[Env, Agent, 6]` (`deploy_remaining`, `stuck`, `view_range`, `battery`, `y`, `x`)

### State Space (Central Critic Input)
`env.state()` returns:
- `state["spatial"]`: `[Env, C_global, H, W]`
- `state["internal"]`: flattened agent+survivor vectors

### Action Space (Hybrid)
Per-agent action:
- movement: 2D vector in `[-1, 1]`
- radio: discrete channel `0..3` (`0`=no transmit, `1-3`=transmit)
Tensor shape: `[num_envs, n_agents, 3]` (`dy`, `dx`, `radio_channel`)

---

## Rendering

- Global: `env.render(env_idx=0)` (undiscovered tiles dimmed, saved survivors white)
- Agent POV: `env.render_pov(agent_idx=0, env_idx=0)` (shows last-known positions, radio knowledge)
- Print radio events: `env.radio_render()`

---

## PettingZoo Parallel API

```python
from hide_and_seek_engine.env_wrapper import SARParallelPettingZooEnv

pz_env = SARParallelPettingZooEnv(
    map_png="test_level/level.png",
    tiles_json="test_level/tiles.json",
    agents_json="test_level/agents.json",
    survivors_json="test_level/survivors.json",
)

obs, infos = pz_env.reset()
actions = {
    agent: {"move": [0.0, 1.0], "radio": 1}
    for agent in pz_env.agents
}
obs, rewards, terminations, truncations, infos = pz_env.step(actions)
```

---

## Test & Benchmark Suite

Run unit checks, 10k-step stress tests, FPS measurements, renderer smoke test:
```bash
python env_spec.py --steps 10000 --envs 1 2 4 8
```
Skip renderer test:
```bash
python env_spec.py --steps 10000 --envs 1 2 4 8 --skip-render
```

---

## Human Data Recorder

Collect SARSA tuples from a human-controlled agent each episode:
```bash
python human_runner.py
```
Record a visual replay:
```bash
python human_runner.py --record
```
Controls: `W`, `A`, `S`, `D` (move), `1`, `2`, `3` (radio)
After each episode, enter a save name. Data is written to `saved_human_behavior/<name>/`:
- `.npy` files for all buffers
- `replay.gif` (if `--record` used)

---

## Research & Technical Highlights

- **POMDP formalism**: Supports decentralized, centralized, and omniscient observability for MARL research.
- **Cache-aligned C++ arena**: 64/256-byte alignment, bit-packed state, NUMA-aware, false-sharing free.
- **Zero-copy PyTorch bridge**: Pinned memory, direct mutation, instant GPU transfer.
- **Mixed CNN/logical encoder**: Example architectures for spatial+internal agent features.
- **Empirical results**: 4M+ FPS (desktop), 12M+ FPS (server), 20–50x speedup over Python engines.

For full technical details, see `main.tex`.
