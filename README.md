# Stable Group-Relative RL Fine-Tuning for Robot Policies

Research codebase for making GRPO-style critic-free RL fine-tuning stable for robot
control. See [`docs/roadmap.md`](docs/roadmap.md) for the full research roadmap (background,
prior work, candidate algorithms, three-month plan) and [`docs/progress.md`](docs/progress.md)
for a running log of what has actually been done.

**Core research question:** Can a critic-free, group-relative method (GRPO-style) match
or beat PPO's stability and sample efficiency in robot policy fine-tuning, while keeping
GRPO's memory and simplicity advantages?

**Main candidate contribution:** SM-GRPO (State-Matched Group Baselines) — per-step,
critic-free advantages computed from k-nearest-neighbour states across a group's rollouts,
instead of a single whole-trajectory group baseline.

## Repo layout

```
envs/        wrappers for Meta-World+, LIBERO, Gymnasium MuJoCo; unified reset(seed),
             success flags, optional progress signals, group-reset support
policies/    BC policy (MLP / small transformer), Gaussian or tanh-squashed action head
algos/       ppo.py, grpo.py, rloo.py, tgrpo.py, sm_grpo.py (shared rollout buffer)
advantages/  group mean/std, robust normalisation, leave-one-out, state-matched k-NN
             baseline, trajectory/step fusion
train.py     config-driven (Hydra), logging + checkpointing
eval.py      success rate, generalisation splits, video recording
analysis/    plotting, rliable statistics, diagnostic dashboards
configs/     Hydra YAML configs per algo/env
scripts/     setup and sanity-check scripts
papers/summaries/  one-page summaries of prior work (Month 1 reading list)
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# optional, heavier installs (see docs/roadmap.md Section 6):
#   LIBERO: clone from https://github.com/Lifelong-Robot-Learning/LIBERO and `pip install -e .`
#   lerobot: pip install lerobot  (for diffusion/ACT policies, later)

python scripts/check_envs.py   # verifies MuJoCo + Meta-World run on this machine
```

## Status

See [`docs/progress.md`](docs/progress.md) — updated as work lands.
