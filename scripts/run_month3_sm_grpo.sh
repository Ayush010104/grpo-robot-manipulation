#!/usr/bin/env bash
# Month 3 (roadmap Section 7): first real test of SM-GRPO (Idea A, the main
# contribution) against the now-properly-controlled Month-2 baselines -- same task,
# same batch size (groups=20), same BC warm-start, same lr=3e-4 as the original
# PPO/GRPO Month-2 runs for a direct comparison.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
mkdir -p logs

echo "resolved config: task=pick-place-v3 iters=60 groups=20 group_size=8 max_steps=150 lr=3e-4 k=4 time_window=5"

python train.py algo=sm_grpo \
  env.env_id=pick-place-v3 env.backend=metaworld env.max_episode_steps=150 \
  policy.init_checkpoint=checkpoints/bc_pick-place-v3.pt \
  reference_policy.checkpoint=checkpoints/bc_pick-place-v3.pt \
  training.num_iterations=60 training.num_groups_per_iteration=20 \
  training.group_size=8 training.max_steps_per_episode=150 training.lr=3e-4 \
  sm_grpo.k=4 sm_grpo.time_window=5 \
  training.checkpoint_every=1000 device=cpu \
  > logs/month3_sm_grpo_lr3e-4.log 2>&1

echo "Month-3 SM-GRPO (lr=3e-4) run finished"
