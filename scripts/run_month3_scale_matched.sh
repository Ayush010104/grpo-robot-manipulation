#!/usr/bin/env bash
# Properly scale-matched GRPO vs SM-GRPO comparison (docs/progress.md, 2026-10-25/26):
# both algorithms' final advantage batches are z-scored to zero-mean/unit-variance
# (training.match_advantage_scale=true) before the policy-gradient loss, closing the
# scale-mismatch confound found in the earlier v1/v2 SM-GRPO runs. Same batch size and
# LR as the rest of the Month-2/3 comparisons (groups=20, lr=3e-4, BC warm-start).
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
mkdir -p logs

echo "resolved config: task=pick-place-v3 iters=60 groups=20 group_size=8 max_steps=150 lr=3e-4 match_advantage_scale=true"

for algo in grpo sm_grpo; do
  echo "launching $algo -> logs/month3_${algo}_scalematched.log"
  python train.py algo=$algo \
    env.env_id=pick-place-v3 env.backend=metaworld env.max_episode_steps=150 \
    policy.init_checkpoint=checkpoints/bc_pick-place-v3.pt \
    reference_policy.checkpoint=checkpoints/bc_pick-place-v3.pt \
    training.num_iterations=60 training.num_groups_per_iteration=20 \
    training.group_size=8 training.max_steps_per_episode=150 training.lr=3e-4 \
    training.match_advantage_scale=true \
    sm_grpo.k=4 sm_grpo.time_window=5 \
    training.checkpoint_every=1000 device=cpu \
    > logs/month3_${algo}_scalematched.log 2>&1 &
done
wait
echo "scale-matched GRPO vs SM-GRPO comparison finished"
