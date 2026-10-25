#!/usr/bin/env bash
# Re-run of Month 3's first SM-GRPO test with sm_grpo.robust_normalize=true. The v1 run
# (run_month3_sm_grpo.sh) collapsed faster than vanilla GRPO because
# state_matched_baselines returns an UNNORMALIZED advantage (raw return-to-go
# difference) -- it started at advantage_abs_mean=12.6 vs GRPO's properly-scaled ~0.74
# at the same LR, a ~17x larger effective step size. robust_normalize divides by
# robust_scale(group.returns), putting it on the same order of magnitude as GRPO's.
# A 10-iteration check confirmed this prevents the immediate collapse -- this runs the
# full 60 iterations to get a real comparison against GRPO/PPO.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
mkdir -p logs

echo "resolved config: task=pick-place-v3 iters=60 groups=20 group_size=8 max_steps=150 lr=3e-4 k=4 time_window=5 robust_normalize=true"

python train.py algo=sm_grpo \
  env.env_id=pick-place-v3 env.backend=metaworld env.max_episode_steps=150 \
  policy.init_checkpoint=checkpoints/bc_pick-place-v3.pt \
  reference_policy.checkpoint=checkpoints/bc_pick-place-v3.pt \
  training.num_iterations=60 training.num_groups_per_iteration=20 \
  training.group_size=8 training.max_steps_per_episode=150 training.lr=3e-4 \
  sm_grpo.k=4 sm_grpo.time_window=5 sm_grpo.robust_normalize=true sm_grpo.sigma_min=0.01 \
  training.checkpoint_every=1000 device=cpu \
  > logs/month3_sm_grpo_v2_robust.log 2>&1

echo "Month-3 SM-GRPO v2 (robust_normalize=true) run finished"
