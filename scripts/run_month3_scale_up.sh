#!/usr/bin/env bash
# User-requested follow-up: does a larger within-group sample size (G) fix or mitigate
# GRPO's instability, with a larger policy network? This directly tests the diagnosis in
# the chat writeup: small G=8 groups produce noisy/degenerate group comparisons
# (degenerate_group_frac climbed to ~0.90-0.95 in the G=8 runs); a much larger G gives a
# less noisy group mean/std estimate and makes degenerate (all-same-outcome) groups less
# likely. Network doubled to [512,512] (required retraining the BC checkpoint --
# checkpoints/bc_pick-place-v3_512.pt, 13.3% success over 30 eval episodes, comparable to
# the original 256x256 checkpoint's 16.7%).
#
# num_groups_per_iteration is reduced as group_size grows to keep wall-clock per
# iteration roughly comparable to the existing group_size=8/groups=20 runs (~160
# episodes/iter): group_size=64 -> groups=3 (192 episodes/iter), group_size=128 ->
# groups=2 (256 episodes/iter).
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
mkdir -p logs
export PYTHONUNBUFFERED=1  # otherwise piped stdout block-buffers and logs look empty
                            # mid-run (caused a false "died silently" diagnosis earlier)

CKPT=checkpoints/bc_pick-place-v3_512.pt

echo "resolved config: task=pick-place-v3 iters=60 hidden_sizes=[512,512] lr=3e-4 match_advantage_scale=true"

python train.py algo=grpo \
  env.env_id=pick-place-v3 env.backend=metaworld env.max_episode_steps=150 \
  'policy.hidden_sizes=[512,512]' \
  policy.init_checkpoint=$CKPT reference_policy.checkpoint=$CKPT \
  training.num_iterations=60 training.num_groups_per_iteration=3 \
  training.group_size=64 training.max_steps_per_episode=150 training.lr=3e-4 \
  training.match_advantage_scale=true training.checkpoint_every=1000 device=cpu \
  > logs/month3_grpo_g64_net512.log 2>&1 &

python train.py algo=grpo \
  env.env_id=pick-place-v3 env.backend=metaworld env.max_episode_steps=150 \
  'policy.hidden_sizes=[512,512]' \
  policy.init_checkpoint=$CKPT reference_policy.checkpoint=$CKPT \
  training.num_iterations=60 training.num_groups_per_iteration=2 \
  training.group_size=128 training.max_steps_per_episode=150 training.lr=3e-4 \
  training.match_advantage_scale=true training.checkpoint_every=1000 device=cpu \
  > logs/month3_grpo_g128_net512.log 2>&1 &

wait
echo "group-size scale-up runs (G=64, G=128) finished"
