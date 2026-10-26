#!/usr/bin/env bash
# Re-launch after an overnight session restart killed the previous round
# (docs/progress.md, 2026-10-27). checkpoint_every lowered to 5 (was 1000, which never
# triggered on a 60-iteration run) so a future interruption doesn't lose everything --
# still no resume-from-checkpoint logic in train.py, so these still start over from the
# BC checkpoint, but at least a policy snapshot will exist partway through next time.
#
# Two jobs only (not re-running G=64 -- already has a clear, complete-enough signal:
# it delays then collapses by iter 26):
#   1. GRPO G=128: open question from the correction -- does it eventually collapse
#      too, and why did success decline (18%->5.5%) even while degenerate_frac stayed
#      at 0.00 the whole time it ran?
#   2. MM-GRPO: the novel extension, never actually run yet. Same settings as the
#      SM-GRPO run that produced the ~3x delay result, for a direct comparison.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
export PYTHONUNBUFFERED=1
mkdir -p logs

CKPT512=checkpoints/bc_pick-place-v3_512.pt
CKPT256=checkpoints/bc_pick-place-v3.pt

echo "launching grpo G=128 -> logs/month3_grpo_g128_net512_round2.log"
python train.py algo=grpo \
  env.env_id=pick-place-v3 env.backend=metaworld env.max_episode_steps=150 \
  'policy.hidden_sizes=[512,512]' \
  policy.init_checkpoint=$CKPT512 reference_policy.checkpoint=$CKPT512 \
  training.num_iterations=60 training.num_groups_per_iteration=2 \
  training.group_size=128 training.max_steps_per_episode=150 training.lr=3e-4 \
  training.match_advantage_scale=true training.checkpoint_every=5 device=cpu \
  > logs/month3_grpo_g128_net512_round2.log 2>&1 &

echo "launching sm_grpo_memory -> logs/month3_sm_grpo_memory.log"
python train.py algo=sm_grpo_memory \
  env.env_id=pick-place-v3 env.backend=metaworld env.max_episode_steps=150 \
  policy.init_checkpoint=$CKPT256 reference_policy.checkpoint=$CKPT256 \
  training.num_iterations=60 training.num_groups_per_iteration=20 \
  training.group_size=8 training.max_steps_per_episode=150 training.lr=3e-4 \
  training.match_advantage_scale=true training.checkpoint_every=5 device=cpu \
  > logs/month3_sm_grpo_memory.log 2>&1 &

wait
echo "round 2 (G=128 continuation + MM-GRPO first run) finished"
