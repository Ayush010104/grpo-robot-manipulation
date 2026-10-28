#!/usr/bin/env bash
# Follow-up to the 2026-10-27c KL-penalty result: GRPO at kl_coef=0.1 was the first
# critic-free run in the whole project to sustain nonzero success through the end of a
# 120-iteration run (final-10 mean 5.3%, grad_norm bounded <1.4 the whole tail) instead
# of permanently collapsing. Three follow-ups, each testing a different axis:
#
#   1. GRPO kl=0.1, G=8, 300 iterations (2.5x the horizon that already worked) -- does
#      performance keep climbing now that gradients are finally PPO-like bounded, or
#      does it plateau around the same 3-8% band it was oscillating in by iteration 120?
#   2. GRPO kl=0.1, G=64, 120 iterations -- group-size scaling alone peaked at 18% success
#      before collapsing (docs/progress.md, 2026-10-27). Does the KL penalty let that
#      higher peak actually sustain instead of decaying?
#   3. SM-GRPO kl=0.1, G=8, 120 iterations -- newly wired (algos/sm_grpo.py). Does
#      state-matching compound with the KL fix, i.e. combine the two best mechanisms
#      found so far (per-step signal that survives degenerate groups + bounded drift)?
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
export PYTHONUNBUFFERED=1
mkdir -p logs

CKPT256=checkpoints/bc_pick-place-v3.pt

echo "launching grpo kl=0.1 G=8 x300iters -> logs/month3_grpo_kl01_long.log"
python train.py algo=grpo \
  env.env_id=pick-place-v3 env.backend=metaworld env.max_episode_steps=150 \
  policy.init_checkpoint=$CKPT256 reference_policy.checkpoint=$CKPT256 \
  training.num_iterations=300 training.num_groups_per_iteration=20 \
  training.group_size=8 training.max_steps_per_episode=150 training.lr=3e-4 \
  training.match_advantage_scale=true training.kl_coef=0.1 training.checkpoint_every=20 \
  device=cpu \
  > logs/month3_grpo_kl01_long.log 2>&1 &

echo "launching grpo kl=0.1 G=64 -> logs/month3_grpo_kl01_g64.log"
python train.py algo=grpo \
  env.env_id=pick-place-v3 env.backend=metaworld env.max_episode_steps=150 \
  policy.init_checkpoint=$CKPT256 reference_policy.checkpoint=$CKPT256 \
  training.num_iterations=120 training.num_groups_per_iteration=3 \
  training.group_size=64 training.max_steps_per_episode=150 training.lr=3e-4 \
  training.match_advantage_scale=true training.kl_coef=0.1 training.checkpoint_every=10 \
  device=cpu \
  > logs/month3_grpo_kl01_g64.log 2>&1 &

echo "launching sm_grpo kl=0.1 -> logs/month3_sm_grpo_kl01.log"
python train.py algo=sm_grpo \
  env.env_id=pick-place-v3 env.backend=metaworld env.max_episode_steps=150 \
  policy.init_checkpoint=$CKPT256 reference_policy.checkpoint=$CKPT256 \
  training.num_iterations=120 training.num_groups_per_iteration=20 \
  training.group_size=8 training.max_steps_per_episode=150 training.lr=3e-4 \
  training.match_advantage_scale=true training.kl_coef=0.1 training.checkpoint_every=10 \
  device=cpu \
  > logs/month3_sm_grpo_kl01.log 2>&1 &

wait
echo "KL follow-up sweep finished"
