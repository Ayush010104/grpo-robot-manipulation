#!/usr/bin/env bash
# The bottleneck ablation (2026-10-28/29, docs/progress.md) found demo QUANTITY, not
# noise, explains almost all of the gap: 20 episodes -> 0% deterministic BC success,
# 150 episodes -> 56% (noise_std 0.3 vs 0.1 barely mattered once there were enough
# demos). 56% is already near the roadmap's original 30-60% Month-1 target that the
# old 20-episode checkpoint undershot badly.
#
# Every GRPO/PPO run so far warm-started from the old, weak 10%-success checkpoint.
# This re-runs our best critic-free config (GRPO + kl_coef=0.1) AND the PPO ceiling-
# setter from the new 56%-success checkpoint instead, to see how much of the "70%"
# target gap closes just from fixing the warm-start -- the single highest-leverage
# change identified so far.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
export PYTHONUNBUFFERED=1
mkdir -p logs

CKPT=checkpoints/bc_ablation/ep150_noise03.pt

echo "launching grpo kl=0.1 from new BC checkpoint -> logs/month3_grpo_kl01_newbc.log"
python train.py algo=grpo \
  env.env_id=pick-place-v3 env.backend=metaworld env.max_episode_steps=150 \
  policy.init_checkpoint=$CKPT reference_policy.checkpoint=$CKPT \
  training.num_iterations=150 training.num_groups_per_iteration=20 \
  training.group_size=8 training.max_steps_per_episode=150 training.lr=3e-4 \
  training.match_advantage_scale=true training.kl_coef=0.1 training.checkpoint_every=10 \
  device=cpu \
  > logs/month3_grpo_kl01_newbc.log 2>&1 &

echo "launching ppo from new BC checkpoint -> logs/month3_ppo_newbc.log"
python train.py algo=ppo \
  env.env_id=pick-place-v3 env.backend=metaworld env.max_episode_steps=150 \
  policy.init_checkpoint=$CKPT reference_policy.checkpoint=$CKPT \
  training.num_iterations=150 training.num_groups_per_iteration=20 \
  training.group_size=8 training.max_steps_per_episode=150 training.lr=3e-4 \
  training.checkpoint_every=10 device=cpu \
  > logs/month3_ppo_newbc.log 2>&1 &

wait
echo "new-BC-warmstart comparison finished"
