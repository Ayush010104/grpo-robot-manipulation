#!/usr/bin/env bash
# GRPO+kl_coef=0.1 from the new 56%-success BC checkpoint (2026-10-29/30) peaked at
# 31.2% success (iteration 36, degenerate_frac=0.00) -- a genuine new high for critic-free
# GRPO on this task -- but then drifted and re-collapsed to 0% by iteration 150, grad_norm
# climbing the whole way down (same signature as every other collapse in this project).
#
# Hypothesis: kl_coef=0.1 was calibrated against the OLD, weak (10%-success) reference,
# where "stay close to reference" was nearly synonymous with "stay bad but stable". Against
# a genuinely good (56%-success) reference, the same absolute coefficient is a much weaker
# *relative* constraint -- there's actually somewhere better to drift to, and apparently
# also somewhere worse, and 0.1 isn't enough to hold the policy near the good region once
# it has started exploring past it.
#
# Three arms, same new BC checkpoint, 150 iterations: kl_coef = 0.2, 0.3, 0.5. If a
# stronger constraint keeps the policy near its peak instead of drifting away and
# collapsing, that confirms the hypothesis and gives an actionable config for pushing
# toward the 70% target with GRPO specifically (not just PPO).
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
export PYTHONUNBUFFERED=1
mkdir -p logs

CKPT=checkpoints/bc_ablation/ep150_noise03.pt
COMMON="env.env_id=pick-place-v3 env.backend=metaworld env.max_episode_steps=150 \
  policy.init_checkpoint=$CKPT reference_policy.checkpoint=$CKPT \
  training.num_iterations=150 training.num_groups_per_iteration=20 \
  training.group_size=8 training.max_steps_per_episode=150 training.lr=3e-4 \
  training.match_advantage_scale=true training.checkpoint_every=5 device=cpu"

for KL in 0.2 0.3 0.5; do
  TAG=$(echo "$KL" | tr -d '.')
  echo "launching grpo kl_coef=$KL from new BC -> logs/month3_grpo_newbc_kl${TAG}.log"
  python train.py algo=grpo $COMMON training.kl_coef=$KL \
    > logs/month3_grpo_newbc_kl${TAG}.log 2>&1 &
done

wait
echo "new-BC kl_coef retune finished"
