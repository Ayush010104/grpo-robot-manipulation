#!/usr/bin/env bash
# Head-to-head comparison of collapse-fixes from the literature (docs/novelty_check.md,
# 2026-10-28): the field has at least 5 different fixes to the same degenerate-group /
# std-normalization collapse problem, scattered across different papers, never compared
# on one benchmark at matched budget. All arms: G=8, lr=3e-4, match_advantage_scale=true,
# 120 iterations, same BC warm-start -- directly comparable to the existing baselines:
#   - no fix at all: logs/month3_grpo_scalematched.log (60 iters, permanently dead by 14)
#   - KL penalty only (kl=0.1): logs/month3_grpo_klpen_01.log (120 iters, 5.3% final)
#
# New arms this script launches:
#   1. robust_normalize=true (Idea B / sigma_min floor on the group std -- the
#      project's own original "robust scale" fix, never tested alone without KL)
#   2. normalize_mode=mean_only (REINFORCE++-style: no per-group std division at all,
#      rely solely on the global batch normalization -- advantages/group_baselines.py::
#      group_mean_centered_advantages, new 2026-10-28)
#   3. robust_normalize=true + kl_coef=0.1 (best-of-both test)
#   4. normalize_mode=mean_only + kl_coef=0.1 (best-of-both test)
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
export PYTHONUNBUFFERED=1
mkdir -p logs

CKPT256=checkpoints/bc_pick-place-v3.pt
COMMON="env.env_id=pick-place-v3 env.backend=metaworld env.max_episode_steps=150 \
  policy.init_checkpoint=$CKPT256 reference_policy.checkpoint=$CKPT256 \
  training.num_iterations=120 training.num_groups_per_iteration=20 \
  training.group_size=8 training.max_steps_per_episode=150 training.lr=3e-4 \
  training.match_advantage_scale=true training.checkpoint_every=10 device=cpu"

echo "launching grpo robust_normalize=true -> logs/month3_grpo_robustnorm.log"
python train.py algo=grpo $COMMON grpo.robust_normalize=true \
  > logs/month3_grpo_robustnorm.log 2>&1 &

echo "launching grpo normalize_mode=mean_only -> logs/month3_grpo_meanonly.log"
python train.py algo=grpo $COMMON grpo.normalize_mode=mean_only \
  > logs/month3_grpo_meanonly.log 2>&1 &

echo "launching grpo robust_normalize=true + kl=0.1 -> logs/month3_grpo_robustnorm_kl01.log"
python train.py algo=grpo $COMMON grpo.robust_normalize=true training.kl_coef=0.1 \
  > logs/month3_grpo_robustnorm_kl01.log 2>&1 &

echo "launching grpo mean_only + kl=0.1 -> logs/month3_grpo_meanonly_kl01.log"
python train.py algo=grpo $COMMON grpo.normalize_mode=mean_only training.kl_coef=0.1 \
  > logs/month3_grpo_meanonly_kl01.log 2>&1 &

wait
echo "fixes-comparison sweep finished"
