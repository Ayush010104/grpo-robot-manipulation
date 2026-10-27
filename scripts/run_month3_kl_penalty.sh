#!/usr/bin/env bash
# Test the actual fix for the missing safeguard found 2026-10-27c: kl_to_reference was
# logged by every algos/*.py since Month 2 but never added to the loss. Implemented a
# real differentiable KL(policy || reference) penalty (algos/common.py::kl_penalty_term,
# training.kl_coef), wired into grpo.py and sm_grpo_memory.py only so far.
#
# Four arms, matched LR=3e-4 (the only LR from the previous sweep that didn't reintroduce
# instability), G=8, 120 iterations:
#   1-3. Vanilla GRPO at kl_coef=0.01 / 0.1 / 1.0 -- the sharpest test: does a real KL
#        penalty alone (no memory buffer, no state-matching) stop the collapse that has
#        been reproduced in every unconstrained GRPO run so far? If yes, that is a much
#        simpler and more fundamental fix than MM-GRPO.
#   4. MM-GRPO at kl_coef=0.1 -- does constraining drift let its already-stable gradient
#      actually convert into sustained improvement past the ~9% peak / ~1.5% plateau seen
#      in logs/month3_mmgrpo_exploit_lr3e-4.log?
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
export PYTHONUNBUFFERED=1
mkdir -p logs

CKPT256=checkpoints/bc_pick-place-v3.pt

for KL in 0.01 0.1 1.0; do
  TAG=$(echo "$KL" | tr -d '.')
  echo "launching grpo kl_coef=$KL -> logs/month3_grpo_klpen_${TAG}.log"
  python train.py algo=grpo \
    env.env_id=pick-place-v3 env.backend=metaworld env.max_episode_steps=150 \
    policy.init_checkpoint=$CKPT256 reference_policy.checkpoint=$CKPT256 \
    training.num_iterations=120 training.num_groups_per_iteration=20 \
    training.group_size=8 training.max_steps_per_episode=150 training.lr=3e-4 \
    training.match_advantage_scale=true training.kl_coef=$KL training.checkpoint_every=10 \
    device=cpu \
    > logs/month3_grpo_klpen_${TAG}.log 2>&1 &
done

echo "launching sm_grpo_memory kl_coef=0.1 -> logs/month3_mmgrpo_klpen_01.log"
python train.py algo=sm_grpo_memory \
  env.env_id=pick-place-v3 env.backend=metaworld env.max_episode_steps=150 \
  policy.init_checkpoint=$CKPT256 reference_policy.checkpoint=$CKPT256 \
  training.num_iterations=120 training.num_groups_per_iteration=20 \
  training.group_size=8 training.max_steps_per_episode=150 training.lr=3e-4 \
  training.match_advantage_scale=true training.kl_coef=0.1 training.checkpoint_every=10 \
  device=cpu \
  > logs/month3_mmgrpo_klpen_01.log 2>&1 &

wait
echo "KL-penalty sweep finished"
