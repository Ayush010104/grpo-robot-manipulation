#!/usr/bin/env bash
# Month 2 (roadmap Section 7): "Implement baselines on top of the behaviour-cloning
# policy: PPO (with critic), vanilla GRPO (trajectory-level, std-normalised), RLOO and
# TGRPO-style fused advantages ... Confirm the instability: show GRPO's variance or
# collapse compared with PPO."
#
# Runs all four on pick-place-v3, warm-started from the Month-1 BC checkpoint, logging
# to outputs/<algo>/<timestamp>/ (TensorBoard) and logs/month2_<algo>.log (stdout).
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
mkdir -p logs

TASK=pick-place-v3
CKPT=checkpoints/bc_${TASK}.pt
ITERS=${ITERS:-60}
GROUPS=${GROUPS:-3}
GROUP_SIZE=${GROUP_SIZE:-8}
MAX_STEPS=${MAX_STEPS:-150}

for algo in ppo grpo rloo tgrpo; do
  echo "launching $algo -> logs/month2_${algo}.log"
  python train.py algo=$algo \
    env.env_id=$TASK env.backend=metaworld env.max_episode_steps=$MAX_STEPS \
    policy.init_checkpoint=$CKPT reference_policy.checkpoint=$CKPT \
    training.num_iterations=$ITERS training.num_groups_per_iteration=$GROUPS \
    training.group_size=$GROUP_SIZE training.max_steps_per_episode=$MAX_STEPS \
    training.checkpoint_every=1000 device=cpu \
    > logs/month2_${algo}.log 2>&1 &
done
wait
echo "all four runs finished"
