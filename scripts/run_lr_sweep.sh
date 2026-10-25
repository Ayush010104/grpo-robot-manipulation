#!/usr/bin/env bash
# Follow-up to Month 2: is GRPO's collapse really the advantage-estimator instability
# (roadmap Section 4.2), or just a too-high learning rate reused unchanged from the
# Hopper-v5 smoke-test default? Runs GRPO at two much lower LRs, same everything else.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
mkdir -p logs

TASK=pick-place-v3
CKPT=checkpoints/bc_${TASK}.pt
ITERS=${ITERS:-25}

for lr in 3e-5 3e-6; do
  tag="grpo_lr${lr}"
  echo "launching $tag -> logs/${tag}.log"
  python train.py algo=grpo \
    env.env_id=$TASK env.backend=metaworld env.max_episode_steps=150 \
    policy.init_checkpoint=$CKPT reference_policy.checkpoint=$CKPT \
    training.num_iterations=$ITERS training.num_groups_per_iteration=3 \
    training.group_size=8 training.max_steps_per_episode=150 training.lr=$lr \
    training.checkpoint_every=1000 device=cpu \
    > logs/${tag}.log 2>&1 &
done
wait
echo "lr sweep finished"
