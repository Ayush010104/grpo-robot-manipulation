#!/usr/bin/env bash
# Follow-up to the LR sweep (docs/progress.md): does GRPO/RLOO/TGRPO actually trend
# upward over the full 60 iterations at a safe LR (3e-5), the way PPO did at the
# original LR in the Month 2 run -- or do they stay flat even given enough steps?
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
mkdir -p logs

TASK=pick-place-v3
CKPT=checkpoints/bc_${TASK}.pt
ITERS=${ITERS:-60}
LR=${LR:-3e-5}

for algo in grpo rloo tgrpo; do
  tag="${algo}_lowlr_full"
  echo "launching $tag -> logs/${tag}.log"
  python train.py algo=$algo \
    env.env_id=$TASK env.backend=metaworld env.max_episode_steps=150 \
    policy.init_checkpoint=$CKPT reference_policy.checkpoint=$CKPT \
    training.num_iterations=$ITERS training.num_groups_per_iteration=3 \
    training.group_size=8 training.max_steps_per_episode=150 training.lr=$LR \
    training.checkpoint_every=1000 device=cpu \
    > logs/${tag}.log 2>&1 &
done
wait
echo "low-lr full runs finished"
