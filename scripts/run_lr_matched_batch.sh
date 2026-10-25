#!/usr/bin/env bash
# Corrected re-run of the LR question with the batch size actually MATCHED to the
# Month-2 run. docs/progress.md flags that scripts/run_lr_sweep.sh and
# run_month2_lowlr.sh accidentally used num_groups_per_iteration=3 (shell variable
# indirection with ${GROUPS:-3}) while the real Month-2 run used 20 -- root cause not
# fully pinned down, not a shell-profile leak (checked ~/.zshrc, ~/.zprofile), but
# confirmed from each process's own printed Hydra config, not just the overrides.yaml
# file (which collided on disk since all 4 Month-2 processes launched the same second).
#
# This script hardcodes every value directly (no shell-variable defaulting) so the same
# mistake can't recur silently, and runs GRPO at lr=3e-5 with num_groups_per_iteration=20
# -- an apples-to-apples comparison against the existing Month-2 grpo run (lr=3e-4,
# groups=20, in logs/month2_grpo.log).
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
mkdir -p logs

echo "resolved config: task=pick-place-v3 iters=60 groups=20 group_size=8 max_steps=150 lr=3e-5"

python train.py algo=grpo \
  env.env_id=pick-place-v3 env.backend=metaworld env.max_episode_steps=150 \
  policy.init_checkpoint=checkpoints/bc_pick-place-v3.pt \
  reference_policy.checkpoint=checkpoints/bc_pick-place-v3.pt \
  training.num_iterations=60 training.num_groups_per_iteration=20 \
  training.group_size=8 training.max_steps_per_episode=150 training.lr=3e-5 \
  training.checkpoint_every=1000 device=cpu \
  > logs/grpo_lr3e-5_matched.log 2>&1

echo "matched-batch lr=3e-5 run finished"
