#!/usr/bin/env bash
# Isolates WHY SM-GRPO delayed the collapse ~3x vs GRPO in the scale-matched comparison
# (docs/progress.md, 2026-10-26): is it specifically the k-NN STATE-MATCHING, or just
# having any per-step (vs. whole-trajectory) advantage signal? TGRPO already fuses a
# per-step term (same-timestep group mean, no state-matching) with the trajectory-level
# term -- if TGRPO shows a similar delay under the same scale-matched conditions, the
# benefit isn't specific to state-matching. RLOO (still trajectory-level, like GRPO, but
# leave-one-out instead of group-mean baseline) is the complementary check: if RLOO looks
# like GRPO, trajectory-level-vs-trajectory-level baseline choice doesn't matter much
# either, further isolating per-step-ness as the relevant variable.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
mkdir -p logs

echo "resolved config: task=pick-place-v3 iters=60 groups=20 group_size=8 max_steps=150 lr=3e-4 match_advantage_scale=true"

for algo in rloo tgrpo; do
  echo "launching $algo -> logs/month3_${algo}_scalematched.log"
  python train.py algo=$algo \
    env.env_id=pick-place-v3 env.backend=metaworld env.max_episode_steps=150 \
    policy.init_checkpoint=checkpoints/bc_pick-place-v3.pt \
    reference_policy.checkpoint=checkpoints/bc_pick-place-v3.pt \
    training.num_iterations=60 training.num_groups_per_iteration=20 \
    training.group_size=8 training.max_steps_per_episode=150 training.lr=3e-4 \
    training.match_advantage_scale=true \
    training.checkpoint_every=1000 device=cpu \
    > logs/month3_${algo}_scalematched.log 2>&1 &
done
wait
echo "ablation-isolation runs (rloo, tgrpo) finished"
