#!/usr/bin/env bash
# Bottleneck calculation: the roadmap's own Month-1 target for BC was "30 to 60% partial
# success" (bc_train.py's docstring), deliberately degraded with noisy-expert demos so RL
# has room to improve on it. Our actual measured deterministic BC success is 10% -- well
# below even the low end of that intended range (2026-10-28 conversation, eval.py over 50
# held-out episodes). scripts/collect_bc_data.py defaults to only 20 episodes and
# noise_std=0.3 (substantial corruption on a [-1,1] action space). This ablation isolates
# which factor (too few demos, too much noise, or both) explains the 10% vs 30-60% gap --
# a direct, controlled experiment, not a guess.
#
# 4 variants, each: collect demos -> train BC (50 epochs, default) -> deterministic eval
# (50 held-out episodes, same protocol as the existing BC/GRPO comparison):
#   1. ep20_noise03  = the CURRENT setup (20 episodes, noise_std=0.3) -- re-collected
#      under a fresh seed for a clean side-by-side comparison, not reusing the original
#      data/bc_demos/pick-place-v3.npz (kept untouched -- still used as the reference
#      checkpoint for every running GRPO/KL experiment).
#   2. ep150_noise03 -- 7.5x more demos, same noise (isolates demo QUANTITY)
#   3. ep20_noise01  -- same demo count, much less noise (isolates demo QUALITY/noise)
#   4. ep150_noise01 -- both improvements together (best case)
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
export PYTHONUNBUFFERED=1
mkdir -p logs data/bc_ablation checkpoints/bc_ablation

run_variant () {
  local name=$1 episodes=$2 noise=$3
  echo "=== $name: episodes=$episodes noise_std=$noise ===" | tee -a logs/bc_bottleneck_ablation.log
  python scripts/collect_bc_data.py --tasks pick-place-v3 --episodes "$episodes" \
    --noise-std "$noise" --seed 777 --out-dir "data/bc_ablation/$name" \
    >> logs/bc_bottleneck_ablation.log 2>&1
  python bc_train.py --data "data/bc_ablation/$name/pick-place-v3.npz" \
    --out "checkpoints/bc_ablation/${name}.pt" \
    >> logs/bc_bottleneck_ablation.log 2>&1
  python eval.py --checkpoint "checkpoints/bc_ablation/${name}.pt" --env-id pick-place-v3 \
    --backend metaworld --episodes 50 | tee -a logs/bc_bottleneck_ablation.log
}

run_variant ep20_noise03 20 0.3
run_variant ep150_noise03 150 0.3
run_variant ep20_noise01 20 0.1
run_variant ep150_noise01 150 0.1

echo "=== bottleneck ablation finished -- see logs/bc_bottleneck_ablation.log for the full table ==="
