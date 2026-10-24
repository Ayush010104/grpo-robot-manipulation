# RIPT-VLA — "Interactive Post-Training for Vision-Language-Action Models"

**arXiv 2505.17016** (Tan, Dou, Zhao, Krähenbühl; 2025). *The roadmap calls this
"RIPT-VLA (Tan et al.)" without a number; this arXiv id was found by matching the
RLOO/QueST/OpenVLA-OFT description — confirm it's the right paper before citing.*
Grounded via web search, not a full read.

**Setup.** Lightweight post-training stage on top of an already-SFT'd policy, tested on
two base models of very different scale: QueST (~20M params, tokenized/discrete action
head) and OpenVLA-OFT (7B, continuous L1-regression action head). Interactive —
collects on-policy rollouts in the environment during post-training.

**Reward.** Sparse binary success reward only (no dense shaping, no learned reward
model) — directly relevant to this project's Idea C (progress-signal tie-breaker), since
RIPT-VLA apparently gets strong results *without* densifying the reward at all.

**Group construction / advantage.** **RLOO (leave-one-out)**: critic-free, compares
multiple attempts at the same task against each other rather than against a trained value
network. This is exactly `algos/rloo.py` / `advantages.leave_one_out_advantages` in this
repo — RIPT-VLA is effectively an existing empirical validation that RLOO's
self-exclusion baseline (vs. GRPO's group mean, which includes each rollout's own return)
is enough to get strong results, which is a useful baseline-strength expectation for
Month 2.

**Results reported.**
- QueST: 82.7% → 93.6% average success across the LIBERO suites with RIPT.
- OpenVLA-OFT (7B): up to 97.5% success.
- Few-shot claim (one version of the abstract): with a single demonstration, an
  unworkable 4%-success SFT model reaches 97% success within 15 RIPT iterations — a
  striking sample-efficiency number worth re-verifying directly from the PDF, since this
  would be a strong existence proof that critic-free group-relative RL can work with very
  little data, which bears directly on how many BC demos Month 1 actually needs.

**Relevance to this project.** The sparse-reward-only, RLOO-baseline result is the
closest thing in this reading list to a validated RLOO Month-2 baseline — if this
project's own RLOO run (Month 2) doesn't come close to reproducing RIPT-VLA's stability,
that's a signal something in the repro (group size, reward scale, KL penalty) differs
from their setup and should be checked against the actual paper before concluding
anything about GRPO/SM-GRPO.
