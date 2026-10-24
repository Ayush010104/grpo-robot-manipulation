# RL4VLA — "What Can RL Bring to VLA Generalization? An Empirical Study"

**arXiv 2505.19789** (Liu, Gao, Wei, et al.; submitted May 2025, v4 Jan 2026). Project
page: `rlvla.github.io`. *Note: "RL4VLA" is the roadmap's own shorthand for this paper —
the paper itself doesn't appear to use that name; the empirical-study framing is
confirmed from the abstract.* Grounded via web search, not a full read.

**Setup.** Empirical/comparative study (not a single new method) asking whether RL
fine-tuning improves VLA generalization over SFT, and which RL algorithm works best.
Argues SFT is brittle under distribution shift (compounding errors), motivating RL that
optimizes directly for task success instead of imitating a fixed dataset.

**Reward.** Task-success based (consistent with the roadmap's framing of this paper
comparing algorithms under a shared reward setup, not proposing a new reward).

**Group construction.** N/A directly — this is a comparison across PPO, GRPO, and DPO,
not a single group-construction scheme. The **PPO vs GRPO vs DPO comparison** is the
paper's central contribution, with a companion "simple recipe for efficient PPO training
on VLAs."

**Results / failure modes reported — this is the key paper motivating this whole
project's research question:**
- **PPO beats GRPO and DPO** as the RL algorithm for VLA fine-tuning.
- RL (PPO especially) improves generalization in semantic understanding and execution
  robustness over SFT, while visual-robustness generalization stays roughly flat.
- (Per roadmap Section 3 table, not independently re-confirmed here:) non-stationary
  dynamics destabilise GRPO specifically; sparse rewards and distribution shift hurt DPO;
  a shared actor-critic backbone saves ~45% VRAM for PPO.

**Relevance to this project.** This is the paper the whole roadmap's open problem is
built on — "GRPO loses to PPO in robot/VLA RL fine-tuning" is exactly the gap SM-GRPO is
trying to close while keeping GRPO's critic-free memory advantage. **Read this one first
and in full** before writing Month 2's results section — the diagnostics this repo logs
(group return std, degenerate-group fraction, KL-to-reference) should be checked against
whatever RL4VLA itself used to diagnose GRPO's instability, so Month 2's "reproduce the
failure" step reproduces the *same* failure mode, not a different one that happens to
also look unstable.
