# SimpleVLA-RL

**arXiv 2509.09674** (Li et al.; submitted Sep 2025, ICLR 2026). Code:
`github.com/PRIME-RL/SimpleVLA-RL`. *Grounded via web search of the abstract/HF page,
not a full read — re-verify before citing exact numbers (the arXiv and ICLR versions
already disagree with each other, see below).*

**Setup.** Outcome-driven RL for VLA models, built on the `veRL` RL training stack, with
VLA-specific trajectory sampling, parallel rollout collection, multi-environment
rendering, and optimized loss computation. Base model: OpenVLA-OFT (also supports plain
OpenVLA; RoboTwin 2.0 support added Oct 2025).

**Reward.** Outcome/success reward (binary task completion), in the spirit of
outcome-reward RL for LLM reasoning — the paper's framing question is explicitly "can
outcome rewards do for VLA long-horizon planning what they did for LLM reasoning."

**Group construction.** GRPO-style (per roadmap Section 3 table) — not independently
confirmed from the abstract text pulled here; re-check the method section.

**Results / failure modes reported.**
- arXiv v1 abstract: SOTA on LIBERO, beats pi0 on RoboTwin 1.0 and 2.0 *when
  exploration-enhancing strategies are used* (implying exploration collapse/stalling is
  a failure mode they had to engineer around — consistent with roadmap Section 4.2 #3,
  sparse-reward groups giving zero gradient).
- ICLR 2026 camera-ready abstract instead reports 99% of SOTA on LIBERO and an 80%
  relative improvement on RoboTwin — **numbers differ between versions**, use the version
  you actually cite.
- Describes a "pushcut" phenomenon: the RL-tuned policy discovers action patterns absent
  from its SFT prior — i.e., genuine new behavior from RL, not just SFT-prior
  interpolation.
- A later (Jan 2026, repo-only, not peer-reviewed) note claims ~300% improvement on a
  real-world long-horizon dexterous task.

**Relevance to this project.** Confirms GRPO/outcome-RL *can* work on OpenVLA-OFT given
enough engineering around exploration — useful as a "GRPO can succeed" counterpoint to
RL4VLA's "PPO beats GRPO" finding; worth reading in full to see what their
exploration-enhancing strategies were, since that may overlap with the degenerate-group
problem (Idea B/C) this project is also attacking.
