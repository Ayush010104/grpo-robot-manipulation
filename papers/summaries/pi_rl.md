# pi_RL — "Online RL Fine-tuning for Flow-based Vision-Language-Action Models"

**arXiv 2510.25889** (Chen et al., 14 co-authors; submitted Oct 2025, v3 Jan 2026). Open
source within the `RLinf` framework (`rlinf.readthedocs.io`). Grounded via web search,
not a full read.

**Setup.** Addresses a specific blocker for RL on flow-matching action heads (pi0,
pi0.5): standard policy-gradient RL (PPO/GRPO) needs an action log-likelihood, but
flow-matching generates actions via iterative denoising, so there's no tractable
log-likelihood for the executed action out of the box. Two fixes proposed:
- **Flow-Noise** — adds a learnable noise network inside the denoising process, reframed
  as a discrete-time MDP, giving an *exact* log-likelihood.
- **Flow-SDE** — converts the deterministic ODE denoising into an SDE with the same
  marginal distribution, also making a likelihood tractable.

Either lets PPO- or GRPO-style policy-gradient training run on top of a flow-matching
policy as normal.

**Reward.** Task-success based (per-benchmark; not a new reward scheme — the paper's
contribution is making the *policy gradient itself* well-defined for flow policies).

**Group construction.** Both PPO- and GRPO-style training are supported on top of
Flow-Noise/Flow-SDE — i.e., this paper is orthogonal to the GRPO-stability question this
project asks (it solves "how do you even compute a policy gradient for a flow VLA",
not "is GRPO's group-relative baseline stable"), but **is directly relevant to the
Section 10 stretch goal** of scaling SM-GRPO to a pi0-family model, since any such attempt
would need one of Flow-Noise/Flow-SDE (or an equivalent trick) to make the per-step
log-probabilities SM-GRPO's advantage × log-prob loss needs.

**Results reported (version-dependent, re-check before citing):**
- LIBERO: pi0 57.6% → 97.6%, pi0.5 77.1% → 98.3% (v3, few-shot SFT baseline).
- ManiSkill: v1 and v3 report different numbers (e.g. v1: pi0 41.6%→85.7%; v3: pi0
  38.4%→78.8%) — the two versions aren't directly comparable.
- MetaWorld MT50 (per RLinf docs, matches v1 abstract): pi0 85.8% with Flow-Noise; pi0.5
  70.7% with Flow-Noise / 78.1% with Flow-SDE (vs. 43.8% SFT).

**Relevance to this project.** Not a direct GRPO-stability result, but the MetaWorld MT50
numbers are a useful rough sanity check for what "good" looks like on the same benchmark
family this project is using for Month 1-3 (Meta-World+) — once SM-GRPO is working on the
small BC policy, these numbers are a reference point for whether the project's own
success rates are in a plausible range before attempting the Section 10 scale-up.
