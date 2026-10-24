# TGRPO (Trajectory-wise Group Relative Policy Optimization)

**arXiv 2506.08440** (Chen, Niu, Kong, Wang, Xing, Fan — Jilin University). Code:
`github.com/hahans/TGRPO`. *Grounded via web search of the abstract (v1/v3), not a full
read.*

**Setup.** Online RL fine-tuning for VLA models, motivated by VLA training relying mostly
on static-demonstration SFT, which generalizes poorly to new environments; RL is proposed
as the fix but runs into sparse rewards and unstable optimization on long-horizon tasks.

**Reward.** Uses an LLM to automatically construct **dense** reward functions from task
analysis — i.e., addresses sparse-reward instability (roadmap Section 4.2 #3) by
densifying the reward itself rather than by changing the advantage estimator. This is a
different mitigation strategy than this project's Idea C (progress-signal tie-breaker
only on all-fail groups); worth comparing the two approaches directly.

**Group construction / advantage.** Samples and normalizes multiple trajectories in
parallel (GRPO-style group), then **fuses trajectory-level and step-level advantage
estimates** — this is exactly the mechanism `algos/tgrpo.py` in this repo approximates
(our version uses same-timestep group-mean return-to-go for the step-level term, since
TGRPO's own step-level construction isn't confirmed from the abstract alone — re-check
the method section and correct `algos/tgrpo.py` if it differs). No value network used.

**Results reported.** Version-dependent (flag before citing a number):
- v3 abstract: 80.7% average success rate across four LIBERO task categories, 4.2 points
  above SFT.
- v1 text: evaluated on ten manipulation tasks from the LIBERO-Object benchmark instead.

**Relevance to this project.** TGRPO's fusion of trajectory- and step-level advantages is
the direct ablation baseline for SM-GRPO's claim that *state-matched* step-level
baselines (not just same-timestep ones) further reduce variance — the comparison in
Month 3 (SM-GRPO vs. TGRPO) is the most direct test of Idea A's actual contribution over
existing step-level-fusion work.
