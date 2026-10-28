# Broader literature review — 2026-10-28

Done via web search (standard mode, multiple queries), prompted by a user request to
survey the current "LLM + RL for robotic manipulation" field and locate scope for
innovation. Supersedes/extends the narrower 2026-10-24 novelty check below for Idea A
specifically. As with the 2026-10-24 check: second-hand via search snippets, verify
primary sources before citing in anything formal.

## The field, in one line

Current VLA (Vision-Language-Action) models are large transformers pretrained via
imitation learning, now getting the *exact* post-training RL recipe used for LLM
reasoning (GRPO, PPO) applied directly to the robot policy's own weights. That's the
actual "LLM + RL" connection — not hierarchical LLM-planner architectures (see the
2026-10-28 conversation's read of arXiv:2603.30022, a different, lower-rigor sub-area
that doesn't touch GRPO/collapse/KL at all).

## VLA+RL methods found

SimpleVLA-RL (GRPO, arXiv:2509.09674, "pushcut" phenomenon — RL discovers action
patterns outside the demo distribution), RLinf-VLA (arXiv:2510.06710, unified PPO+GRPO
framework, **their own experiments found PPO more stable than GRPO** — external
corroboration of our finding), TGRPO (arXiv:2506.08440), RL4VLA, RIPT-VLA, πRL, VLA-RL,
VLA-RFT. Survey: arXiv:2508.13073. Curated list: `Denghaoyuan123/Awesome-RL-VLA`.

## GRPO-for-continuous-control theory/method papers

- **Khanda et al., arXiv:2507.19555** — theoretical only, no experiments. State-aware
  advantage via **DBSCAN clustering** (different mechanism from SM-GRPO's k-NN
  per-step/time-window matching), trajectory-based policy clustering, regularized
  updates. Confirms: still the closest prior work to Idea A, still theory-only, still
  doesn't cover the degenerate-group/grad-norm mechanism we found empirically.
- **GR2PO, arXiv:2609.19850** — critic-free, normalizes group *returns*, clipped
  surrogate + entropy regularization + **KL-based early stopping** (trust-region style,
  different form from our continuous KL penalty term). Worth a direct comparison:
  early-stopping vs. continuous penalty, same benchmark.
- **Prism-GRPO, arXiv:2608.17423** — fixes wasted rollouts from same-outcome groups via
  a weighted execution-quality score on top of binary rewards; validated on real robots
  (RoboTwin), up to 56% fewer rollouts to target.
- **de Oliveira et al., "Learning Without Critics?", arXiv:2511.03527 (NeurIPS 2025)**
  — first systematic GRPO-vs-PPO study on classical continuous control (HalfCheetah,
  Humanoid, MountainCarContinuous). Critic-free trails PPO except short-horizon
  (corroborates our core finding in a non-VLA setting). **Smaller groups outperformed
  larger ones** — attributed to *batch-based* grouping mixing unrelated episodes. This
  directly contradicts our own group-size finding (G=64/128 delay collapse), but our
  groups are *true shared-state resets* (`envs/wrappers.py::make_group_envs`), not
  batch-mixed — open question worth resolving explicitly, see progress.md queue.

## The collapse mechanism, in the literature

- **"Phantom Advantages", arXiv:2609.13866** — std-normalization inflates tiny
  within-group spread (among all-fail rollouts under shaped rewards) into full-magnitude
  advantages. Same core mechanism as our grad-norm finding, LLM-RLVR context, not robot
  control, and framed around a filter-metric/shaped-reward mismatch rather than general
  group-std collapse.
- **REINFORCE++** replaces per-group std-normalization with global batch normalization
  for the same reason — this is what `grpo.normalize_mode="mean_only"` (added
  2026-10-28, see `advantages/group_baselines.py::group_mean_centered_advantages`) now
  lets us test directly, head-to-head with our KL fix.
- **"Why GRPO Needs Normalization", arXiv:2601.23135** — theoretical framing of
  std-normalization as an implicit, curvature-dependent adaptive step size. Theoretical
  grounding for *why* pre-clip grad_norm climbs as group std shrinks (our empirical
  finding, 2026-10-27b).

## KL-regularization-to-reference, in the literature

- **Rudner et al., NeurIPS 2021** — KL-regularized RL anchored to a frozen *parametric*
  BC/expert reference can itself cause pathological (slow, unstable, suboptimal)
  training dynamics; remedied by a **non-parametric** reference policy instead. Our
  `kl_coef=1.0` run got *worse*, not just slower (2026-10-27d) — consistent with this
  being the Rudner pathology, not simple over-regularization. Untested idea worth
  trying: a non-parametric reference built from `StateReturnBuffer` instead of the
  frozen BC network.
- LLM RLVR convention is often β=0 (no KL) since rule-based verifiers remove
  distribution-shift concerns — reasoning that doesn't obviously transfer to robotics,
  where the BC anchor is about staying in a feasible/safe motion manifold.

## Memory/replay mechanisms, in the literature

RePO and rollout-level-replay (arXiv:2506.09340, arXiv:2606.04560) reuse stored
trajectories for **additional gradient updates** (sample efficiency) — a USPTO filing
covers similar "Replay-GRPO" ground. **None of these use a buffer purely to improve the
non-parametric baseline's statistical quality without ever reusing it for extra policy
updates** — which is exactly what MM-GRPO does
(`algos/sm_grpo_memory.py`/`advantages/replay_buffer.py`). This looks like a genuinely
distinct, currently-unclaimed mechanism, though MM-GRPO's own results so far (3.6% vs.
plain GRPO+KL's 5.3% final success) don't yet demonstrate it winning — needs the
higher-kl_coef follow-up in the progress.md queue before claiming the positioning pays
off empirically, not just mechanistically.

## Where scope for innovation lies (summary, see conversation for full reasoning)

1. Mechanistic evidence (grad-norm trajectories) for the collapse in continuous robot
   control specifically — established in LLM-RL, not yet shown this way in robotics.
2. A controlled head-to-head of the ~5 scattered fixes on one benchmark — **in
   progress**, `scripts/run_month3_fixes_comparison.sh`, launched 2026-10-28.
3. KL-to-BC-reference as a standalone fix for critic-free continuous-control collapse,
   cleanly ablated — **done**, 2026-10-27d, this is probably our strongest result.
4. Memory-as-baseline-quality (not memory-as-replay) — **implemented** (MM-GRPO), not
   yet shown to clearly win; needs more tuning.
5. The group-size contradiction (true shared-state reset vs. batch-mixed grouping) —
   **open**, flagged for the progress.md queue.
6. Long-horizon, sparse-terminal-reward generalization — **open**, nothing tested here
   is sparse-reward yet (Meta-World pick-place-v3 is dense-shaped).

---

# Novelty check for Idea A (SM-GRPO) — roadmap Section 11, step 1

Done via web search on 2026-10-24 (arXiv full-text search tool unavailable; results below
are second-hand via search snippets and need primary-source verification before citing
or finalizing the research framing).

## Closest prior work found

- **arXiv 2507.19555 (Khanda et al., July 2025)** — "a theoretical framework extending
  GRPO to continuous control environments... introduces trajectory-based policy
  clustering, **state-aware advantage estimation**, and regularized policy updates
  designed for robotic applications." Per the search summary: **theory only, no
  experiments** ("establishing a foundation for future empirical validation on
  locomotion and manipulation tasks"). This is the closest match to Idea A's core
  mechanism and **must be read in full before finalizing SM-GRPO as the main
  contribution** — if its "state-aware advantage estimation" is the same k-NN /
  state-matched-baseline idea, SM-GRPO's novelty claim should shift to: (a) the specific
  k-NN-across-sibling-rollouts mechanism, (b) empirical validation (which 2507.19555
  apparently lacks), or (c) a different idea from Section 5 (B/C/D/E) should move up.
  **Action: fetch and read arXiv:2507.19555 in full next session.**
- **Prism-GRPO (arXiv 2608.17423)** — addresses the same degenerate-group problem as
  Idea B/C (all-success/all-fail groups wasting rollout budget) via a weighted
  execution-quality score added to the binary reward, plus dynamic sampling that drops
  zero-advantage groups. Closer to Ideas B/C than Idea A, but worth reading since it's
  attacking the same Section 4.2 #3 failure mode.
- **KRPO (~May 2025)** — reportedly replaces the static group mean with a Kalman-filter
  baseline to address "static group mean is vulnerable to noise" — a different fix to
  the same Section 4.2 #1/#2 problem (noisy group comparisons), not state-matching.
- **PAPO-VLA, RL Bootstrapping of OpenVLA-OFT** — relevant stability context, not
  competing with Idea A's mechanism directly.

## Not found

No paper surfaced that specifically does **k-nearest-neighbour, per-step baselines in a
state-embedding space computed across a GRPO group's sibling rollouts within a time
window** (Idea A's exact mechanism as specified in the roadmap). The Khanda et al. paper
is the one risk to that specific novelty claim and needs to be read directly.

## GitHub reading-list cross-check (`Denghaoyuan123/Awesome-RL-VLA`)

Fetched the repo's README directly (`gh api` confirmed the repo exists; title: "A Survey
on Reinforcement Learning of Vision-Language-Action Models for Robotic Manipulation").
**No entry describes state-matched baselines, k-NN advantage estimation, or
per-step/state-conditioned GRPO baselines.** Closest categories: GRPO-based online
methods (TGRPO, SimpleVLA-RL, RLVLA, WMPO, and others listed only as "uses GRPO", no
baseline-design detail visible from the README table alone) and advantage-weighted
offline methods (ARFM, RECAP, AW-BC, AWR, IG-AWR — these are offline/weighted-regression
methods, not GRPO-group k-NN baselines). This is a second independent signal (after the
web search) that Idea A's specific mechanism isn't an already-cataloged technique in this
survey -- still doesn't replace actually reading arXiv:2507.19555.

## Recommendation

Proceed with the Month 1-2 plan (BC + PPO/GRPO/RLOO/TGRPO baselines) regardless — that
work is needed either way. Before locking in SM-GRPO as *the* paper's contribution (start
of Month 3), read arXiv:2507.19555 in full and re-run this search against
`Denghaoyuan123/Awesome-RL-VLA` on GitHub (not checked yet — needs a GitHub fetch, not
just web search).
