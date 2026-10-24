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
