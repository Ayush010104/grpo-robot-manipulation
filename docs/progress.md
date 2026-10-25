# Progress log

## 2026-10-25 — Month 2: reproduced the GRPO collapse

Ran `scripts/run_month2_comparison.sh`: PPO, GRPO, RLOO, TGRPO, all warm-started from the
`bc_pick-place-v3.pt` checkpoint, 60 iterations each (group_size=8, 3 groups/iter, 150
max steps/episode → 1.44M env steps per algo). Results (`analysis/plot_comparison.py`,
plots saved to `analysis/month2_success_rate.png` and `analysis/month2_mean_return.png`):

| algo | success_rate: last | success_rate: best | mean_return: last | mean_return: best |
|---|---|---|---|---|
| **ppo** | **28.8%** | **30.6%** | **234.5** | **243.9** |
| grpo | 0.0% | 5.6% | 0.88 | 38.1 |
| rloo | 0.0% | 5.6% | 0.88 | 38.1 |
| tgrpo | 0.0% | 5.6% | 0.86 | 38.1 |

**This is a clean reproduction of the roadmap's central hypothesis (Section 4.2 / the
RL4VLA finding "PPO beats GRPO").** All three critic-free, group-relative methods start
at the BC policy's success rate (~5-7%, matching the noisy training-rollout estimate of
the Month-1 BC checkpoint) and collapse to ~0% success / near-zero return within ~6-9
iterations (~150-240k env steps) and never recover over the full 1.44M-step run. PPO
instead improves steadily to ~29-31% success. See the success-rate plot — GRPO/RLOO/TGRPO
are visually indistinguishable (all three collapse the same way), while PPO's curve
climbs the whole time.

**Bug found and fixed during this analysis:** `is_degenerate_group` (used for the
`degenerate_group_frac` diagnostic) was being called on continuous trajectory *returns*
with a `> 0` threshold. Meta-World's reward is dense-shaped, so returns are almost always
positive regardless of real task success — every group logged `degenerate_group_frac =
1.00` for all four algos, which is an artifact of the threshold, not evidence every group
was truly all-success-or-all-fail. Added `is_degenerate_group_by_success` (operates on
the real `info["success"]` flag) and switched all four `algos/*.py` to use it, plus a
regression test (`test_is_degenerate_group_by_success_ignores_dense_return_magnitude`).
**This bug did not affect the collapse result above** — the advantage computations that
actually drive the policy update were always on the real returns, untouched by this
diagnostic; only the degenerate-group logging was wrong. Re-running Month 2 with the
fixed diagnostic (to see the *real* degenerate-group fraction over training, which is
likely still relevant via `info["success"]` directly) is a good next step but wasn't
re-run tonight to avoid spending another ~2 hours of wall-clock on a result whose
headline finding (collapse vs. no collapse) is already unambiguous.

**Open question for Month 3 / next session:** since GRPO/RLOO/TGRPO all collapse near-
identically and near-immediately, it's worth checking *why* before assuming it's the
"noisy group comparison" story from Section 4.2 #1 — e.g. log the actual gradient
direction / action-mean drift over the first ~10 iterations, since a collapse this fast
and this uniform across three different advantage estimators is also consistent with a
too-high learning rate (`training.lr = 3e-4` was reused unchanged from the Hopper-v5
smoke-test default, not tuned for a BC-warm-started policy) rather than purely the
advantage-estimator instability the roadmap describes. Worth a quick learning-rate sweep
on GRPO before concluding the collapse is really about advantage *estimator* quality
(which is what Month 3's SM-GRPO is supposed to fix) rather than just step size.

## 2026-10-24 — initial scaffold + Month-1 start (autonomous session)

Executed roadmap Section 11 "Immediate Next Steps" 1-3, and most of Section 9's code
structure, while Ayush was away.

**Done:**
1. **Novelty check (step 1).** Web search only (no arXiv API / GitHub fetch available
   this session) — see [`novelty_check.md`](novelty_check.md). Key finding: arXiv
   **2507.19555** (Khanda et al., Jul 2025) proposes "state-aware advantage estimation"
   for GRPO in continuous control, theory-only, no experiments. Closest prior work to
   Idea A — **read it before finalizing SM-GRPO as the main contribution.** Also check
   `Denghaoyuan123/Awesome-RL-VLA` on GitHub directly (not done, needs repo fetch).
2. **Environment setup (step 2 / roadmap Section 6).**
   - `.venv` (Python 3.12, arm64) with torch 2.14, gymnasium 1.4, metaworld, minari,
     wandb, hydra-core, rliable, etc. — see [`requirements.txt`](../requirements.txt).
   - MPS backend confirmed available (`torch.backends.mps.is_available() == True`).
   - Verified with `scripts/check_envs.py`: random policy runs clean on Gymnasium MuJoCo
     (Hopper-v5, Walker2d-v5, Ant-v5) and Meta-World+ (pick-place-v3, door-open-v3,
     drawer-open-v3, button-press-v3); Meta-World's scripted expert policy
     (`SawyerPickPlaceV3Policy`) also verified (succeeded within 200 steps).
   - **LIBERO not installed** — heavier install (robosuite + deps), left for next
     session; `envs/wrappers.py::_build_libero` is stubbed and raises `NotImplementedError`
     until wired up.
3. **Code structure (roadmap Section 9)** — all in place and smoke-tested:
   - `envs/wrappers.py` — unified `reset(seed)`/success/progress interface over
     Gymnasium and Meta-World, plus `make_group_envs` for group-reset (shared initial
     state across a GRPO group).
   - `policies/mlp_policy.py` — tanh-squashed Gaussian MLP policy + value net.
   - `advantages/group_baselines.py` — vanilla GRPO advantage, robust normalisation
     (Idea B), leave-one-out (RLOO), **state-matched k-NN baselines (Idea A / SM-GRPO)**,
     trajectory/step fusion (TGRPO-style). **11 unit tests, all passing**
     (`tests/test_advantages.py`).
   - `algos/` — `ppo.py` (GAE + clipped surrogate + critic), `grpo.py`, `rloo.py`,
     `tgrpo.py`, `sm_grpo.py`, sharing `rollout_buffer.py` and `common.py`. **All 5
     algorithms smoke-tested end-to-end on Hopper-v5** via `train.py` (a few iterations
     each, no crashes).
   - `train.py` — Hydra-config-driven (`configs/config.yaml`), TensorBoard logging,
     checkpointing.
   - `eval.py` — success-rate / mean-return evaluation from a checkpoint.
   - `bc_train.py` + `scripts/collect_bc_data.py` — BC pipeline using Meta-World's
     scripted experts (+ action noise) as demonstrations, trained via exact NLL under
     the policy's own tanh-Gaussian.
4. **Month 1, BC policies (step 3)** — ran the full pipeline on 3 Meta-World tasks
   (pick-place-v3, door-open-v3, drawer-open-v3; 20 noisy-expert episodes each,
   noise_std=0.35, 60 BC epochs). Held-out eval (30 episodes, disjoint seeds):

   | Task | Success rate | Mean return |
   |---|---|---|
   | pick-place-v3 | 16.7% | 81.9 |
   | door-open-v3 | 66.7% | 748.6 |
   | drawer-open-v3 | 83.3% | 495.3 |

   Roadmap target is ~30-60% partial success (leaving room for RL to help). Only
   door-open is close; pick-place is too low and drawer-open too high. **Next-session
   tuning knob:** `--noise-std` / `--episodes` in `scripts/collect_bc_data.py` and
   `--epochs` in `bc_train.py` — more noise or fewer epochs should pull door-open/
   drawer-open down; pick-place-v3 may need more demo episodes (it's the hardest task,
   3-stage grasp-lift-place) rather than less noise.

**Not done / explicitly deferred (per roadmap, flagged rather than skipped silently):**
- One-page summaries of SimpleVLA-RL, TGRPO, RL4VLA, RIPT-VLA, pi_RL (Month 1 reading;
  `papers/summaries/` dir created but empty).
- LIBERO install/integration (Section 6).
- Month 2: haven't yet run PPO vs. GRPO vs. RLOO vs. TGRPO *on top of* a BC policy with
  the full diagnostic suite (group return std, degenerate-group fraction, KL-to-reference,
  etc. over a real multi-iteration run) to reproduce the documented GRPO instability --
  only short smoke tests from random init so far.
- `analysis/` (rliable plots, diagnostic dashboards) — empty dir, not started.
- Reading-list tracker cross-check against `Denghaoyuan123/Awesome-RL-VLA` on GitHub.

**Repo:** git-initialized locally at the project root, not yet pushed anywhere (no
GitHub remote configured this session).
