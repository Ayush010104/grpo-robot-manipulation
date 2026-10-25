# Progress log

## 2026-10-25 — CORRECTION: the LR-sweep conclusion below was confounded by batch size

**The "GRPO collapse is partly a step-size artifact" conclusion in the section below is
unreliable as stated — treat it as superseded until the matched-batch re-run (this
section) is read.** While generating a fresh comparison plot, the per-iteration env-step
counts didn't match between the Month-2 run and the LR-sweep follow-ups (24000 vs 3600 at
iteration 0 -- a clean 20/3 ratio). Checked each process's own printed Hydra config (the
authoritative source — not the on-disk `.hydra/overrides.yaml`, which collided and got
overwritten since all 4 Month-2 processes launched in the same wall-clock second) and
confirmed: **the real Month-2 run used `num_groups_per_iteration=20`**, not the `3` that
`scripts/run_month2_comparison.sh`'s `GROUPS=${GROUPS:-3}` default was supposed to
produce, while `run_lr_sweep.sh` / `run_month2_lowlr.sh` *did* get the intended `3`.
Root cause not fully pinned down — checked `~/.zshrc` and `~/.zprofile` for a leaked
`GROUPS` env var, found nothing, so it's unexplained, not just "obviously a profile
collision." Either way: **the LR-sweep's "no collapse but no clear trend either" result
was measured at a 6.67x smaller batch size than the Month-2 run it was supposed to be
compared against**, so it doesn't actually isolate the learning-rate variable -- smaller
batch size alone would also be expected to make training noisier and slower to show a
trend, independent of LR.

**Fix applied:** `scripts/run_lr_matched_batch.sh` hardcodes every value directly (no
shell-variable indirection) and launched GRPO at `lr=3e-5` with
`num_groups_per_iteration=20` (matching the real Month-2 batch size exactly) for the
full 60 iterations, to properly isolate the LR effect against the existing
`logs/month2_grpo.log` (`lr=3e-4`, `groups=20`) run. **Results pending** — check this
file for an update, or `logs/grpo_lr3e-5_matched.log` directly, before relying on
anything in the "LR sweep" section below.

## 2026-10-25 — LR sweep: the "collapse" is partly a step-size artifact (SUPERSEDED, see correction above)

Follow-up to the open question below. Ran GRPO at `lr=3e-5` and `lr=3e-6` (vs. the
original `lr=3e-4`), 25 iterations each, same everything else (`scripts/run_lr_sweep.sh`,
logs in `logs/grpo_lr*.log`):

| lr | mean success_rate | first-5-iter mean | last-5-iter mean | max | iters at 0% |
|---|---|---|---|---|---|
| 3e-4 (Month 2 run) | → 0 and stays there | 0.062 | **0.000** | 0.056 | 54/60 (iter 6 onward) |
| 3e-5 | 0.063 | 0.066 | 0.050 | 0.208 | 4/25 |
| 3e-6 | 0.062 | 0.067 | 0.050 | 0.167 | 5/25 |

**Verdict: the original lr=3e-4 result overstates GRPO's instability.** At 10x and 100x
lower LR, GRPO does *not* collapse to a permanent 0% floor — it stays noisy, bouncing
between 0% and up to 17-21% success rate, which is qualitatively different from the
irreversible collapse seen at the higher LR. So part of the Month-2 "collapse" was a
too-large step size amplifying whatever noisy/degenerate-group gradient GRPO produces
into a destructive update, not pure advantage-estimator instability independent of step
size.

**But this isn't a clean exoneration of GRPO either:** at the lower LRs, GRPO is merely
*not collapsing* — it shows no clear upward trend over these 25 iterations (first-5 mean
≈ last-5 mean ≈ 0.05-0.07 at both LRs), whereas Month 2's PPO run climbed steadily from
~5% to ~29-31% over a comparable number of steps. So the refined picture is: **GRPO is
highly LR-sensitive (collapses at a step size PPO tolerates fine) and, even at a
safe LR, doesn't show PPO's steady improvement in this short a window** — both still
support the roadmap's core research question (a stable, sample-efficient, critic-free
method is the open problem), just with "collapses outright" softened to "collapses at
naive LR, stagnates at safe LR."

**Caveat on this sweep itself:** only 25 iterations (375k env steps) per LR — too short
to rule out GRPO starting to climb later, the way PPO's own early iterations also looked
flat/noisy before trending up around iteration 20+ in the Month 2 run. **Next step:**
rerun GRPO at lr=3e-5 (or a small sweep around it) for the full 60 iterations before
treating "GRPO stagnates even at a safe LR" as settled, and do the same lr-matching for
RLOO/TGRPO rather than assuming they behave identically to GRPO just because they looked
identical at lr=3e-4.

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
