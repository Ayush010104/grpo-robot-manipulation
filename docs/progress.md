# Progress log

## 2026-10-27b — Root cause of the collapse, and the first genuinely new result: MM-GRPO doesn't die

Both relaunched jobs (G=128 continuation, MM-GRPO's first real run) completed the full
60 iterations this time (`checkpoint_every=5` worked -- no restart hit mid-run anyway).
Analysis script: `analysis/resilience_analysis.py`. Plot: `analysis/gradnorm_mechanism.png`.

**G=128's open question is answered: it also collapses.** `degenerate_group_frac` hits
1.00 at iteration 37 and success_rate goes to 0.000 and stays there through iteration 59
(`logs/month3_grpo_g128_net512_round2.log`). So larger G does not prevent collapse, full
stop -- it only delays the onset (G=8 ~iter 6-14, G=64 ~iter 19-26, G=128 ~iter 36-37).
That settles the group-size question.

**Root cause, with actual mechanistic evidence (not just a guess).** `clip_grad_norm_`
in `algos/common.py::clip_grad_and_step` returns the *pre-clip* gradient norm before
rescaling the step. Across every group-std-normalized method (GRPO, SM-GRPO, and GRPO at
every tested G), once a critical fraction of groups goes degenerate, this pre-clip
grad_norm does not shrink toward zero (as "no learning signal" would predict) -- it
*climbs*, in GRPO G=128's case up to 76.5 (max_norm=1.0, so the raw gradient is ~76x the
clip threshold) and keeps climbing for the rest of the run even as success_rate sits at
flat 0.000. Mechanism: `robust_scale`'s `sigma_min=0.01` floor exists precisely because a
degenerate group's return std is near-zero but *not exactly* zero (dense reward still
varies slightly between rollouts that share the same binary success outcome). Dividing a
small-but-nonzero numerator by a clipped-to-tiny denominator produces a handful of
extreme advantage values. `clip_grad_norm_` bounds the resulting *step size* but not the
*direction* -- after clipping, the update is still unit-norm in whatever direction those
few outlier samples point, so the policy keeps moving, just along a direction dictated by
noise rather than the averaged batch signal. This is a positive feedback loop: the bad
update makes the policy worse -> more groups go degenerate next iteration -> std shrinks
further -> grad_norm climbs further. It is why collapse is *irreversible* for every
critic-free method tested: once `success_rate` hits 0 and stays there, it never recovers
for the remainder of any run (GRPO G=8/64/128, SM-GRPO all show zero "bounce-backs" after
their last nonzero-success iteration, per `resilience_analysis.py`). PPO is immune
because its critic-relative advantage doesn't depend on in-batch outcome variance at all,
and its grad_norm stays bounded (1-4) for the entire 60 iterations regardless of
success_rate.

**MM-GRPO (`sm_grpo_memory`, the novel cross-iteration-memory extension) is the first
method tested that does not show this failure mode.** Same task, same G=8, same LR
(3e-4), same scale-matching as the SM-GRPO comparison run. Results
(`logs/month3_sm_grpo_memory.log`, full 60/60 iterations):
- Peak success_rate 0.094 @ iteration 24 -- notably *higher* than GRPO's or SM-GRPO's
  peak (0.056, both at iteration 0, i.e. before any gradient update -- meaning GRPO and
  SM-GRPO's training loop never beat the BC warm-start checkpoint even once, while
  MM-GRPO's did).
- Last iteration with nonzero success_rate: **iteration 59 -- the last iteration of the
  run.** It never goes permanently dead. 5 separate "bounce-backs" (nonzero success
  immediately following a fully-degenerate iteration) vs. 0 for every pure-GRPO config
  and 6 for SM-GRPO but only within an early window before it does permanently die at
  iteration 40.
- Pre-clip grad_norm stays in **0.08-0.39 for the entire run** -- never exceeds the
  clip threshold, let alone explodes. This is *more* stable than PPO (1-4), despite
  being critic-free.

Mechanistic explanation for why: MM-GRPO's k-NN baseline (`state_matched_baselines` with
`extra_states`/`extra_returns` from `StateReturnBuffer`) draws candidates from up to
50k cross-iteration states, not just the 8 sibling rollouts in the current group. Two
effects: (1) the baseline is a much better-conditioned estimate of "expected
return-to-go from this state" -- smaller candidate pool = noisier baseline = more
outlier-prone raw advantages, and the memory buffer directly fixes that; (2) because the
buffer persists across iterations, it still contains states from *before* the current
policy started struggling, so even as current-iteration rollouts degrade, the baseline
doesn't degrade in lockstep -- there's no equivalent of "the whole group agrees, so std
collapses to near-zero," because the comparison set was never just the current group's 8
rollouts. This directly targets the root cause identified above, and the grad_norm data
confirms it worked exactly as hypothesized.

**Honest caveat -- this is a stability result, not yet a performance win.** MM-GRPO's
final-5-iteration mean success_rate is 0.0098 (~1%), far below PPO's final-5 mean
(~27%). It fixed the catastrophic-collapse failure mode but has not yet matched PPO's
absolute performance within 60 iterations. The natural hypothesis: MM-GRPO's gradients
are ~10x smaller in magnitude than PPO's for the whole run, so at a matched LR it is
simply taking much smaller effective steps -- it may just be under-training, not
under-powered. Testable and worth doing before claiming more. See "next" below.

**Next experiment launched** (`scripts/run_month3_mmgrpo_exploit.sh`,
`checkpoint_every=5`): MM-GRPO at `lr=1e-3` and `lr=3e-3` (3x and 10x the matched LR),
120 iterations instead of 60, to test whether the gradient-stability headroom can be
spent on faster learning without reintroducing the collapse. If either beats PPO's ~27%
or even just keeps climbing past where GRPO/SM-GRPO always die, that is the paper's
clearest result so far: a critic-free method that is not just "less unstable than GRPO"
but quantitatively*and mechanistically* fixes the specific failure mode that makes
critic-free RL unreliable for continuous robot control.

## 2026-10-27 — CORRECTION: group size delays the collapse, doesn't fix it; full ablation data

The session restarted overnight (killed all 4 background jobs -- no checkpointing was
saved, `checkpoint_every=1000` on 60-iteration runs means none of these hit a
checkpoint; worth fixing, see below). Before reporting the final numbers, pulled
whatever each process had actually logged (stdout where `PYTHONUNBUFFERED=1` was set,
TensorBoard otherwise, which flushes independently of stdout buffering).

**Group-size scaling -- the 2026-10-26 "strongest result yet" writeup was premature.**
With the full trajectory now visible (plot: `analysis/group_size_corrected.png`):

| config | iterations reached | pattern |
|---|---|---|
| GRPO, G=8, net=[256,256] | 60/60 (complete) | collapses hard by iter ~6-13, `degenerate_frac` -> 1.00 |
| GRPO, G=64, net=[512,512] | 27/60 (stopped) | **stable 8-17% through iter ~13, then declines, `degenerate_frac` starts climbing at iter 19, fully degenerate (1.00) by iter 26 -- it DOES eventually collapse, just ~2x later than G=8** |
| GRPO, G=128, net=[512,512] | 20/60 (stopped) | stable/slowly declining 18%->5.5% through iter 19, **`degenerate_frac` = 0.00 every single iteration so far** -- declining in success rate without the group-degeneracy mechanism that's killing G=8/G=64 |

**Corrected takeaway: larger G delays the collapse (roughly proportionally -- G=64
survived ~2x longer than G=8), it does not eliminate it.** G=128 hasn't shown
degenerate groups yet, but it's also only 1/3 of the way through its budget and its
success rate is declining anyway (18%->5.5%), so "G=128 solves it" is not yet a
supportable claim either -- it may just be that 128 is enough to delay group
degeneracy past where this run got cut off, not that the mechanism is fixed. The
success-rate decline happening even at `degenerate_frac=0.00` for G=128 is worth a note
for next session: it suggests group-level degeneracy isn't the *only* thing driving the
decline -- something else (policy entropy collapsing with training? the group std
denominator shrinking even without full agreement?) may also be eroding performance,
independent of the specific failure mode the group-size experiment targeted.

**RLOO and TGRPO ablation (both now effectively complete or near-complete):**

| algo (G=8, scale-matched) | iterations | nonzero | last nonzero iter | overall mean |
|---|---|---|---|---|
| GRPO | 60/60 | 8/60 | 13 | 0.0035 |
| RLOO | 54/60 (stopped, 41 straight zeros before that) | 8/54 | 13 | 0.0038 |
| TGRPO | 16/60 (stopped) | already mostly 0 by iter 13-16 | ~13-16 | n/a, too short |
| **SM-GRPO** | 60/60 (complete) | **27/60** | **39** | **0.0081** |

**This closes the ablation-isolation question from 2026-10-26:** RLOO (trajectory-level,
leave-one-out baseline) collapses identically to GRPO (same last-nonzero iteration: 13).
TGRPO (per-step, but same-timestep alignment, not state-matched) also collapses early,
same ballpark as GRPO/RLOO. **Only SM-GRPO, with actual k-NN state-matching, shows the
~3x delay.** This is good evidence the earlier positive result really is about
state-*matching* specifically, not just "any critic-free variant" or "any per-step
signal" -- strengthens Idea A's case.

**Operational fix for next time:** none of tonight's 60-iteration runs ever hit a
checkpoint (`checkpoint_every=1000`), so a session restart loses everything and forces
a full restart from the BC checkpoint with no way to resume. Should default
`checkpoint_every` much lower (e.g. 5-10) for any run expected to take this long, given
session restarts have now interrupted background jobs twice in two days.

## 2026-10-26 — Group-size scaling: the strongest result in the project so far (in progress, SUPERSEDED by the correction above)

Early results from `scripts/run_month3_scale_up.sh` (GRPO, `hidden_sizes=[512,512]`,
`match_advantage_scale=true`, `lr=3e-4`; G=64 at 13/60 iterations, G=128 at 10/60 --
**both still running**, numbers below will firm up):

| config | iterations so far | degenerate_group_frac | success rate range | mean success |
|---|---|---|---|---|
| GRPO, G=8, net=[256,256] | 60/60 (done) | climbed to 0.90-1.00 | collapsed to 0% by iter ~13 | 0.0035 |
| **GRPO, G=64, net=[512,512]** | 13/60 | **0.00 every iteration so far** | **7.8% - 16.7%, no collapse** | **0.118** |
| **GRPO, G=128, net=[512,512]** | 10/60 | **0.00 every iteration so far** | **10.2% - 18.0%, no collapse** | **0.132** |

**This is the best GRPO result in the whole project.** Zero degenerate groups in every
single iteration so far (vs. climbing to 90-100% at G=8), and success rate holding
*stably* in the 10-18% range rather than collapsing -- already well above G=8's brief
peak of 5.6% before it died. Plot: `analysis/results_to_date_full.png` (right panel).

**Important caveat -- this is confounded, not a clean G-only ablation:** the network
size changed too (`[256,256]` -> `[512,512]`), because the BC checkpoint had to be
retrained at the new size to warm-start from. So this result shows "bigger group +
bigger network helps a lot," not "bigger group alone helps" -- disentangling the two
(e.g. G=64 at the *original* `[256,256]` network, or G=8 at `[512,512]`) is the natural
next ablation before attributing the improvement to group size specifically.

**Also still incomplete:** both runs are ~15-20% of the way through their 60-iteration
budget. The pattern (zero degenerate groups, stable success rate) has been completely
consistent so far, but "stable for 10-13 iterations" is not yet "stable for 60" --
worth re-checking once they finish before calling this settled.

## 2026-10-26 — Operational note: TGRPO crashed (disk space), RLOO/TGRPO ablation data

`logs/month3_tgrpo_scalematched.log` crashed with `OSError: [Errno 28] No space left on
device` at iteration 26/60 -- `df -h /` showed only 4.3GB free (74% used) at the time.
This is a **system-level disk space issue on this Mac, not caused by this project**
(the project's own `outputs/`/`checkpoints/` dirs total a few MB) -- worth the user
checking what's consuming disk space generally. Relaunched TGRPO
(`logs/month3_tgrpo_scalematched_retry.log`, with `PYTHONUNBUFFERED=1` this time).

Partial data before the crash, plus RLOO's data (37/60 iterations, still running) --
useful for the "is SM-GRPO's delay specific to state-matching" question even
incomplete:

| algo (G=8, scale-matched) | nonzero iterations | last nonzero iter | overall mean success |
|---|---|---|---|
| GRPO | 8/60 | 13 | 0.0035 |
| RLOO (37/60 so far) | 8/37 | 13 | 0.0056 |
| TGRPO (26/60, crashed) | 10/26 | 25 (incomplete -- crashed while still nonzero) | 0.0058 |
| **SM-GRPO (full 60/60)** | **27/60** | **39** | **0.0081** |

RLOO tracks GRPO almost exactly (same last-nonzero iteration, 13) -- leave-one-out vs.
group-mean baseline doesn't matter much here. TGRPO's data is too incomplete to
conclude anything (it was still nonzero when it crashed) -- **wait for the TGRPO
retry to finish before drawing the "is it state-matching specifically" conclusion**;
right now it's the one open question the Month-3 positive result still depends on.

Also found a second instance of the TensorBoard-output-directory collision from
Month 2 (two `train.py` processes launched within the same wall-clock second get the
same `outputs/<date>/<time>/` dir and overwrite each other's event files) -- affected
the G=64/G=128 group-size runs tonight. Not yet fixed in code (just worked around by
reading the per-process stdout logs instead, which are written to distinct files); a
proper fix would make `train.py`'s log directory include the PID or a random suffix.

## 2026-10-26 — Novel extension proposed and implemented: MM-GRPO (not yet run)

User asked for "something novel." Diagnosis of why GRPO fails here (see chat) comes down
to two things PPO's critic gives that GRPO lacks: (1) a baseline that generalizes across
the *whole* training run, not just one iteration's rollouts, and (2) a baseline that
doesn't go degenerate just because one small group happened to agree. SM-GRPO already
measurably helps with a version of (2) (the scale-matched comparison above). **MM-GRPO
(Memory-Matched GRPO)** extends SM-GRPO to also address (1): give its k-NN search a
persistent, cross-iteration memory of (state, return-to-go) pairs, not just the current
group's siblings -- a non-parametric, critic-free approximation of a value function's
cross-iteration generalization. A flat index over float vectors is orders of magnitude
cheaper than a value network at VLA scale, so most of GRPO's memory advantage over PPO
should survive.

**Implemented and tested, not yet run:**
- `advantages/replay_buffer.py`: `StateReturnBuffer`, a FIFO buffer of (state,
  return-to-go, iteration) triples with optional age-based filtering.
- `advantages/group_baselines.py`: `state_matched_baselines` takes optional
  `extra_states`/`extra_returns` -- an untimed candidate pool merged into every query
  step's k-NN search alongside the current group's siblings.
- `algos/sm_grpo_memory.py`: identical to `sm_grpo.py` except it takes (and mutates) a
  buffer + the current iteration index -- the one `update()` function in the codebase
  that isn't stateless, since persistence across calls is the whole point.
- `configs/config.yaml` / `train.py`: wired in as `algo=sm_grpo_memory`.
- 6 new unit tests (`tests/test_replay_buffer.py`), all passing (18/18 total). Smoke-
  tested end-to-end (4 iterations, buffer visibly accumulating across them).

**Known risk, not yet mitigated:** buffer entries from early iterations reflect an
older, usually-worse policy, so their return-to-go may not reflect what the *current*
policy would get from a similar state -- the standard off-policy-baseline staleness
problem. Current mitigation is just `memory_max_age` (a recency window); a
policy-similarity-weighted variant is a natural follow-up, not implemented.

**Why not launched yet:** 4 training processes already running (the RLOO/TGRPO ablation
isolation + GRPO at G=64/G=128) on an 8-core machine with load average already ~10-12 --
launching a 5th CPU-bound job now would slow all of them down proportionally rather than
genuinely add parallel throughput. Queued to run once the current batch finishes.
Suggested first config: same scale-matched settings as the SM-GRPO win above (`groups=20`,
`group_size=8`, `lr=3e-4`, `match_advantage_scale=true`) so it's a direct, fair
comparison against the ~3x-delay result SM-GRPO already achieved.

## 2026-10-26 — Group-size scale-up launched (G=64, G=128) + larger network

User-requested follow-up: does a much larger within-group sample size (G) mitigate
GRPO's instability, with more policy capacity? Motivated by the diagnosis that small
G=8 groups give noisy, easily-degenerate comparisons (`degenerate_group_frac` climbed to
~0.90-0.95 in the G=8 scale-matched runs).

- Added `--hidden-sizes` to `bc_train.py` and `eval.py` (previously hardcoded to
  `[256, 256]`) so a differently-sized network can be trained/evaluated.
- Retrained the BC checkpoint at `[512, 512]` (doubled from `[256, 256]`):
  `checkpoints/bc_pick-place-v3_512.pt`, 13.3% success over 30 eval episodes
  (comparable to the original `[256,256]` checkpoint's 16.7% -- not identical, expected,
  different network/training run, same ballpark).
- Launched `scripts/run_month3_scale_up.sh`: GRPO with `hidden_sizes=[512,512]`,
  `match_advantage_scale=true`, `lr=3e-4`, at **G=64** (`num_groups_per_iteration=3`,
  192 episodes/iter) and **G=128** (`num_groups_per_iteration=2`, 256 episodes/iter) --
  `num_groups_per_iteration` reduced as G grows to keep episodes/iteration roughly
  comparable to the existing G=8/groups=20 runs (~160/iter), so wall-clock per iteration
  stays in the same ballpark rather than scaling 8-16x. 60 iterations each. Results
  pending in `logs/month3_grpo_g64_net512.log` / `logs/month3_grpo_g128_net512.log`.

**Operational note for future background runs:** found that piped/redirected Python
stdout is block-buffered by default, so a healthy, actively-running background job's log
file can look empty for a long time -- this caused a false "the job died silently"
diagnosis mid-session (it hadn't; a stale `ps aux | grep "python train.py"` pattern that
doesn't match the capitalized `Python` binary name in `ps` output also contributed, and
very nearly caused a duplicate run to be launched on top of the still-healthy original).
Added `export PYTHONUNBUFFERED=1` to `run_month3_scale_up.sh`; worth adding to future
scripts, and checking `ps aux | grep train.py` (no `python` prefix) rather than `grep
"python train.py"` when verifying a background job is actually running.

## 2026-10-26 — First positive (partial) result: SM-GRPO measurably delays the collapse

`scripts/run_month3_scale_matched.sh` finished. Plot: `analysis/month3_scale_matched_final.png`.

| run | nonzero-success iterations | last nonzero iter (of 60) | overall mean success | last-5 mean |
|---|---|---|---|---|
| PPO, lr=3e-4 | 60/60 | — (never zero) | 0.135 | 0.266 |
| GRPO, scale-matched, lr=3e-4 | 8/60 | iteration 13 | 0.0035 | 0.000 |
| **SM-GRPO, scale-matched, lr=3e-4** | **27/60** | **iteration 39** | **0.0081** | 0.000 |

**This is the first comparison in the whole project where SM-GRPO beats vanilla GRPO
under conditions I'd actually call fair** (same batch size, same LR, same advantage
scale). GRPO is functionally dead by iteration ~6-13 (one stray nonzero blip at 13, then
permanently 0 for the remaining 46 iterations). SM-GRPO keeps producing successful
rollouts intermittently all the way to iteration 39 — roughly **3x longer** before going
fully silent — and its overall mean success rate across the full run is **2.3x GRPO's**.

**Being precise about what this is and isn't:** SM-GRPO does *not* solve the problem.
Both algorithms end at a hard 0% success floor by the end of the 1.44M-step budget, and
neither gets remotely close to PPO's ~27-31%. What this run shows is narrower and more
defensible: **state-matched per-step baselines measurably slow the collapse relative to
vanilla GRPO's whole-trajectory baseline, under matched advantage scale, LR, and batch
size.** That's a real, positive signal for Idea A's core mechanism (Section 4.2's claim
that trajectory-level comparisons are noisier than they need to be) — the first one this
project has produced after two negative/confounded attempts.

**Caveats before this goes in a paper draft:**
- Single seed, single task (pick-place-v3), single hyperparameter setting (k=4,
  time_window=5) — the roadmap's actual ablation sweep (k, time_window, embedding,
  seeds) still hasn't run. This result could be noise; it needs replication before it's
  a claim, not just a lead.
- "Delays collapse by 3x" is still a negative outcome in absolute terms (collapses
  either way) — the honest framing for a paper is "reduces instability" or "partially
  mitigates," not "solves" or "stabilizes."
- RLOO and TGRPO still haven't been re-tested with `match_advantage_scale=true` — it's
  possible they'd show a similar partial delay too, which would weaken the claim that
  state-matching *specifically* (vs. any alternative critic-free baseline) is doing the
  work. This is the most important next experiment to rule out before trusting the
  result above as evidence for Idea A specifically.

## 2026-10-26 — Closed the scale-mismatch confound, launched the real GRPO-vs-SM-GRPO test

Follow-up to 2026-10-25's note that SM-GRPO v2's `advantage_abs_mean` (~0.21-0.24) and
GRPO's (~0.74-0.85) weren't actually scale-matched to each other despite both being
individually "normalized" -- so any earlier head-to-head wasn't trustworthy.

**Fix**: added `training.match_advantage_scale` (default `false`, so existing runs stay
reproducible) to `configs/config.yaml`, wired through `grpo.py`, `rloo.py`, `tgrpo.py`,
`sm_grpo.py` -- when `true`, the final flattened advantage batch is z-scored to
zero-mean/unit-variance (`algos/common.py::normalize_batch_advantages`, which already
existed but was unused) right before the policy-gradient loss, for every critic-free algo
identically. PPO doesn't need this -- its own GAE advantages are already batch-normalized.

**Verified with a 2-iteration smoke test** before committing to a long run:
`advantage_abs_mean` with `match_advantage_scale=true` came out to 0.705-0.846 (GRPO) vs.
0.464-0.467 (SM-GRPO) -- much closer than the prior 17x and 3-4x mismatches. Not
*identical*, and that's expected: GRPO's advantage is piecewise-constant across an entire
trajectory (same value broadcast to every step) while SM-GRPO's varies continuously
per-step, so even at matched batch variance the two have different mean-absolute-value
under a different distribution shape. This is close enough to trust a head-to-head now.

**Launched** `scripts/run_month3_scale_matched.sh`: GRPO and SM-GRPO, both with
`match_advantage_scale=true`, same settings as every other Month-2/3 run (`groups=20`,
`lr=3e-4`, BC warm-start on pick-place-v3), full 60 iterations each, in parallel. This is
the first comparison between these two algorithms that should actually be fair. Results
pending -- check `logs/month3_grpo_scalematched.log` / `logs/month3_sm_grpo_scalematched.log`.

## 2026-10-25 — Month 3 v2 result: fixed, but still an honest negative/neutral finding

`scripts/run_month3_sm_grpo_v2.sh` finished (full 60 iterations, `robust_normalize=true`,
`lr=3e-4`, `groups=20`, same everything else as the controlled Month-2 baselines). Full
five-way comparison plot: `analysis/month3_full_comparison.png`.

| run | last-5 mean success | collapses to a hard 0% floor? | when |
|---|---|---|---|
| PPO, lr=3e-4 | **0.266** | no — climbs | — |
| GRPO, lr=3e-5 | 0.010 | no — stays noisy/nonzero | — |
| **SM-GRPO v2, lr=3e-4 (fixed)** | **0.000** | **yes** | **~iteration 19-20** |
| GRPO, lr=3e-4 | 0.000 | yes | ~iteration 6 |
| SM-GRPO v1, lr=3e-4 (buggy, unnormalized) | 0.000 | yes | ~iteration 3 |

**Honest read: fixing the normalization bug helped (collapses later than v1, and later
than vanilla GRPO at the same LR), but SM-GRPO still ends up at the same destination —
a hard, permanent 0% success floor — within the 60-iteration budget.** It does *not*
currently beat GRPO at its best setting (`lr=3e-5`, which degrades but never hits a hard
floor). So as implemented today (`k=4`, `time_window=5`, identity/raw-obs embedding),
**SM-GRPO has not yet demonstrated an advantage over vanilla GRPO** — Idea A's core claim
(state-matched per-step baselines reduce the instability from Section 4.2) is not yet
supported by these runs.

**One more scale-mismatch wrinkle worth knowing before reading too much into the "v2
collapses slower" result:** `sm_grpo v2`'s `advantage_abs_mean` at iteration 0 was
**~0.21-0.24** — noticeably *smaller* than GRPO's ~0.74-0.85 at the same nominal LR (not
larger, this time). `robust_normalize` divides by `robust_scale(group.returns)`, a
group-level trajectory-return scale, which isn't the same normalization GRPO itself uses
(division by the group's own std) — the two aren't actually scale-matched to each other,
just each individually "reasonable." So part of why v2 collapses *slower* than vanilla
GRPO could simply be a smaller effective step size again, not evidence that state-matching
itself is a better credit-assignment signal. **This confound needs to be closed (e.g., by
making both algorithms' advantages have matched unit variance, the way PPO's GAE
normalization does) before SM-GRPO vs. GRPO head-to-head results can be trusted.**

**What this session did NOT do, which is the obvious next step:** the actual Month-3
ablation sweep the roadmap asks for (k, time_window, embedding choice, with/without
common random numbers) — this was only ever "does a first naive run work at all," and the
honest answer right now is "not yet, and the normalization confound above needs fixing
before the comparison is even fair." Recommended next steps, in order:
1. Properly scale-match GRPO's and SM-GRPO's advantages (e.g. z-score both to unit
   variance over the training batch) before any further head-to-head comparison.
2. Only then run the k / time_window ablation.
3. Consider whether the identity embedding (raw 39-dim Meta-World obs) is actually a
   good state-similarity metric for k-NN matching -- per-dimension scales are fairly
   uniform here (checked: std 0.02-0.2 across dims, no dominant-scale dimension), so this
   is probably not the first thing to fix, but worth a sanity check (e.g., does k-NN
   actually find states from a similar *phase* of the task, not just nearby raw
   coordinates) once (1) and (2) are settled.

This is a good stopping point for an unattended session: Month 2 is fully and rigorously
settled (GRPO/RLOO/TGRPO underperform PPO, confirmed controlling for both LR and batch
size), Month 3 has a first honest result plus two real implementation bugs found and
fixed along the way, and the next steps require a judgment call (how to scale-match the
advantages, which embedding to try) that's worth a human decision rather than more
autonomous compute.

## 2026-10-25 — Month 3 v1 negative result diagnosed and fixed: missing normalization

SM-GRPO's first run (below) collapsed *faster* than vanilla GRPO (0% success by
iteration 3 vs. GRPO's iteration 6), which on its face looked bad for Idea A. Checked
`advantage_abs_mean` (logged to TensorBoard, not stdout) for both runs:

| run | iter 0 advantage_abs_mean | iter 5 |
|---|---|---|
| grpo, lr=3e-4 | 0.742 | 0.733 |
| **sm_grpo v1, lr=3e-4** | **12.591** | **0.350** (already collapsing) |

**Root cause: `state_matched_baselines` (`advantages/group_baselines.py`) is the only
advantage function in the codebase that returns a raw, unnormalized
`return_to_go - baseline` difference** — `trajectory_advantages`,
`robust_trajectory_advantages`, and `leave_one_out_advantages` all divide by a std or
robust-scale estimate; this one didn't. At `lr=3e-4`, that's a ~17x larger effective
step size than GRPO's properly-scaled advantage, at the *same* nominal learning rate —
enough on its own to explain the faster collapse, independent of whether state-matching
itself is a good idea.

**Fix:** `algos/sm_grpo.py` already had an unused `robust_normalize` option (divides by
`robust_scale(group.returns)`, the same scale GRPO's Idea-B toggle uses) — just never
turned on by default. Flipped `sm_grpo.robust_normalize` to `true` in
`configs/config.yaml` (was `false`), added a loud warning to
`state_matched_baselines`'s docstring, and verified with a 10-iteration check
(`logs/`, not committed) that it prevents the immediate collapse before committing to a
full 60-iteration re-run (`scripts/run_month3_sm_grpo_v2.sh`,
`logs/month3_sm_grpo_v2_robust.log`). **Results of that full run are in the next
section** (or pending, if you're reading this before it's finished).

**Lesson for next session:** before trusting *any* advantage estimator's comparison
against GRPO/PPO in this codebase, check `advantage_abs_mean` in TensorBoard first — it's
logged but not printed to stdout, so it's easy to miss a scale mismatch like this one
just from the printed `[iter ...]` lines.

## 2026-10-25 — Month 3 kicked off: first real SM-GRPO run (v1, see correction above)

Launched `scripts/run_month3_sm_grpo.sh`: SM-GRPO (Idea A, `k=4`, `time_window=5`) at
the exact Month-2-matched settings (pick-place-v3, BC warm-start, `groups=20`,
`group_size=8`, `lr=3e-4`, 60 iterations) for a direct comparison against the existing
PPO (~27-31%) and GRPO (collapses to 0%) runs at the same LR. Timed one iteration first
(~93s, same cost as GRPO/PPO at this scale -- the k-NN overhead is negligible here) before
committing to the full ~93-minute run. Results pending.

**Caveat going in:** this is a first run at default k/time_window, not yet the Month-3
ablation sweep the roadmap asks for (k, time_window, with/without robust normalisation,
with/without common random numbers) -- treat this as "does it work at all" before
"how well does it work relative to its own hyperparameters."

## 2026-10-25 — Month 2 final verdict (properly controlled): GRPO degrades at both LRs

`scripts/run_lr_matched_batch.sh` finished: GRPO at `lr=3e-5`, `groups=20` (matching the
real Month-2 batch size exactly), full 60 iterations. Plot:
`analysis/month2_lr_matched_final.png`.

| run | first-10 mean | last-10 mean | last-5 mean | max |
|---|---|---|---|---|
| PPO, lr=3e-4 | ~0.06 | — | **0.266** | 0.306 |
| GRPO, lr=3e-4 (original Month 2) | 0.062 | 0.000 (from iter 6 on) | 0.000 | 0.056 |
| **GRPO, lr=3e-5 (matched batch)** | **0.054** | **0.019** | **0.010** | **0.094** |

**Verdict, now properly controlled for batch size: lower LR does not fix GRPO's
degradation, it only slows it down.** At `lr=3e-5` GRPO doesn't hit a hard 0% floor the
way it does at `lr=3e-4`, but it still trends *downward* over the full 60 iterations
(first-10 mean 0.054 → last-10 mean 0.019, roughly a 65% relative decline) while PPO
climbs to ~27-31% over the same budget. **This restores confidence in the original
Month-2 headline finding** — GRPO's problem on this task isn't simply "the Hopper-v5
learning rate was too high for a BC-warm-started policy," it degrades at both a 10x-lower
LR and the original LR, just at different speeds. This is consistent with the roadmap's
framing (Section 4.2): a real advantage-estimator / group-construction issue, which is
what motivates Month 3's SM-GRPO, not merely a hyperparameter that needed tuning.

**Still open, for rigor before writing this up as a result:** RLOO and TGRPO haven't
been re-tested at matched batch size + lower LR (only GRPO was, to answer the question
cheaply) — they looked identical to GRPO at `lr=3e-4`/`groups=3`, but that run is now
known to be batch-mismatched, so don't cite their lr=3e-5 behavior without a matched-batch
re-run. `degenerate_group_frac` in the matched lr=3e-5 run climbs to ~0.90-0.95 by the
final iterations (using the now-fixed success-based diagnostic) — worth plotting
alongside success_rate next session, since "most groups becoming degenerate as training
progresses" would be a very direct confirmation of the Section 4.2 #3 sparse-reward
failure mode as the actual mechanism, not just a correlated symptom.

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
