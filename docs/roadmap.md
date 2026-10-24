# Research Roadmap: Stable Group-Relative RL Fine-Tuning for Robot Policies

Source: `RL Robotics Research Roadmap Stable GRPO for Robot Policies.pdf` (prepared Oct 2026).
This is a condensed, in-repo copy for reference during the coding session.

## 1. Background

- Prior work: two papers on RL for wireless comms (RIS, stacked RIS, joint power/phase
  optimisation). Algorithms already implemented: PPO, SAC, TD3, GRPO, DDPG and variants.
- Goal: move into continuous control for robotics, produce a genuinely novel algorithmic
  contribution.
- Transferable strengths: high-dim continuous action spaces, constrained optimisation
  (power budgets), hands-on GRPO experience.

## 2. Chosen direction

RL fine-tuning of pretrained robot policies, focused on making GRPO-style critic-free
optimisation stable for robot control.

- VLA models (OpenVLA, OpenVLA-OFT, pi0, pi0.5) are trained mainly by imitation; RL
  fine-tuning is now the main way to push past demonstrations.
- GRPO drops the critic → ~half the memory at VLA scale.
- GRPO is known to be unstable in robotics — open problem, that's the contribution gap.

## 3. Prior work (verify each before citing)

| Work | Date | Base model | Algorithm | Key point |
|---|---|---|---|---|
| SimpleVLA-RL (Li et al.) | 2025 | OpenVLA-OFT | GRPO | RL improves long-horizon planning when demos are scarce |
| TGRPO (arXiv 2506.08440) | 2025 | OpenVLA | Trajectory-wise GRPO | Fuses step-level and trajectory-level advantages |
| RL4VLA (arXiv 2505.19789) | 2025 | OpenVLA | PPO vs GRPO vs DPO | PPO beat GRPO; non-stationary dynamics destabilise GRPO; shared actor-critic backbone saves ~45% VRAM |
| VLA-RL (Lu et al.) | 2025 | OpenVLA | PPO + process reward model | Learned robotic process reward model |
| iRe-VLA (Guo et al.) | 2025 | various | Alternating RL/SFT | Iterates RL exploration and supervised updates |
| RIPT-VLA (Tan et al.) | 2025 | QueST, OpenVLA-OFT | RLOO | Leave-one-out REINFORCE baseline |
| pi_RL (arXiv 2510.25889) | 2025 | pi0/pi0.5 (flow) | Online RL | RL for flow-matching action heads |
| WMPO | 2025, ICLR 2026 poster | OpenVLA-OFT | GRPO in world model | Fine-tunes inside a learned world model |
| ProphRL | 2025 | VLA-Adapter, pi0.5, OpenVLA-OFT | FA-GRPO | Flow-action GRPO, model-based |
| EVOLVE-VLA | Dec 2025 | OpenVLA-OFT | GRPO + VLAC reward | Test-time / self-improvement style |
| SA-VLA | Jan 2026 | pi0.5 | PPO | |
| Green-VLA | Jan 2026 | Green-VLA | IQL + actor-critic | Offline-to-online |
| EG-GRPO | 2026 | VLA, aerial nav | Expert-guided GRPO | ~2.13x success over SFT |
| RLRC | 2025 | OpenVLA (ManiSkill) | SFT + PPO | Recovers performance after 90% pruning |

Reading list tracker: GitHub `Denghaoyuan123/Awesome-RL-VLA`. Before committing to any
Section 5 idea, search arXiv for 2026 papers on: "GRPO robot advantage normalization",
"state-matched group baseline", "GRPO sparse reward robot", "critic-free RLVLA stability".

## 4. The open problem

**4.1 Why GRPO works for LLMs.** Every completion in a group starts from the same prompt;
environment is deterministic given tokens. Group mean is a fair baseline:
`A_i = (R_i - mean(R)) / (std(R) + eps)`.

**4.2 Why it breaks for robots**
1. Stochastic, non-stationary dynamics — same init state, different rollouts; high return
   may be luck, not skill → noisy group comparisons.
2. Std normalisation blow-up — near-identical group returns → tiny std → noise amplified.
3. Sparse binary rewards — early training, whole groups fail (all zero) → zero advantage,
   zero gradient, policy stalls.
4. Long horizons / credit assignment — a single trajectory-level advantage spread
   uniformly over hundreds of steps.
5. Reset diversity — groups need identical initial states; fine in sim, not on real robots.

**4.3 Research question.** Can a critic-free, group-relative method match or beat PPO's
stability and sample efficiency in robot policy fine-tuning, while keeping GRPO's memory
and simplicity advantages?

## 5. Candidate novel algorithms

Pick one as the main contribution, keep others as ablations/future work.

- **Idea A — State-Matched Group Baselines (SM-GRPO) [RECOMMENDED MAIN PATH].**
  Baselines from segments of different rollouts passing through similar states, not one
  baseline per whole-trajectory group. Embed states (proprioception + object pose, or
  frozen visual encoder); at each step find k-NN across the group's rollouts within a time
  window, use their mean return-to-go as baseline. Per-step, critic-free advantage, lower
  variance from dynamics noise. Risks: k-NN cost, embedding quality for images — mitigate
  by starting with low-dim state.
- **Idea B — Robust Advantage Normalisation.** Replace std with `max(std, sigma_min)`,
  MAD, or running global scale across groups. Mask/down-weight degenerate (all-success or
  all-fail) groups. Cheap; likely an ablation rather than a standalone paper.
- **Idea C — Sub-Goal Partial Credit for All-Fail Groups.** Rank all-fail rollouts by
  progress signals (gripper-object distance, grasp achieved, object lifted, distance to
  target) for a dense tie-breaker reward, used only when binary reward gives no signal.
  Could later use a VLM as progress scorer.
- **Idea D — Noise-Controlled Grouping (Common Random Numbers).** Fix the physics seed
  across all group members in sim, so only the policy's action sampling differs — isolates
  the policy's contribution to return. Measure how much GRPO instability disappears under
  controlled noise (itself a useful analysis result). Later: sim-to-real-friendly variant
  that randomises seeds but pairs rollouts.
- **Idea E (bridge to wireless background) — Constraint-Aware GRPO.** Torque/energy/safety
  budgets via a Lagrangian multiplier on group-relative constraint advantages (mirrors
  power constraints in RIS work). Benchmark on Safety-Gymnasium or a ManiSkill energy-cost
  task.

**Recommended main path:** Idea A as the core contribution, Ideas B and D as analysis/
ablations, Idea C as an optional add-on.

## 6. Compute reality and environment setup (Mac)

Full 7B VLA RL fine-tuning is not feasible on a Mac. Do algorithm development with small
pretrained policies (MLP/small transformer via BC, or a small diffusion/ACT policy from
LeRobot). The algorithmic question transfers; scale to OpenVLA-OFT later on a rented GPU.

Simulators that run on Apple Silicon:
- **Gymnasium MuJoCo** (Hopper, Walker2d, Ant) — quick sanity checks.
- **Meta-World+** (Farama, NeurIPS 2025 D&B) — 50 MuJoCo manipulation tasks; good for
  sparse-success manipulation and multi-task tests.
- **LIBERO** (robosuite + MuJoCo) — the benchmark used by most VLA-RL papers. Check Mac
  install, use CPU/MPS backend.
- **ManiSkill3** — best with GPU; low-parallelism CPU possible, use later on cloud.

Libraries: PyTorch (MPS backend), Gymnasium, Metaworld, LeRobot (HF), Minari (offline
datasets), W&B/TensorBoard.

Install sequence: Python 3.10/3.11 venv → `pip install torch gymnasium[mujoco] metaworld
lerobot minari wandb` → install LIBERO from GitHub, run one demo task → verify random and
scripted policies run in each environment.

## 7. Three-month plan

**Month 1 — Foundations.** Read SimpleVLA-RL, TGRPO, RL4VLA, RIPT-VLA, pi_RL; one-page
summary each (setup, reward, group construction, failure modes). Set up Meta-World+ and
LIBERO on the Mac. Train a BC policy on 3-5 tasks (Meta-World pick-place/door-open/
drawer-open/button-press, or LIBERO-Spatial) to ~30-60% partial success, leaving room for
RL to help.

**Month 2 — Reproduce the failure.** Implement PPO (with critic), vanilla GRPO
(trajectory-level, std-normalised), RLOO, TGRPO-style fused advantages on top of the BC
policy. Binary success reward + optional dense-shaped variant. Log: group return std,
fraction of degenerate groups, advantage magnitude, gradient norm, KL to BC policy,
success rate, env steps to target success rate. Confirm the instability (GRPO variance/
collapse vs PPO).

**Month 3 — Novel method and ablations.** Implement SM-GRPO (Idea A). Ablate: k, time
window, embedding type, with/without robust normalisation (B), with/without common random
numbers (D). Compare vs PPO/GRPO/RLOO/TGRPO, 5+ seeds/task. Write a 4-8 page workshop
paper draft (CoRL/ICRA/NeurIPS workshops, or arXiv).

## 8. Experimental protocol

- Seeds: ≥5 per configuration. Report mean + 95% CI (rliable / IQM).
- Fixed budget: equal env steps and equal wall-clock comparisons.
- Metrics: final success rate, AUC of success curve, steps to 80% of final success,
  training stability (variance across seeds, # collapses), memory use.
- Generalisation: held-out object positions/textures/unseen tasks (RL4VLA vision/
  semantics/execution axes).

## 9. Code structure

```
envs/        wrappers for Meta-World+, LIBERO, Gymnasium; unified reset(seed), success
             flags, optional progress signals, group-reset support (same init state for
             G rollouts)
policies/    BC policy (MLP/small transformer), Gaussian/tanh-squashed head, optional
             diffusion head later
algos/       ppo.py, grpo.py, rloo.py, tgrpo.py, sm_grpo.py — shared rollout buffer
advantages/  group mean/std, robust normalisation, leave-one-out, state-matched k-NN
             baseline, trajectory/step fusion
train.py     config-driven (Hydra/YAML), logging + checkpointing
eval.py      success rate, generalisation splits, video recording
analysis/    plotting, rliable statistics, diagnostic dashboards
```

Implementation notes:
- Group construction: G = 8-16 rollouts from the same initial state, optional shared
  physics seed.
- Keep a KL penalty or clipping to the reference (BC) policy to prevent collapse.
- Start from low-dim state observations, move to images once the algorithm works.

## 10. Stretch goals

- Scale best method to OpenVLA-OFT or pi0-family on LIBERO, cloud GPU.
- Model-based variant: group rollouts inside a learned world model, vs WMPO.
- Meta-RL angle: group-relative updates for fast test-time adaptation, Meta-World ML10.
- Bridge to wireless background: robots adapting under communication delay/dropout.

## 11. Immediate next steps

1. Verify Idea A has not been published (arXiv + Awesome-RL-VLA).
2. Set up the environment (Section 6).
3. Train BC policies on 3 Meta-World+ tasks.
4. Implement vanilla GRPO and PPO, reproduce the instability.
5. Implement SM-GRPO and run ablations.
