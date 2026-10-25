"""Advantage estimators shared across algos/*.py (roadmap Section 9: "advantages/: group
mean and std, robust normalisation, leave-one-out, state-matched k-NN baseline,
trajectory and step fusion").

All functions take/return numpy arrays and operate on a single group (G rollouts sharing
one initial state, roadmap Section 4.2/9) so they can be unit-tested without touching the
rollout buffer or any particular env.
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Vanilla GRPO: trajectory-level, std-normalised group-relative advantage.
# ---------------------------------------------------------------------------
def trajectory_advantages(returns: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Vanilla GRPO advantage: A_i = (R_i - mean(R)) / (std(R) + eps).

    `returns` has shape (G,) -- one scalar return per rollout in the group. This is the
    baseline we expect to reproduce the instability described in roadmap Section 4.2.
    """
    mean = returns.mean()
    std = returns.std()
    return (returns - mean) / (std + eps)


# ---------------------------------------------------------------------------
# Idea B: robust advantage normalisation.
# ---------------------------------------------------------------------------
def robust_scale(returns: np.ndarray, sigma_min: float = 1e-2, method: str = "clip") -> float:
    """A denominator that can't collapse to ~0 and blow up the advantage (Section 4.2 #2).

    method:
      "clip" -- max(std, sigma_min)
      "mad"  -- 1.4826 * median absolute deviation (robust to outlier rollouts), floored
                at sigma_min
    """
    if method == "clip":
        return max(float(returns.std()), sigma_min)
    if method == "mad":
        median = np.median(returns)
        mad = np.median(np.abs(returns - median))
        return max(1.4826 * float(mad), sigma_min)
    raise ValueError(f"unknown robust_scale method: {method}")


def robust_trajectory_advantages(
    returns: np.ndarray, sigma_min: float = 1e-2, method: str = "clip"
) -> np.ndarray:
    mean = returns.mean()
    scale = robust_scale(returns, sigma_min=sigma_min, method=method)
    return (returns - mean) / scale


def is_degenerate_group(returns: np.ndarray, success_threshold: float = 0.0) -> bool:
    """True if every rollout in the group scored above `success_threshold` or every
    rollout scored at/below it (Section 4.2 #3). Such groups give ~zero advantage and
    ~zero gradient under vanilla GRPO.

    Only meaningful for sparse/binary rewards where `returns` really are 0/1 (or a
    threshold cleanly separates success from failure). For a dense-shaped reward (e.g.
    Meta-World's default per-step shaping), trajectory returns are continuous and almost
    always > 0 regardless of task success, so this degenerates to "always True" and
    stops being informative -- use `is_degenerate_group_by_success` instead in that case.
    """
    successes = returns > success_threshold
    return bool(successes.all() or (~successes).all())


def is_degenerate_group_by_success(successes: np.ndarray) -> bool:
    """Same check as `is_degenerate_group`, but on the env's actual binary
    info["success"] flag per rollout rather than a threshold on a (possibly dense,
    continuous) return. This is the correct diagnostic for Meta-World-style envs with
    dense shaping rewards (see `is_degenerate_group`'s docstring).
    """
    successes = np.asarray(successes, dtype=bool)
    return bool(successes.all() or (~successes).all())


# ---------------------------------------------------------------------------
# RLOO: leave-one-out baseline (critic-free, lower variance than the group mean because
# rollout i's own return never contributes to its own baseline).
# ---------------------------------------------------------------------------
def leave_one_out_advantages(returns: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    g = len(returns)
    total = returns.sum()
    loo_mean = (total - returns) / (g - 1)
    std = returns.std()
    return (returns - loo_mean) / (std + eps)


# ---------------------------------------------------------------------------
# Idea A (main contribution): State-Matched Group Baselines (SM-GRPO).
# ---------------------------------------------------------------------------
def state_matched_baselines(
    group_states: list[np.ndarray],
    group_returns_to_go: list[np.ndarray],
    k: int = 4,
    time_window: int = 5,
) -> list[np.ndarray]:
    """Per-step, critic-free baseline from k-NN across the group's *other* rollouts in a
    low-dimensional state-embedding space, within a time window (roadmap Section 5, Idea
    A). Reduces variance from stochastic dynamics vs. a single whole-trajectory baseline.

    Args:
        group_states: length-G list of (T_g, D) state embeddings, one array per rollout
            (T_g may differ per rollout).
        group_returns_to_go: length-G list of (T_g,) return-to-go arrays, aligned with
            group_states.
        k: number of neighbours to average.
        time_window: only consider neighbour steps within +/- time_window timesteps of
            the query step (keeps the match temporally local, not just state-similar).

    Returns:
        length-G list of (T_g,) per-step advantages: return_to_go - kNN baseline.
    """
    g = len(group_states)
    advantages = []
    for i in range(g):
        states_i = group_states[i]
        rtg_i = group_returns_to_go[i]
        t_len = len(states_i)
        adv_i = np.zeros(t_len, dtype=np.float64)

        for t in range(t_len):
            query = states_i[t]
            candidates_states = []
            candidates_rtg = []
            for j in range(g):
                if j == i:
                    continue
                lo = max(0, t - time_window)
                hi = min(len(group_states[j]), t + time_window + 1)
                if lo >= hi:
                    continue
                candidates_states.append(group_states[j][lo:hi])
                candidates_rtg.append(group_returns_to_go[j][lo:hi])

            if not candidates_states:
                # No other rollout reaches a nearby timestep -- fall back to the group's
                # trajectory-level mean return-to-go at this step's own rollout horizon.
                adv_i[t] = rtg_i[t] - rtg_i.mean()
                continue

            cand_states = np.concatenate(candidates_states, axis=0)
            cand_rtg = np.concatenate(candidates_rtg, axis=0)

            dists = np.linalg.norm(cand_states - query[None, :], axis=1)
            kk = min(k, len(dists))
            nn_idx = np.argpartition(dists, kk - 1)[:kk]
            baseline = cand_rtg[nn_idx].mean()
            adv_i[t] = rtg_i[t] - baseline

        advantages.append(adv_i)
    return advantages


# ---------------------------------------------------------------------------
# TGRPO-style fusion of trajectory-level and step-level advantages.
# ---------------------------------------------------------------------------
def fuse_trajectory_and_step(
    trajectory_adv: float, step_adv: np.ndarray, alpha: float = 0.5
) -> np.ndarray:
    """A_t = alpha * trajectory_adv + (1 - alpha) * step_adv_t (roadmap: TGRPO "fuses
    step-level and trajectory-level advantages"). `alpha` is a tunable ablation knob.
    """
    return alpha * trajectory_adv + (1 - alpha) * step_adv


def compute_returns_to_go(rewards: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    rtg = np.zeros_like(rewards, dtype=np.float64)
    running = 0.0
    for t in reversed(range(len(rewards))):
        running = rewards[t] + gamma * running
        rtg[t] = running
    return rtg
