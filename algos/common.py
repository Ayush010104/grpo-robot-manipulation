"""Shared helpers for algos/*.py: flattening groups into training batches, gradient
clipping, and KL-to-reference tracking (roadmap Section 9: "Keep a KL penalty or clipping
to the reference (BC) policy to prevent collapse").
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from algos.rollout_buffer import RolloutGroup


def flatten_group_steps(
    groups: list[RolloutGroup], per_trajectory_advantage: list[list[np.ndarray]]
):
    """Flatten (obs, action, advantage) across every step of every trajectory in every
    group. `per_trajectory_advantage[g][i]` is a per-step advantage array (T_i,) for
    trajectory i of group g -- for trajectory-level methods just broadcast the scalar.
    """
    obs_chunks, act_chunks, adv_chunks = [], [], []
    for group, group_adv in zip(groups, per_trajectory_advantage):
        for traj, adv in zip(group.trajectories, group_adv):
            obs_chunks.append(traj.obs)
            act_chunks.append(traj.actions)
            adv_chunks.append(np.asarray(adv, dtype=np.float32))
    return (
        np.concatenate(obs_chunks, axis=0),
        np.concatenate(act_chunks, axis=0),
        np.concatenate(adv_chunks, axis=0),
    )


def normalize_batch_advantages(adv: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Optional extra batch-level standardisation on top of the group-level advantage
    (common PPO/GRPO trick; keep off by default to study the group-level effect in
    isolation -- toggle from train.py config).
    """
    return (adv - adv.mean()) / (adv.std() + eps)


@torch.no_grad()
def kl_to_reference(policy, reference_policy, obs: torch.Tensor) -> float:
    """Approximate KL(policy || reference) via the analytic Gaussian KL on the
    pre-squash distribution (reference is typically the frozen BC policy).
    """
    p = policy.distribution(obs)
    q = reference_policy.distribution(obs)
    kl = torch.distributions.kl_divergence(p, q).sum(-1)
    return float(kl.mean().item())


def clip_grad_and_step(policy: nn.Module, optimizer: torch.optim.Optimizer, max_norm: float = 1.0) -> float:
    grad_norm = nn.utils.clip_grad_norm_(policy.parameters(), max_norm)
    optimizer.step()
    optimizer.zero_grad()
    return float(grad_norm)


def group_diagnostics(groups: list[RolloutGroup], degenerate_mask: list[bool]) -> dict:
    returns = np.concatenate([g.returns for g in groups])
    return {
        "mean_return": float(returns.mean()),
        "group_return_std_mean": float(np.mean([g.returns.std() for g in groups])),
        "degenerate_group_frac": float(np.mean(degenerate_mask)),
        "success_rate": float(np.mean([g.success_rate for g in groups])),
    }
