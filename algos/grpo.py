"""Vanilla GRPO baseline (roadmap Section 7, Month 2): trajectory-level, std-normalised
group-relative advantage. Expected to reproduce the instability from Section 4.2.
"""
from __future__ import annotations

import numpy as np
import torch

from algos.common import (
    clip_grad_and_step,
    flatten_group_steps,
    group_diagnostics,
    kl_penalty_term,
    kl_to_reference,
    normalize_batch_advantages,
)
from algos.rollout_buffer import RolloutGroup
from advantages.group_baselines import (
    group_mean_centered_advantages,
    is_degenerate_group_by_success,
    robust_trajectory_advantages,
    trajectory_advantages,
)


def update(
    policy,
    optimizer,
    groups: list[RolloutGroup],
    reference_policy=None,
    device: str = "cpu",
    robust_normalize: bool = False,  # Idea B toggle
    sigma_min: float = 1e-2,
    grad_clip: float = 1.0,
    match_advantage_scale: bool = False,
    kl_coef: float = 0.0,
    # "group_std" (default) = current GRPO, divide by (robust-scaled) group std.
    # "mean_only" = REINFORCE++-style: no per-group division at all, rely solely on
    # match_advantage_scale's single global batch normalization. Added 2026-10-28 for
    # the collapse-fixes comparison (docs/novelty_check.md); see
    # advantages/group_baselines.py::group_mean_centered_advantages.
    normalize_mode: str = "group_std",
) -> dict:
    per_traj_adv = []
    degenerate_mask = []
    for group in groups:
        returns = group.returns
        successes = np.array([t.succeeded for t in group.trajectories])
        degenerate_mask.append(is_degenerate_group_by_success(successes))
        if normalize_mode == "mean_only":
            adv = group_mean_centered_advantages(returns)
        elif robust_normalize:
            adv = robust_trajectory_advantages(returns, sigma_min=sigma_min)
        else:
            adv = trajectory_advantages(returns)
        # broadcast each trajectory's scalar advantage over all its timesteps
        per_traj_adv.append(
            [np.full(len(t.rewards), a, dtype=np.float32) for t, a in zip(group.trajectories, adv)]
        )

    obs_np, act_np, adv_np = flatten_group_steps(groups, per_traj_adv)
    if match_advantage_scale:
        adv_np = normalize_batch_advantages(adv_np)
    obs = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
    actions = torch.as_tensor(act_np, dtype=torch.float32, device=device)
    advantages = torch.as_tensor(adv_np, dtype=torch.float32, device=device)

    log_probs = policy.log_prob(obs, actions)
    pg_loss = -(log_probs * advantages).mean()
    kl_term = None
    if kl_coef > 0.0 and reference_policy is not None:
        kl_term = kl_penalty_term(policy, reference_policy, obs)
        loss = pg_loss + kl_coef * kl_term
    else:
        loss = pg_loss
    loss.backward()
    grad_norm = clip_grad_and_step(policy, optimizer, max_norm=grad_clip)

    diag = group_diagnostics(groups, degenerate_mask)
    diag.update(
        {
            "loss": float(loss.item()),
            "pg_loss": float(pg_loss.item()),
            "advantage_abs_mean": float(np.abs(adv_np).mean()),
            "grad_norm": grad_norm,
        }
    )
    if reference_policy is not None:
        diag["kl_to_reference"] = (
            float(kl_term.item()) if kl_term is not None else kl_to_reference(policy, reference_policy, obs)
        )
    return diag
