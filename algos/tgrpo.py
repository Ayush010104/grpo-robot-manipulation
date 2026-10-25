"""TGRPO-style baseline (roadmap Section 3): fuses a step-level advantage (return-to-go
vs. the group's mean return-to-go at the *same timestep*, no state-matching -- that's
SM-GRPO's job) with the vanilla trajectory-level GRPO advantage.
"""
from __future__ import annotations

import numpy as np
import torch

from algos.common import clip_grad_and_step, group_diagnostics, kl_to_reference
from algos.rollout_buffer import RolloutGroup
from advantages.group_baselines import (
    fuse_trajectory_and_step,
    is_degenerate_group_by_success,
    trajectory_advantages,
)


def _same_timestep_step_advantage(group_rtg: list[np.ndarray]) -> list[np.ndarray]:
    """Per-step advantage vs. the mean return-to-go of all rollouts still alive at that
    timestep (same-index alignment, not state-matched -- the simple baseline TGRPO uses
    before SM-GRPO's state matching).
    """
    max_t = max(len(r) for r in group_rtg)
    sums = np.zeros(max_t)
    counts = np.zeros(max_t)
    for r in group_rtg:
        sums[: len(r)] += r
        counts[: len(r)] += 1
    means = sums / np.maximum(counts, 1)

    out = []
    for r in group_rtg:
        out.append(r - means[: len(r)])
    return out


def update(
    policy,
    optimizer,
    groups: list[RolloutGroup],
    reference_policy=None,
    device: str = "cpu",
    alpha: float = 0.5,  # trajectory/step fusion weight
    gamma: float = 1.0,
    grad_clip: float = 1.0,
) -> dict:
    obs_chunks, act_chunks, adv_chunks = [], [], []
    degenerate_mask = []

    for group in groups:
        returns = group.returns
        successes = np.array([t.succeeded for t in group.trajectories])
        degenerate_mask.append(is_degenerate_group_by_success(successes))
        traj_adv = trajectory_advantages(returns)

        group_rtg = [t.returns_to_go(gamma=gamma) for t in group.trajectories]
        step_adv = _same_timestep_step_advantage(group_rtg)

        for traj, t_adv, s_adv in zip(group.trajectories, traj_adv, step_adv):
            fused = fuse_trajectory_and_step(t_adv, s_adv, alpha=alpha)
            obs_chunks.append(traj.obs)
            act_chunks.append(traj.actions)
            adv_chunks.append(fused.astype(np.float32))

    obs_np = np.concatenate(obs_chunks, axis=0)
    act_np = np.concatenate(act_chunks, axis=0)
    adv_np = np.concatenate(adv_chunks, axis=0)

    obs = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
    actions = torch.as_tensor(act_np, dtype=torch.float32, device=device)
    advantages = torch.as_tensor(adv_np, dtype=torch.float32, device=device)

    log_probs = policy.log_prob(obs, actions)
    loss = -(log_probs * advantages).mean()
    loss.backward()
    grad_norm = clip_grad_and_step(policy, optimizer, max_norm=grad_clip)

    diag = group_diagnostics(groups, degenerate_mask)
    diag.update(
        {
            "loss": float(loss.item()),
            "advantage_abs_mean": float(np.abs(adv_np).mean()),
            "grad_norm": grad_norm,
        }
    )
    if reference_policy is not None:
        diag["kl_to_reference"] = kl_to_reference(policy, reference_policy, obs)
    return diag
