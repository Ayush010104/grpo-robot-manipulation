"""SM-GRPO: State-Matched Group Baselines (roadmap Section 5, Idea A -- the recommended
main contribution). Per-step, critic-free advantage: return-to-go minus the mean
return-to-go of the k nearest neighbours (in state-embedding space, within a time window)
across the *other* rollouts in the group.

Month 3 ablation knobs (roadmap Section 7): k, time_window, embedding_fn, robust
normalisation (Idea B), and -- orthogonally, at rollout-collection time -- common random
numbers (Idea D; note that `collect_group`'s shared seed only fixes the *initial* state
today, not true step-wise common random numbers -- see docs/roadmap.md Section 5, Idea D).
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import torch

from algos.common import clip_grad_and_step, group_diagnostics, kl_to_reference
from algos.rollout_buffer import RolloutGroup
from advantages.group_baselines import (
    is_degenerate_group_by_success,
    robust_scale,
    state_matched_baselines,
)


def _default_embedding(obs: np.ndarray) -> np.ndarray:
    """Identity embedding for low-dimensional state observations (roadmap Section 9:
    start low-dim, move to a frozen visual encoder for images later).
    """
    return obs


def update(
    policy,
    optimizer,
    groups: list[RolloutGroup],
    reference_policy=None,
    device: str = "cpu",
    k: int = 4,
    time_window: int = 5,
    embedding_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    robust_normalize: bool = False,  # Idea B combined with Idea A
    sigma_min: float = 1e-2,
    gamma: float = 1.0,
    grad_clip: float = 1.0,
) -> dict:
    embedding_fn = embedding_fn or _default_embedding

    obs_chunks, act_chunks, adv_chunks = [], [], []
    degenerate_mask = []

    for group in groups:
        successes = np.array([t.succeeded for t in group.trajectories])
        degenerate_mask.append(is_degenerate_group_by_success(successes))

        group_states = [embedding_fn(t.obs) for t in group.trajectories]
        group_rtg = [t.returns_to_go(gamma=gamma) for t in group.trajectories]

        per_step_adv = state_matched_baselines(
            group_states, group_rtg, k=k, time_window=time_window
        )

        if robust_normalize:
            scale = robust_scale(group.returns, sigma_min=sigma_min)
            per_step_adv = [a / scale for a in per_step_adv]

        for traj, adv in zip(group.trajectories, per_step_adv):
            obs_chunks.append(traj.obs)
            act_chunks.append(traj.actions)
            adv_chunks.append(adv.astype(np.float32))

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
