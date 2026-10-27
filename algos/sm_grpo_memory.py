"""MM-GRPO: Memory-Matched GRPO -- a novel extension of SM-GRPO (Idea A), proposed
2026-10-26. See advantages/replay_buffer.py for the full motivation and the staleness
caveat.

Identical to sm_grpo.py except the k-NN baseline draws extra candidates from a
persistent `StateReturnBuffer` accumulated across iterations, not just the current
group -- the hope is a less noisy, less frequently degenerate baseline without a
differentiable critic. Unlike algos/*.py's other `update()` functions, this one is NOT
stateless: it takes (and mutates) a buffer object and the current iteration index, since
the whole point is persistence across calls. The caller (train.py) owns the buffer.
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import torch

from algos.common import (
    clip_grad_and_step,
    group_diagnostics,
    kl_penalty_term,
    kl_to_reference,
    normalize_batch_advantages,
)
from algos.rollout_buffer import RolloutGroup
from advantages.group_baselines import (
    is_degenerate_group_by_success,
    robust_scale,
    state_matched_baselines,
)
from advantages.replay_buffer import StateReturnBuffer


def _default_embedding(obs: np.ndarray) -> np.ndarray:
    return obs


def update(
    policy,
    optimizer,
    groups: list[RolloutGroup],
    buffer: StateReturnBuffer,
    iteration: int,
    reference_policy=None,
    device: str = "cpu",
    k: int = 4,
    time_window: int = 5,
    memory_max_age: Optional[int] = None,
    embedding_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    robust_normalize: bool = True,  # on by default -- see sm_grpo.py's history of why
    sigma_min: float = 1e-2,
    gamma: float = 1.0,
    grad_clip: float = 1.0,
    match_advantage_scale: bool = False,
    kl_coef: float = 0.0,
) -> dict:
    embedding_fn = embedding_fn or _default_embedding

    extra_states, extra_returns = buffer.sample_candidates(iteration, max_age=memory_max_age)

    obs_chunks, act_chunks, adv_chunks = [], [], []
    degenerate_mask = []
    new_buffer_states, new_buffer_returns = [], []

    for group in groups:
        successes = np.array([t.succeeded for t in group.trajectories])
        degenerate_mask.append(is_degenerate_group_by_success(successes))

        group_states = [embedding_fn(t.obs) for t in group.trajectories]
        group_rtg = [t.returns_to_go(gamma=gamma) for t in group.trajectories]

        per_step_adv = state_matched_baselines(
            group_states, group_rtg, k=k, time_window=time_window,
            extra_states=extra_states, extra_returns=extra_returns,
        )

        if robust_normalize:
            scale = robust_scale(group.returns, sigma_min=sigma_min)
            per_step_adv = [a / scale for a in per_step_adv]

        for traj, adv, states, rtg in zip(group.trajectories, per_step_adv, group_states, group_rtg):
            obs_chunks.append(traj.obs)
            act_chunks.append(traj.actions)
            adv_chunks.append(adv.astype(np.float32))
            new_buffer_states.append(states)
            new_buffer_returns.append(rtg)

    obs_np = np.concatenate(obs_chunks, axis=0)
    act_np = np.concatenate(act_chunks, axis=0)
    adv_np = np.concatenate(adv_chunks, axis=0)
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

    # Add this iteration's (state, return-to-go) pairs to the buffer for FUTURE
    # iterations only -- this iteration's own baseline above used the pre-update buffer.
    for states, rtg in zip(new_buffer_states, new_buffer_returns):
        buffer.add(states, rtg, iteration)

    diag = group_diagnostics(groups, degenerate_mask)
    diag.update(
        {
            "loss": float(loss.item()),
            "pg_loss": float(pg_loss.item()),
            "advantage_abs_mean": float(np.abs(adv_np).mean()),
            "grad_norm": grad_norm,
            "buffer_size": float(len(buffer)),
        }
    )
    if reference_policy is not None:
        diag["kl_to_reference"] = (
            float(kl_term.item()) if kl_term is not None else kl_to_reference(policy, reference_policy, obs)
        )
    return diag
