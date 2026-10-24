"""RLOO: leave-one-out REINFORCE baseline (roadmap Section 3, RIPT-VLA). Critic-free, like
GRPO, but each rollout's baseline excludes its own return -- removes the self-comparison
bias that can make vanilla GRPO's gradient estimator noisier.
"""
from __future__ import annotations

import numpy as np
import torch

from algos.common import clip_grad_and_step, flatten_group_steps, group_diagnostics, kl_to_reference
from algos.rollout_buffer import RolloutGroup
from advantages.group_baselines import is_degenerate_group, leave_one_out_advantages


def update(
    policy,
    optimizer,
    groups: list[RolloutGroup],
    reference_policy=None,
    device: str = "cpu",
    grad_clip: float = 1.0,
) -> dict:
    per_traj_adv = []
    degenerate_mask = []
    for group in groups:
        returns = group.returns
        degenerate_mask.append(is_degenerate_group(returns))
        adv = leave_one_out_advantages(returns)
        per_traj_adv.append(
            [np.full(len(t.rewards), a, dtype=np.float32) for t, a in zip(group.trajectories, adv)]
        )

    obs_np, act_np, adv_np = flatten_group_steps(groups, per_traj_adv)
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
