"""PPO with a critic (roadmap Section 7, Month 2 baseline -- the method RL4VLA found
beats GRPO; this is the stability/sample-efficiency bar every group-relative method here
needs to clear). Uses the same grouped rollout collection as the other algos (for an
equal-env-steps comparison) but critic-free group structure is irrelevant to PPO itself --
it's flattened into one batch with GAE(lambda).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from algos.common import kl_to_reference
from algos.rollout_buffer import RolloutGroup


def compute_gae(
    rewards: np.ndarray, values: np.ndarray, gamma: float = 0.99, lam: float = 0.95
) -> tuple[np.ndarray, np.ndarray]:
    """values has one extra entry (bootstrap value at T, 0 if the episode terminated)."""
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float64)
    last_gae = 0.0
    for t in reversed(range(T)):
        delta = rewards[t] + gamma * values[t + 1] - values[t]
        last_gae = delta + gamma * lam * last_gae
        adv[t] = last_gae
    returns = adv + values[:T]
    return adv, returns


def update(
    policy,
    value_fn,
    policy_optimizer,
    value_optimizer,
    groups: list[RolloutGroup],
    reference_policy=None,
    device: str = "cpu",
    gamma: float = 0.99,
    lam: float = 0.95,
    clip_ratio: float = 0.2,
    epochs: int = 4,
    minibatch_size: int = 256,
    value_coef: float = 0.5,
    entropy_coef: float = 0.0,
    grad_clip: float = 1.0,
) -> dict:
    obs_chunks, act_chunks, adv_chunks, ret_chunks, old_logp_chunks = [], [], [], [], []

    with torch.no_grad():
        for group in groups:
            for traj in group.trajectories:
                obs_t = torch.as_tensor(traj.obs, dtype=torch.float32, device=device)
                values = value_fn(obs_t).cpu().numpy()
                bootstrap = 0.0  # every trajectory here ends at terminal or truncation
                values = np.concatenate([values, [bootstrap]])
                adv, ret = compute_gae(traj.rewards.astype(np.float64), values, gamma, lam)

                obs_chunks.append(traj.obs)
                act_chunks.append(traj.actions)
                adv_chunks.append(adv.astype(np.float32))
                ret_chunks.append(ret.astype(np.float32))
                old_logp_chunks.append(traj.log_probs)

    obs_np = np.concatenate(obs_chunks, axis=0)
    act_np = np.concatenate(act_chunks, axis=0)
    adv_np = np.concatenate(adv_chunks, axis=0)
    ret_np = np.concatenate(ret_chunks, axis=0)
    old_logp_np = np.concatenate(old_logp_chunks, axis=0)
    adv_np = (adv_np - adv_np.mean()) / (adv_np.std() + 1e-8)

    obs = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
    actions = torch.as_tensor(act_np, dtype=torch.float32, device=device)
    advantages = torch.as_tensor(adv_np, dtype=torch.float32, device=device)
    returns = torch.as_tensor(ret_np, dtype=torch.float32, device=device)
    old_log_probs = torch.as_tensor(old_logp_np, dtype=torch.float32, device=device)

    n = obs.shape[0]
    last_policy_loss = last_value_loss = last_grad_norm = 0.0

    for _ in range(epochs):
        perm = torch.randperm(n)
        for start in range(0, n, minibatch_size):
            idx = perm[start : start + minibatch_size]
            mb_obs, mb_act = obs[idx], actions[idx]
            mb_adv, mb_ret, mb_old_logp = advantages[idx], returns[idx], old_log_probs[idx]

            new_log_probs = policy.log_prob(mb_obs, mb_act)
            ratio = torch.exp(new_log_probs - mb_old_logp)
            surr1 = ratio * mb_adv
            surr2 = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * mb_adv
            policy_loss = -torch.min(surr1, surr2).mean()
            entropy_bonus = policy.entropy(mb_obs).mean()
            policy_loss = policy_loss - entropy_coef * entropy_bonus

            value_pred = value_fn(mb_obs)
            value_loss = nn.functional.mse_loss(value_pred, mb_ret)

            policy_optimizer.zero_grad()
            policy_loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(policy.parameters(), grad_clip)
            policy_optimizer.step()

            value_optimizer.zero_grad()
            (value_coef * value_loss).backward()
            nn.utils.clip_grad_norm_(value_fn.parameters(), grad_clip)
            value_optimizer.step()

            last_policy_loss = float(policy_loss.item())
            last_value_loss = float(value_loss.item())
            last_grad_norm = float(grad_norm)

    diag = {
        "mean_return": float(np.mean([g.returns.mean() for g in groups])),
        "success_rate": float(np.mean([g.success_rate for g in groups])),
        "policy_loss": last_policy_loss,
        "value_loss": last_value_loss,
        "advantage_abs_mean": float(np.abs(adv_np).mean()),
        "grad_norm": last_grad_norm,
    }
    if reference_policy is not None:
        with torch.no_grad():
            diag["kl_to_reference"] = kl_to_reference(policy, reference_policy, obs)
    return diag
