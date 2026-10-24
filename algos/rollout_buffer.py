"""Shared rollout buffer used by ppo.py, grpo.py, rloo.py, tgrpo.py, sm_grpo.py
(roadmap Section 9: "algos/: ... sharing a common rollout buffer").

Collects *groups* of G rollouts that share one initial state (group-reset, Section 4.2
"reset diversity" / Section 9 "same initial state for G rollouts"), since every
group-relative algo here needs that; PPO just flattens groups back into one batch.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from envs.wrappers import EnvSpec, make_group_envs
from advantages.group_baselines import compute_returns_to_go


@dataclass
class Trajectory:
    obs: np.ndarray  # (T, obs_dim)
    actions: np.ndarray  # (T, act_dim), squashed actions actually applied
    rewards: np.ndarray  # (T,)
    log_probs: np.ndarray  # (T,) behaviour-policy log-prob at collection time
    successes: np.ndarray  # (T,) bool, per-step info["success"]
    progress: np.ndarray  # (T,) float, per-step info["progress"]
    group_id: int = 0

    @property
    def trajectory_return(self) -> float:
        return float(self.rewards.sum())

    @property
    def succeeded(self) -> bool:
        return bool(self.successes.any())

    def returns_to_go(self, gamma: float = 1.0) -> np.ndarray:
        return compute_returns_to_go(self.rewards, gamma=gamma)


@dataclass
class RolloutGroup:
    group_id: int
    seed: int
    trajectories: list[Trajectory] = field(default_factory=list)

    @property
    def returns(self) -> np.ndarray:
        return np.array([t.trajectory_return for t in self.trajectories])

    @property
    def success_rate(self) -> float:
        return float(np.mean([t.succeeded for t in self.trajectories]))


@torch.no_grad()
def collect_group(
    spec: EnvSpec,
    policy,
    group_size: int,
    seed: int,
    max_steps: int,
    device: str = "cpu",
    group_id: int = 0,
) -> RolloutGroup:
    """Roll out `group_size` trajectories from the *same* initial state (same seed)."""
    envs = make_group_envs(spec, group_size, group_seed=seed)
    trajectories = []

    for env in envs:
        obs, info = env.reset(seed=seed)
        obs_buf, act_buf, rew_buf, logp_buf, succ_buf, prog_buf = [], [], [], [], [], []

        for _ in range(max_steps):
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            action_t, log_prob_t = policy.act(obs_t)
            action = action_t.squeeze(0).cpu().numpy()

            obs_buf.append(obs)
            act_buf.append(action)
            logp_buf.append(float(log_prob_t.item()))

            obs, reward, terminated, truncated, info = env.step(action)
            rew_buf.append(float(reward))
            succ_buf.append(bool(info["success"]))
            prog_buf.append(float(info["progress"]))

            if terminated or truncated:
                break

        trajectories.append(
            Trajectory(
                obs=np.array(obs_buf, dtype=np.float32),
                actions=np.array(act_buf, dtype=np.float32),
                rewards=np.array(rew_buf, dtype=np.float32),
                log_probs=np.array(logp_buf, dtype=np.float32),
                successes=np.array(succ_buf, dtype=bool),
                progress=np.array(prog_buf, dtype=np.float32),
                group_id=group_id,
            )
        )
        env.close()

    return RolloutGroup(group_id=group_id, seed=seed, trajectories=trajectories)


def collect_groups(
    spec: EnvSpec,
    policy,
    num_groups: int,
    group_size: int,
    base_seed: int,
    max_steps: int,
    device: str = "cpu",
) -> list[RolloutGroup]:
    return [
        collect_group(
            spec,
            policy,
            group_size=group_size,
            seed=base_seed + g,
            max_steps=max_steps,
            device=device,
            group_id=g,
        )
        for g in range(num_groups)
    ]
