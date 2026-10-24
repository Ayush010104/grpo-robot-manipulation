"""BC / RL policy for low-dimensional state observations (roadmap Section 9: "Start from
low-dimensional state observations, then move to images once the algorithm works").

A small transformer or diffusion/ACT head (LeRobot) can be swapped in later behind the
same `act`/`log_prob`/`entropy` interface used by algos/*.py.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal


def _mlp(in_dim: int, out_dim: int, hidden_sizes=(256, 256), activation=nn.Tanh):
    layers = []
    last = in_dim
    for h in hidden_sizes:
        layers += [nn.Linear(last, h), activation()]
        last = h
    layers += [nn.Linear(last, out_dim)]
    return nn.Sequential(*layers)


class GaussianMLPPolicy(nn.Module):
    """Tanh-squashed Gaussian policy over a continuous action space."""

    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes=(256, 256), log_std_init=-0.5):
        super().__init__()
        self.mean_net = _mlp(obs_dim, act_dim, hidden_sizes)
        self.log_std = nn.Parameter(torch.full((act_dim,), float(log_std_init)))
        self.act_dim = act_dim

    def distribution(self, obs: torch.Tensor) -> Normal:
        mean = self.mean_net(obs)
        std = self.log_std.exp().expand_as(mean)
        return Normal(mean, std)

    def act(self, obs: torch.Tensor, deterministic: bool = False):
        """Returns (tanh-squashed action, log_prob of the *pre-squash* sample)."""
        dist = self.distribution(obs)
        raw_action = dist.mean if deterministic else dist.rsample()
        log_prob = dist.log_prob(raw_action).sum(-1)
        # tanh squash + the standard log-det-Jacobian correction
        action = torch.tanh(raw_action)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6).sum(-1)
        return action, log_prob

    def log_prob(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """log_prob of an already-squashed action (inverts tanh; used for PPO ratios)."""
        action = action.clamp(-1 + 1e-6, 1 - 1e-6)
        raw_action = torch.atanh(action)
        dist = self.distribution(obs)
        log_prob = dist.log_prob(raw_action).sum(-1)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6).sum(-1)
        return log_prob

    def entropy(self, obs: torch.Tensor) -> torch.Tensor:
        # Entropy of the underlying Gaussian; ignores the tanh correction (standard
        # approximation used by SAC/PPO-continuous implementations for the entropy bonus).
        return self.distribution(obs).entropy().sum(-1)


class ValueMLP(nn.Module):
    """Critic for PPO. GRPO/RLOO/SM-GRPO are critic-free by design and do not use this."""

    def __init__(self, obs_dim: int, hidden_sizes=(256, 256)):
        super().__init__()
        self.net = _mlp(obs_dim, 1, hidden_sizes)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)
