"""Persistent cross-iteration state/return-to-go memory for MM-GRPO (Memory-Matched
GRPO) -- a novel extension of SM-GRPO (Idea A) proposed 2026-10-26.

Motivation (see docs/progress.md and the chat diagnosis of why GRPO fails here): PPO's
critic gives two things GRPO lacks -- (1) a baseline that generalizes across the whole
training run, not just this iteration's rollouts, and (2) a baseline that doesn't go
degenerate just because this one group of 8-20 rollouts happened to agree. SM-GRPO's
k-NN state-matching already measurably helps (it delayed GRPO's collapse ~3x in the
scale-matched comparison) by comparing *states*, not whole trajectories -- but its
candidate pool is still only the current group's siblings, discarded every iteration.

MM-GRPO keeps SM-GRPO's exact mechanism (k-NN over a state embedding, no differentiable
critic, no backprop through a value network) but draws k-NN candidates from a persistent
buffer of (state, return-to-go) pairs accumulated across iterations, in addition to the
current group. This is a non-parametric, memory-cheap approximation of what a critic's
cross-iteration generalization gives -- a flat/ball-tree index over float vectors is
orders of magnitude cheaper than a value network at VLA scale, so most of GRPO's memory
advantage over PPO survives.

Known risk, flagged rather than hidden: buffer entries from early iterations reflect an
older, usually worse policy, so their return-to-go may not be what the *current* policy
would get from a similar state (the standard off-policy-baseline staleness problem any
critic also has to manage, usually via a moving target network or importance weighting --
MM-GRPO's current mitigation is just a recency window, `max_age`; a policy-similarity-
weighted variant is a natural follow-up ablation, not implemented yet).
"""
from __future__ import annotations

from typing import Optional

import numpy as np


class StateReturnBuffer:
    """FIFO buffer of (state, return_to_go, iteration) triples."""

    def __init__(self, capacity: int = 50_000):
        self.capacity = capacity
        self._states: list[np.ndarray] = []
        self._returns: list[float] = []
        self._iters: list[int] = []

    def add(self, states: np.ndarray, returns: np.ndarray, iteration: int) -> None:
        for s, r in zip(states, returns):
            self._states.append(np.asarray(s, dtype=np.float32))
            self._returns.append(float(r))
            self._iters.append(iteration)
        overflow = len(self._states) - self.capacity
        if overflow > 0:
            del self._states[:overflow]
            del self._returns[:overflow]
            del self._iters[:overflow]

    def __len__(self) -> int:
        return len(self._states)

    def sample_candidates(
        self, current_iteration: int, max_age: Optional[int] = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Returns (states (N, D), returns (N,)) -- every buffered entry within
        `max_age` iterations of `current_iteration` (None = the whole buffer).
        """
        if not self._states:
            return np.zeros((0, 0), dtype=np.float32), np.zeros((0,), dtype=np.float32)
        if max_age is None:
            idx = range(len(self._states))
        else:
            idx = [i for i, it in enumerate(self._iters) if current_iteration - it <= max_age]
        if not idx:
            return np.zeros((0, 0), dtype=np.float32), np.zeros((0,), dtype=np.float32)
        states = np.stack([self._states[i] for i in idx]).astype(np.float32)
        returns = np.array([self._returns[i] for i in idx], dtype=np.float32)
        return states, returns
