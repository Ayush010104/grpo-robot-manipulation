"""Unified env interface per roadmap Section 9: wrappers for Meta-World+, LIBERO and
Gymnasium MuJoCo, a reset(seed) interface, success flags, optional progress signals, and
group-reset support (same initial state for G rollouts of a GRPO group).

Meta-World and LIBERO are optional dependencies -- imported lazily so Gymnasium-only
environments work even before those installs are verified on this machine.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import gymnasium as gym
import numpy as np


@dataclass
class EnvSpec:
    """Describes how to build one env and how to read success/progress out of it."""

    env_id: str
    backend: str = "gymnasium"  # "gymnasium" | "metaworld" | "libero"
    max_episode_steps: Optional[int] = None
    # Fallback for backends (e.g. plain Gymnasium locomotion) with no native success
    # flag: given (obs, reward, terminated, truncated, info) -> bool.
    success_fn: Optional[Callable] = None
    # Given (obs, info) -> float in [0, 1], a dense "how close to done" signal used only
    # for Idea C (sub-goal partial credit on all-fail groups). Optional.
    progress_fn: Optional[Callable] = None


class UnifiedEnvWrapper(gym.Wrapper):
    """Normalizes info["success"] (bool) and info["progress"] (float) across backends,
    and makes reset(seed=...) the single source of stochasticity so group rollouts can
    share an initial state by sharing a seed.
    """

    def __init__(self, env: gym.Env, spec: EnvSpec):
        super().__init__(env)
        # NOTE: `self.spec` is a gymnasium.Wrapper property (env registration spec) --
        # must not shadow it, hence `env_spec` for our own EnvSpec config object.
        self.env_spec = spec

    def reset(self, *, seed: Optional[int] = None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        info.setdefault("success", False)
        info.setdefault("progress", 0.0)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        if "success" not in info:
            if self.env_spec.success_fn is not None:
                info["success"] = bool(
                    self.env_spec.success_fn(obs, reward, terminated, truncated, info)
                )
            else:
                info["success"] = False
        else:
            info["success"] = bool(info["success"])

        if "progress" not in info:
            info["progress"] = (
                float(self.env_spec.progress_fn(obs, info))
                if self.env_spec.progress_fn is not None
                else float(info["success"])
            )

        return obs, reward, terminated, truncated, info


def _build_gymnasium(spec: EnvSpec) -> gym.Env:
    kwargs = {}
    if spec.max_episode_steps is not None:
        kwargs["max_episode_steps"] = spec.max_episode_steps
    env = gym.make(spec.env_id, **kwargs)
    return env


def _build_metaworld(spec: EnvSpec) -> gym.Env:
    # Lazy import: metaworld is an optional, heavier dependency (see docs/roadmap.md S6).
    import metaworld  # type: ignore

    mt = metaworld.MT1(spec.env_id)
    env = mt.train_classes[spec.env_id]()
    task = mt.train_tasks[0]
    env.set_task(task)
    # Meta-World's raw env is already Gymnasium-API-compatible (reset/step with
    # terminated/truncated) in recent versions; wrap minimally if needed upstream.
    return env


def _build_libero(spec: EnvSpec) -> gym.Env:
    raise NotImplementedError(
        "LIBERO integration is a stretch install (roadmap Section 6) -- wire this up "
        "once `LIBERO` is cloned and importable. See scripts/check_envs.py."
    )


_BUILDERS = {
    "gymnasium": _build_gymnasium,
    "metaworld": _build_metaworld,
    "libero": _build_libero,
}


def make_env(spec: EnvSpec, seed: Optional[int] = None) -> gym.Env:
    """Build one env from a spec and reset it with the given seed."""
    raw_env = _BUILDERS[spec.backend](spec)
    env = UnifiedEnvWrapper(raw_env, spec)
    env.reset(seed=seed)
    return env


def make_group_envs(spec: EnvSpec, group_size: int, group_seed: int) -> list[gym.Env]:
    """Build `group_size` envs that all reset from the *same* initial state (same seed).

    This is the group-reset support called for in roadmap Section 9: GRPO-style methods
    need G rollouts starting from an identical state so that return differences reflect
    the policy's action sampling, not a different starting position.
    """
    return [make_env(spec, seed=group_seed) for _ in range(group_size)]


def rollout_random_policy(spec: EnvSpec, steps: int = 200, seed: int = 0) -> dict:
    """Sanity-check helper: run a uniform-random policy for `steps` steps and report
    basic stats. Used by scripts/check_envs.py.
    """
    env = make_env(spec, seed=seed)
    obs, info = env.reset(seed=seed)
    total_reward = 0.0
    successes = 0
    for _ in range(steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        successes += int(info["success"])
        if terminated or truncated:
            obs, info = env.reset()
    env.close()
    return {"total_reward": total_reward, "successes": successes, "steps": steps}
