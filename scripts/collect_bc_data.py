"""Month 1 (roadmap Section 7): generate behaviour-cloning data for Meta-World+ tasks
using the library's scripted expert policies, with action noise so the resulting BC
policy lands at the "30 to 60% partial success" target the roadmap asks for, leaving
room for RL to help.

Usage:
    python scripts/collect_bc_data.py --tasks pick-place-v3 door-open-v3 drawer-open-v3
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from envs.wrappers import EnvSpec, make_env

TASK_TO_POLICY = {
    "pick-place-v3": "SawyerPickPlaceV3Policy",
    "door-open-v3": "SawyerDoorOpenV3Policy",
    "drawer-open-v3": "SawyerDrawerOpenV3Policy",
    "button-press-v3": "SawyerButtonPressV3Policy",
}


def collect_task(task: str, episodes: int, noise_std: float, max_steps: int, seed: int):
    import metaworld.policies as mp

    spec = EnvSpec(env_id=task, backend="metaworld", max_episode_steps=max_steps)
    policy_cls = getattr(mp, TASK_TO_POLICY[task])
    expert = policy_cls()

    obs_list, act_list = [], []
    successes = 0
    rng = np.random.default_rng(seed)

    for ep in range(episodes):
        env = make_env(spec, seed=seed + ep)
        obs, info = env.reset(seed=seed + ep)
        succeeded = False
        for _ in range(max_steps):
            action = expert.get_action(obs)
            noisy = np.clip(action + rng.normal(0, noise_std, size=action.shape), -1.0, 1.0)
            obs_list.append(obs)
            act_list.append(noisy)
            obs, reward, terminated, truncated, info = env.step(noisy)
            succeeded = succeeded or bool(info["success"])
            if terminated or truncated:
                break
        successes += int(succeeded)
        env.close()

    print(f"[{task}] noisy-expert success: {successes}/{episodes} episodes, {len(obs_list)} transitions")
    return np.array(obs_list, dtype=np.float32), np.array(act_list, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=list(TASK_TO_POLICY.keys())[:3])
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--noise-std", type=float, default=0.3)
    ap.add_argument("--max-steps", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="data/bc_demos")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    for task in args.tasks:
        obs, act = collect_task(task, args.episodes, args.noise_std, args.max_steps, args.seed)
        out_path = os.path.join(args.out_dir, f"{task}.npz")
        np.savez(out_path, obs=obs, actions=act)
        print(f"  saved -> {out_path}")


if __name__ == "__main__":
    main()
