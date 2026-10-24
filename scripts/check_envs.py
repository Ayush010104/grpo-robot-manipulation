"""Roadmap Section 6, install step 4: "Verify a random policy and a scripted policy run
in each environment." Run after `pip install -r requirements.txt`:

    python scripts/check_envs.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.wrappers import EnvSpec, rollout_random_policy

GYMNASIUM_SANITY_ENVS = ["Hopper-v5", "Walker2d-v5", "Ant-v5"]
METAWORLD_SANITY_TASKS = ["pick-place-v3", "door-open-v3", "drawer-open-v3", "button-press-v3"]


def check_gymnasium():
    print("== Gymnasium MuJoCo ==")
    for env_id in GYMNASIUM_SANITY_ENVS:
        spec = EnvSpec(env_id=env_id, backend="gymnasium", max_episode_steps=100)
        try:
            result = rollout_random_policy(spec, steps=100, seed=0)
            print(f"  [OK] {env_id}: {result}")
        except Exception as e:
            print(f"  [FAIL] {env_id}: {e}")


def check_metaworld():
    print("== Meta-World+ ==")
    try:
        import metaworld  # noqa: F401
    except ImportError:
        print("  [SKIP] metaworld not installed")
        return
    for task in METAWORLD_SANITY_TASKS:
        spec = EnvSpec(env_id=task, backend="metaworld", max_episode_steps=100)
        try:
            result = rollout_random_policy(spec, steps=100, seed=0)
            print(f"  [OK] {task}: {result}")
        except Exception as e:
            print(f"  [FAIL] {task}: {e}")


def check_metaworld_scripted_policy(task: str = "pick-place-v3"):
    print("== Meta-World+ scripted expert policy ==")
    try:
        import metaworld
        import metaworld.policies as mp
    except ImportError:
        print("  [SKIP] metaworld not installed")
        return
    # naming convention: "pick-place-v3" -> "SawyerPickPlaceV3Policy"
    class_name = "Sawyer" + "".join(p.capitalize() for p in task.replace("-v3", "").split("-")) + "V3Policy"
    policy_cls = getattr(mp, class_name, None)
    if policy_cls is None:
        print(f"  [SKIP] no scripted policy class found for {task} (tried {class_name})")
        return

    mt1 = metaworld.MT1(task)
    env = mt1.train_classes[task]()
    env.set_task(mt1.train_tasks[0])
    expert = policy_cls()

    obs, info = env.reset(seed=0)
    total_reward, success = 0.0, False
    for _ in range(200):
        action = expert.get_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        success = success or bool(info.get("success", 0.0))
        if terminated or truncated:
            break
    env.close()
    print(f"  [OK] {task} scripted policy: total_reward={total_reward:.2f} success={success}")


def check_libero():
    print("== LIBERO ==")
    try:
        import libero  # noqa: F401
        print("  [OK] libero importable (wire up envs/wrappers.py::_build_libero next)")
    except ImportError:
        print("  [SKIP] LIBERO not installed -- see docs/roadmap.md Section 6 for install steps")


if __name__ == "__main__":
    check_gymnasium()
    check_metaworld()
    check_metaworld_scripted_policy()
    check_libero()
    sys.exit(0)
