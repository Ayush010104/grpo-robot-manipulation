"""Roadmap Section 9: "eval.py: success rate, generalisation splits and video recording."
Video recording and generalisation splits (held-out object positions/textures) are left
as a follow-up -- this gives the success-rate metric needed to confirm the Month-1 BC
target (30-60% partial success) and, later, Month-2/3 algo comparisons.

Usage:
    python eval.py --checkpoint checkpoints/bc_pick-place-v3.pt --env-id pick-place-v3 \
        --backend metaworld --episodes 50
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from envs.wrappers import EnvSpec, make_env
from policies import GaussianMLPPolicy


@torch.no_grad()
def evaluate(policy, spec: EnvSpec, episodes: int, max_steps: int, seed: int, device: str, deterministic: bool = True):
    successes = 0
    returns = []
    for ep in range(episodes):
        env = make_env(spec, seed=seed + ep)
        obs, info = env.reset(seed=seed + ep)
        ep_return = 0.0
        succeeded = False
        for _ in range(max_steps):
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            action_t, _ = policy.act(obs_t, deterministic=deterministic)
            action = action_t.squeeze(0).cpu().numpy()
            obs, reward, terminated, truncated, info = env.step(action)
            ep_return += reward
            succeeded = succeeded or bool(info["success"])
            if terminated or truncated:
                break
        env.close()
        successes += int(succeeded)
        returns.append(ep_return)
    return {
        "success_rate": successes / episodes,
        "mean_return": float(np.mean(returns)),
        "episodes": episodes,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--env-id", required=True)
    ap.add_argument("--backend", default="metaworld")
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--max-steps", type=int, default=150)
    ap.add_argument("--seed", type=int, default=1000)  # disjoint from BC-data seeds
    ap.add_argument("--device", default=None)
    ap.add_argument(
        "--hidden-sizes", type=int, nargs="+", default=[256, 256],
        help="Must match the checkpoint's own hidden_sizes (bc_train.py --hidden-sizes / "
        "train.py's policy.hidden_sizes) or state_dict loading will fail.",
    )
    args = ap.parse_args()

    device = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    spec = EnvSpec(env_id=args.env_id, backend=args.backend, max_episode_steps=args.max_steps)

    probe = make_env(spec, seed=0)
    obs_dim = int(np.prod(probe.observation_space.shape))
    act_dim = int(np.prod(probe.action_space.shape))
    probe.close()

    policy = GaussianMLPPolicy(obs_dim, act_dim, hidden_sizes=args.hidden_sizes).to(device)
    policy.load_state_dict(torch.load(args.checkpoint, map_location=device))
    policy.eval()

    result = evaluate(policy, spec, args.episodes, args.max_steps, args.seed, device)
    print(f"{args.env_id}: success_rate={result['success_rate']:.2%} mean_return={result['mean_return']:.2f} "
          f"(n={result['episodes']})")


if __name__ == "__main__":
    main()
