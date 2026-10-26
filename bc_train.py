"""Behaviour-cloning trainer (roadmap Section 7, Month 1: "Train a behaviour-cloning
policy on 3 to 5 tasks ... to a partial success rate of around 30 to 60%"). Trains one
GaussianMLPPolicy per task on noisy-expert data from scripts/collect_bc_data.py, by exact
negative log-likelihood under the policy's own tanh-squashed Gaussian (reuses
policies.GaussianMLPPolicy.log_prob, so the resulting checkpoint drops straight into
algos/*.py as a `reference_policy` for the KL-to-reference diagnostic).

Usage:
    python bc_train.py --data data/bc_demos/pick-place-v3.npz --out checkpoints/bc_pick-place-v3.pt
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from policies import GaussianMLPPolicy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--hidden-sizes", type=int, nargs="+", default=[256, 256],
        help="Policy MLP hidden layer sizes -- must match train.py's policy.hidden_sizes "
        "for any RL run that warm-starts from this checkpoint (state_dict shapes must agree).",
    )
    args = ap.parse_args()

    device = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")

    d = np.load(args.data)
    obs, actions = d["obs"], d["actions"]
    print(f"loaded {len(obs)} transitions from {args.data} (obs_dim={obs.shape[1]} act_dim={actions.shape[1]})")
    print(f"policy hidden_sizes={args.hidden_sizes}")

    policy = GaussianMLPPolicy(obs.shape[1], actions.shape[1], hidden_sizes=args.hidden_sizes).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)

    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
    act_t = torch.as_tensor(actions, dtype=torch.float32, device=device).clamp(-1 + 1e-6, 1 - 1e-6)
    n = len(obs_t)

    for epoch in range(args.epochs):
        perm = torch.randperm(n)
        total_loss = 0.0
        for start in range(0, n, args.batch_size):
            idx = perm[start : start + args.batch_size]
            loss = -policy.log_prob(obs_t[idx], act_t[idx]).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(idx)
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            print(f"  epoch {epoch:03d} nll={total_loss / n:.4f}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save(policy.state_dict(), args.out)
    print(f"saved BC policy -> {args.out}")


if __name__ == "__main__":
    main()
