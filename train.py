"""Config-driven training entrypoint (roadmap Section 9: "train.py: config-driven (Hydra
or YAML), logging and checkpointing").

Usage:
    python train.py                                  # defaults (grpo on Hopper-v5)
    python train.py algo=sm_grpo sm_grpo.k=8          # override via Hydra CLI syntax
    python train.py algo=ppo training.num_iterations=50
"""
from __future__ import annotations

import os
import time

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from algos import collect_groups, grpo, ppo, rloo, sm_grpo, sm_grpo_memory, tgrpo
from advantages.replay_buffer import StateReturnBuffer
from envs import EnvSpec, make_env
from policies import GaussianMLPPolicy, ValueMLP


def resolve_device(requested: str) -> str:
    if requested == "mps" and torch.backends.mps.is_available():
        return "mps"
    if requested == "cuda" and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def build_env_spec(cfg: DictConfig) -> EnvSpec:
    return EnvSpec(
        env_id=cfg.env.env_id,
        backend=cfg.env.backend,
        max_episode_steps=cfg.env.max_episode_steps,
    )


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = resolve_device(cfg.device)
    print(f"[train] using device={device}")

    spec = build_env_spec(cfg)
    probe_env = make_env(spec, seed=cfg.seed)
    obs_dim = int(np.prod(probe_env.observation_space.shape))
    act_dim = int(np.prod(probe_env.action_space.shape))
    probe_env.close()
    print(f"[train] env={spec.env_id} obs_dim={obs_dim} act_dim={act_dim}")

    policy = GaussianMLPPolicy(
        obs_dim, act_dim, hidden_sizes=cfg.policy.hidden_sizes, log_std_init=cfg.policy.log_std_init
    ).to(device)
    if cfg.policy.init_checkpoint:
        policy.load_state_dict(torch.load(cfg.policy.init_checkpoint, map_location=device))
        print(f"[train] warm-started policy from {cfg.policy.init_checkpoint}")
    policy_optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.training.lr)

    value_fn = value_optimizer = None
    if cfg.algo == "ppo":
        value_fn = ValueMLP(obs_dim, hidden_sizes=cfg.policy.hidden_sizes).to(device)
        value_optimizer = torch.optim.Adam(value_fn.parameters(), lr=cfg.ppo.value_lr)

    memory_buffer = None
    if cfg.algo == "sm_grpo_memory":
        memory_buffer = StateReturnBuffer(capacity=cfg.sm_grpo_memory.buffer_capacity)

    reference_policy = None
    if cfg.reference_policy.checkpoint:
        reference_policy = GaussianMLPPolicy(obs_dim, act_dim, hidden_sizes=cfg.policy.hidden_sizes).to(device)
        reference_policy.load_state_dict(torch.load(cfg.reference_policy.checkpoint, map_location=device))
        reference_policy.eval()
        for p in reference_policy.parameters():
            p.requires_grad_(False)

    # PID suffix: two processes launched within the same wall-clock second otherwise get
    # the same dir and silently overwrite each other's TensorBoard event files (bit us
    # twice -- Month 2's 4-way run and the G=64/G=128 group-size runs, see docs/progress.md).
    log_dir = os.path.join(
        cfg.training.log_dir, cfg.algo, f"{time.strftime('%Y%m%d-%H%M%S')}-pid{os.getpid()}"
    )
    os.makedirs(log_dir, exist_ok=True)
    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(log_dir)
    except Exception as e:  # pragma: no cover
        print(f"[train] tensorboard logging disabled: {e}")

    total_env_steps = 0
    for it in range(cfg.training.num_iterations):
        t0 = time.time()
        groups = collect_groups(
            spec,
            policy,
            num_groups=cfg.training.num_groups_per_iteration,
            group_size=cfg.training.group_size,
            base_seed=cfg.seed * 100_000 + it * cfg.training.num_groups_per_iteration,
            max_steps=cfg.training.max_steps_per_episode,
            device=device,
        )
        total_env_steps += sum(len(t.rewards) for g in groups for t in g.trajectories)

        if cfg.algo == "grpo":
            diag = grpo.update(
                policy, policy_optimizer, groups, reference_policy=reference_policy, device=device,
                robust_normalize=cfg.grpo.robust_normalize, sigma_min=cfg.grpo.sigma_min,
                grad_clip=cfg.training.grad_clip,
                match_advantage_scale=cfg.training.match_advantage_scale,
            )
        elif cfg.algo == "rloo":
            diag = rloo.update(
                policy, policy_optimizer, groups, reference_policy=reference_policy, device=device,
                grad_clip=cfg.training.grad_clip,
                match_advantage_scale=cfg.training.match_advantage_scale,
            )
        elif cfg.algo == "tgrpo":
            diag = tgrpo.update(
                policy, policy_optimizer, groups, reference_policy=reference_policy, device=device,
                alpha=cfg.tgrpo.alpha, gamma=cfg.training.gamma, grad_clip=cfg.training.grad_clip,
                match_advantage_scale=cfg.training.match_advantage_scale,
            )
        elif cfg.algo == "sm_grpo":
            diag = sm_grpo.update(
                policy, policy_optimizer, groups, reference_policy=reference_policy, device=device,
                k=cfg.sm_grpo.k, time_window=cfg.sm_grpo.time_window,
                robust_normalize=cfg.sm_grpo.robust_normalize, sigma_min=cfg.sm_grpo.sigma_min,
                gamma=cfg.training.gamma, grad_clip=cfg.training.grad_clip,
                match_advantage_scale=cfg.training.match_advantage_scale,
            )
        elif cfg.algo == "sm_grpo_memory":
            diag = sm_grpo_memory.update(
                policy, policy_optimizer, groups, buffer=memory_buffer, iteration=it,
                reference_policy=reference_policy, device=device,
                k=cfg.sm_grpo_memory.k, time_window=cfg.sm_grpo_memory.time_window,
                memory_max_age=cfg.sm_grpo_memory.memory_max_age,
                robust_normalize=cfg.sm_grpo_memory.robust_normalize,
                sigma_min=cfg.sm_grpo_memory.sigma_min,
                gamma=cfg.training.gamma, grad_clip=cfg.training.grad_clip,
                match_advantage_scale=cfg.training.match_advantage_scale,
            )
        elif cfg.algo == "ppo":
            diag = ppo.update(
                policy, value_fn, policy_optimizer, value_optimizer, groups,
                reference_policy=reference_policy, device=device, gamma=cfg.training.gamma,
                lam=cfg.ppo.lam, clip_ratio=cfg.ppo.clip_ratio, epochs=cfg.ppo.epochs,
                minibatch_size=cfg.ppo.minibatch_size, value_coef=cfg.ppo.value_coef,
                entropy_coef=cfg.ppo.entropy_coef, grad_clip=cfg.training.grad_clip,
            )
        else:
            raise ValueError(f"unknown algo: {cfg.algo}")

        dt = time.time() - t0
        print(
            f"[iter {it:04d}] env_steps={total_env_steps} "
            f"success_rate={diag.get('success_rate', float('nan')):.3f} "
            f"mean_return={diag.get('mean_return', float('nan')):.2f} "
            f"degenerate_frac={diag.get('degenerate_group_frac', float('nan')):.2f} "
            f"adv_abs_mean={diag.get('advantage_abs_mean', float('nan')):.3f} "
            f"grad_norm={diag.get('grad_norm', float('nan')):.3f} "
            f"({dt:.1f}s)"
        )
        if writer is not None:
            for key, value in diag.items():
                writer.add_scalar(f"train/{key}", value, total_env_steps)

        if (it + 1) % cfg.training.checkpoint_every == 0:
            ckpt_path = os.path.join(log_dir, f"policy_iter{it+1}.pt")
            torch.save(policy.state_dict(), ckpt_path)
            print(f"[train] saved checkpoint: {ckpt_path}")

    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()
