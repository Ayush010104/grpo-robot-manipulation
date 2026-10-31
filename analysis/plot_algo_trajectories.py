"""Full trajectory comparison across algorithms, all post-bottleneck-fix (new, 56%-
success BC checkpoint) except where noted, for a fair current-state read.
docs/progress.md, 2026-10-30/31.
"""
import re
import matplotlib.pyplot as plt

LINE_RE = re.compile(
    r"\[iter (\d+)\].*?success_rate=([\d.]+).*?grad_norm=([\d.]+)"
)


def parse(path):
    its, succ, gn = [], [], []
    with open(path) as f:
        for line in f:
            m = LINE_RE.search(line)
            if m:
                it, s, g = m.groups()
                its.append(int(it))
                succ.append(float(s))
                gn.append(float(g))
    return its, succ, gn


runs = {
    "PPO (new BC)": ("logs/month3_ppo_newbc.log", "#1971c2", 2.2),
    "GRPO+KL=0.2 (new BC)": ("logs/month3_grpo_newbc_kl02.log", "#2b8a3e", 1.8),
    "GRPO+KL=0.3 (new BC)": ("logs/month3_grpo_newbc_kl03.log", "#37b24d", 1.4),
    "GRPO+KL=0.5 (new BC)": ("logs/month3_grpo_newbc_kl05.log", "#69db7c", 1.4),
    "GRPO+KL=0.1 (new BC) -- collapses": ("logs/month3_grpo_kl01_newbc.log", "#e03131", 1.8),
    "GRPO+KL=0.1 (OLD BC, for contrast)": ("logs/month3_grpo_kl01_long.log", "#adb5bd", 1.2),
}

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
for name, (path, color, lw) in runs.items():
    its, succ, gn = parse(path)
    if not its:
        continue
    axes[0].plot(its, succ, label=name, color=color, linewidth=lw)
    axes[1].plot(its, gn, label=name, color=color, linewidth=lw)

axes[0].axhline(0.70, color="black", linestyle=":", linewidth=0.8)
axes[0].text(2, 0.715, "70% target", fontsize=8)
axes[0].set_title("Success rate (training-time, stochastic rollouts)")
axes[0].set_xlabel("iteration")
axes[0].set_ylabel("success_rate")
axes[0].legend(fontsize=7.5, loc="upper left")

axes[1].set_title("Gradient norm, pre-clip (log scale)")
axes[1].set_xlabel("iteration")
axes[1].set_yscale("log")
axes[1].axhline(1.0, color="gray", linestyle="--", linewidth=0.7)
axes[1].legend(fontsize=7.5)

fig.suptitle(
    "Where things stand: PPO solves the task cleanly; GRPO+KL=0.1 (tuned for the OLD,\n"
    "weak reference) still collapses even from the new BC checkpoint; KL >= 0.2 holds without collapsing",
    fontsize=10,
)
fig.tight_layout()
fig.savefig("analysis/algo_trajectories.png", dpi=150)
print("saved analysis/algo_trajectories.png")
