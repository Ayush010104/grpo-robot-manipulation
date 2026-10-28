"""Headline-result plot: vanilla GRPO with and without a real KL-to-reference penalty.
docs/progress.md, 2026-10-27d.
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
    "GRPO, no KL penalty (collapses, stays dead)": ("logs/month3_grpo_scalematched.log", "#e03131"),
    "GRPO, kl_coef=0.01 (too weak)": ("logs/month3_grpo_klpen_001.log", "#f08c00"),
    "GRPO, kl_coef=0.1 (sustains ~5.3%)": ("logs/month3_grpo_klpen_01.log", "#2b8a3e"),
    "GRPO, kl_coef=1.0 (over-penalized)": ("logs/month3_grpo_klpen_10.log", "#1971c2"),
}

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for name, (path, color) in runs.items():
    its, succ, gn = parse(path)
    if not its:
        continue
    axes[0].plot(its, succ, label=name, color=color, linewidth=1.6)
    axes[1].plot(its, gn, label=name, color=color, linewidth=1.6)

axes[0].set_title("Success rate")
axes[0].set_xlabel("iteration")
axes[0].set_ylabel("success_rate")
axes[0].legend(fontsize=7.5)

axes[1].set_title("Gradient norm, pre-clip (log scale)")
axes[1].set_xlabel("iteration")
axes[1].set_yscale("log")
axes[1].axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
axes[1].legend(fontsize=7.5)

fig.suptitle(
    "A real KL(policy || BC-reference) penalty -- logged since Month 2 but never added\n"
    "to any loss until 2026-10-27c -- alone fixes vanilla GRPO's collapse. No memory buffer needed.",
    fontsize=10,
)
fig.tight_layout()
fig.savefig("analysis/kl_penalty_fix.png", dpi=150)
print("saved analysis/kl_penalty_fix.png")
