"""Root-cause plot: grad_norm (pre-clip, log scale) and success_rate side by side,
across PPO / GRPO / SM-GRPO / MM-GRPO, to show the noise-amplification mechanism
(docs/progress.md, 2026-10-27 root-cause entry).
"""
import re
import matplotlib.pyplot as plt

LINE_RE = re.compile(
    r"\[iter (\d+)\].*?success_rate=([\d.]+).*?degenerate_frac=([\d.nan]+).*?grad_norm=([\d.]+)"
)


def parse(path):
    its, succ, gn = [], [], []
    with open(path) as f:
        for line in f:
            m = LINE_RE.search(line)
            if m:
                it, s, _, g = m.groups()
                its.append(int(it))
                succ.append(float(s))
                gn.append(float(g))
    return its, succ, gn


runs = {
    "PPO (critic, baseline)": ("logs/month2_ppo.log", "#2b8a3e"),
    "GRPO (G=8, critic-free)": ("logs/month3_grpo_scalematched.log", "#e03131"),
    "SM-GRPO (G=8)": ("logs/month3_sm_grpo_scalematched.log", "#f08c00"),
    "MM-GRPO (G=8, novel)": ("logs/month3_sm_grpo_memory.log", "#1971c2"),
    "GRPO (G=128, net512)": ("logs/month3_grpo_g128_net512_round2.log", "#9c36b5"),
}

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for name, (path, color) in runs.items():
    its, succ, gn = parse(path)
    if not its:
        continue
    axes[0].plot(its, succ, label=name, color=color, linewidth=1.8)
    axes[1].plot(its, gn, label=name, color=color, linewidth=1.8)

axes[0].set_title("Success rate")
axes[0].set_xlabel("iteration")
axes[0].set_ylabel("success_rate")
axes[0].legend(fontsize=8)

axes[1].set_title("Gradient norm, pre-clip (log scale)\n— the collapse mechanism")
axes[1].set_xlabel("iteration")
axes[1].set_ylabel("grad_norm (clip_grad_norm_ return value)")
axes[1].set_yscale("log")
axes[1].axhline(1.0, color="gray", linestyle="--", linewidth=0.8, label="clip threshold")
axes[1].legend(fontsize=8)

fig.suptitle(
    "Root cause: once groups go degenerate, pre-clip grad_norm explodes for every\n"
    "group-std-normalized method except MM-GRPO, whose cross-iteration memory keeps it flat",
    fontsize=10,
)
fig.tight_layout()
fig.savefig("analysis/gradnorm_mechanism.png", dpi=150)
print("saved analysis/gradnorm_mechanism.png")
