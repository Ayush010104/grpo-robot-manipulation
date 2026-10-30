"""The project's headline result: after fixing the BC bottleneck, GRPO+KL hits the
user's 70% target at its peak checkpoint (before later re-collapsing), and PPO fully
solves the task. docs/progress.md, 2026-10-30.
"""
import matplotlib.pyplot as plt

labels = [
    "BC warm-start\n(old, 20 demos)",
    "GRPO+KL=0.1\n(old BC, iter 220)",
    "BC warm-start\n(new, 150 demos)",
    "GRPO+KL=0.1\n(new BC, peak iter 40)",
    "GRPO+KL=0.1\n(new BC, final iter 150)",
    "PPO\n(new BC, final iter 150)",
]
success = [10.0, 6.0, 56.0, 70.0, 0.0, 100.0]
colors = ["#adb5bd", "#adb5bd", "#f08c00", "#2b8a3e", "#e03131", "#1971c2"]

fig, ax = plt.subplots(figsize=(10, 5.5))
bars = ax.bar(labels, success, color=colors, width=0.62)
for bar, val in zip(bars, success):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 1.5, f"{val:.0f}%", ha="center", fontsize=11, fontweight="bold")

ax.axhline(70, color="black", linestyle=":", linewidth=1)
ax.text(5.5, 71.5, "70% target", fontsize=8, ha="right")

ax.set_ylabel("Deterministic success rate (50 held-out episodes)")
ax.set_title(
    "Fixing the BC bottleneck (20 -> 150 demo episodes) is what closed the gap\n"
    "GRPO+KL hits 70% at its peak checkpoint; PPO fully solves the task (100%)",
    fontsize=11,
)
ax.set_ylim(0, 110)
plt.xticks(rotation=12, ha="right", fontsize=9)
fig.tight_layout()
fig.savefig("analysis/70pct_milestone.png", dpi=150)
print("saved analysis/70pct_milestone.png")
