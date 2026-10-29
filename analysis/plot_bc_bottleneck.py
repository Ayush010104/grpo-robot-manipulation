"""The bottleneck-ablation headline result: demo quantity, not noise, explains the
BC gap. docs/progress.md, 2026-10-29.
"""
import matplotlib.pyplot as plt
import numpy as np

labels = ["20 episodes\nnoise=0.3\n(original setup)", "20 episodes\nnoise=0.1", "150 episodes\nnoise=0.3", "150 episodes\nnoise=0.1"]
success = [0.0, 2.0, 56.0, 56.0]
colors = ["#e03131", "#e8590c", "#2b8a3e", "#2b8a3e"]

fig, ax = plt.subplots(figsize=(8, 5.5))
bars = ax.bar(labels, success, color=colors, width=0.6)
for bar, val in zip(bars, success):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 1.5, f"{val:.0f}%", ha="center", fontsize=12, fontweight="bold")

ax.axhspan(30, 60, color="gray", alpha=0.12, zorder=0)
ax.text(3.45, 45, "roadmap's original\n30-60% BC target", fontsize=8, color="gray", ha="right", va="center")

ax.set_ylabel("Deterministic BC success rate (50 held-out episodes)")
ax.set_title("Demo quantity, not noise, was the bottleneck\n20→150 demo episodes: 0% → 56% BC success, regardless of noise level", fontsize=11)
ax.set_ylim(0, 70)
fig.tight_layout()
fig.savefig("analysis/bc_bottleneck.png", dpi=150)
print("saved analysis/bc_bottleneck.png")
