"""Render the 560x280 writeup card for the submission from official harness numbers in metrics/PHASE1_DEEP3.md.

Usage:
    python writeup/make_thumbnail.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# (label, ours, earlier pair); Conformity is a 0-1 score shown x100 so all four share one axis.
ROWS = [
    ("Active at MIC <= 16 uM", 79.8, 54.9),
    ("Novelty compliance", 99.4, 73.1),
    ("Conformity (x100)", 80.8, 75.2),
    ("Filtering-1 pass", 20.2, 9.2),
]
OURS, BASE = "#1f6feb", "#9aa0a6"

output_path = Path(__file__).resolve().parent / "thumbnail.png"

figure, axes = plt.subplots(figsize=(5.60, 2.80), dpi=100)
height = 0.36
ys = range(len(ROWS))
for i, (_, ours, base) in zip(ys, ROWS):
    axes.barh(i - height / 2, ours, height=height, color=OURS)
    axes.barh(i + height / 2, base, height=height, color=BASE)
    axes.text(ours + 1.0, i - height / 2, f"{ours:g}", va="center", fontsize=7.5, color=OURS)
    axes.text(base + 1.0, i + height / 2, f"{base:g}", va="center", fontsize=7.5, color="#555555")

axes.set_yticks(list(ys), [r[0] for r in ROWS])
axes.invert_yaxis()
axes.set_xlim(0, 112)
axes.tick_params(labelsize=8, length=0)
axes.set_xticks([])
axes.spines[["top", "right", "bottom"]].set_visible(False)
axes.set_title(
    "ESM-C Flow vs our earlier pair, Phase 1 harness (%)", fontsize=8.5, loc="left", pad=6
)
axes.legend(
    handles=[plt.Rectangle((0, 0), 1, 1, color=OURS), plt.Rectangle((0, 0), 1, 1, color=BASE)],
    labels=["Ours (ESM-C Flow)", "Earlier pair (DiffTabPFN)"],
    fontsize=7.5,
    frameon=False,
    loc="lower right",
)

figure.tight_layout()
figure.savefig(output_path, dpi=100, facecolor="white")
print(f"wrote {output_path}")
