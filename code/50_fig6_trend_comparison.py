# Code Developer: Somnath Luitel (somnath.luitel416@topper.wku.edu)
"""Step 50: manuscript Figure 5 (trend comparison) for the ESS submission.

Redraws the October 2026 trend-comparison figure with the same values and
layout, in the Arial style of the other figures, with "Annual total" for PRCPTOT and the paired
shift in percentage points per decade. Values are from the manuscript's Table 2 (two-way
bootstrap intervals), Table S1 (alternative tests) and Table S3 (minimum eligible years).
Output: figures/ess_labels/Figure_6_Trend_Comparison.png/.pdf
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

OUT = Path(__file__).resolve().parents[1] / "figures" / "ess_labels"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": ["Arial", "DejaVu Sans"], "font.size": 8.5, "pdf.fonttype": 42,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": .65, "xtick.major.width": .6, "ytick.major.width": .6})
BLUE, ORANGE, GREY, INK = "#0072B2", "#D55E00", "#85898E", "#202B35"

fig, axs = plt.subplots(2, 2, figsize=(6.3, 5.35))
fig.subplots_adjust(left=.10, right=.985, bottom=.12, top=.88, wspace=.40, hspace=.55)
labels = ["Rx1day", "Rx5day", "Annual\ntotal", "R95pTOT"]
colors = [GREY, ORANGE, BLUE]
handles = [Patch(color=c, label=t) for c, t in zip(colors, ["Eligible years only", "All years, uncorrected", "All years, gap-corrected"])]
fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.51, .995), ncol=3, frameon=False,
           fontsize=7.2, handlelength=1.1, columnspacing=1.0)


def clean(ax, letter, title):
    ax.set_title(title, loc="left", fontsize=8.6, pad=10)
    ax.text(-.18, 1.08, letter, transform=ax.transAxes, weight="bold", fontsize=11)
    ax.tick_params(labelsize=7.4)
    ax.set_axisbelow(True); ax.grid(axis="y", color="#E1E5E8", lw=.6)


ax = axs[0, 0]; x = np.arange(4); w = .23
for k, vals in enumerate([[7, 4, 9, 11], [13, 5, 41, 22], [10, 5, 25, 17]]):
    bars = ax.bar(x + (k - 1) * w, vals, w, color=colors[k])
    ax.bar_label(bars, padding=2, fontsize=6.9)
ax.set(xticks=x, xticklabels=labels, ylim=(0, 48))
ax.set_ylabel("Stations with FDR-significant decline", fontsize=7.2)
clean(ax, "a", "Declining-station counts")

ax = axs[0, 1]
U = np.array([[-.60, -1.70, .22], [-.71, -1.52, .11], [-1.06, -1.96, -.33], [-1.79, -5.44, .48]])
C = np.array([[-.42, -1.27, .68], [-.41, -.98, .50], [-.31, -.94, .27], [-.90, -4.13, 1.21]])
y = np.arange(4)[::-1]
for data, shift, c in [(U, -.12, ORANGE), (C, .12, BLUE)]:
    ax.errorbar(data[:, 0], y + shift, xerr=np.vstack([data[:, 0] - data[:, 1], data[:, 2] - data[:, 0]]),
                fmt="o", color=c, markersize=3.2, elinewidth=1.0, capsize=0)
ax.axvline(0, color=INK, lw=.8)
ax.set(yticks=y, yticklabels=["Rx1day", "Rx5day", "Annual total", "R95pTOT"], xlim=(-5.8, 1.5), ylim=(-.5, 3.5), xticks=[-5, -3, -1, 1])
ax.set_xlabel("Paired trend shift\n(percentage points per decade)", fontsize=7.3)
clean(ax, "b", "Primary two-way intervals"); ax.grid(False); ax.grid(axis="x", color="#E1E5E8", lw=.6)

ax = axs[1, 0]; x = np.arange(3)
for k, vals in enumerate([[9, 13, 13], [41, 32, 44], [25, 22, 27]]):
    bars = ax.bar(x + (k - 1) * w, vals, w, color=colors[k]); ax.bar_label(bars, padding=2, fontsize=6.9)
ax.set(xticks=x, xticklabels=["Mann–Kendall", "Hamed–Rao", "Pre-whitened"], ylim=(0, 50))
ax.set_ylabel("Stations with FDR-significant decline", fontsize=7.2)
clean(ax, "c", "Annual-total test sensitivity")

ax = axs[1, 1]; x = [22, 26, 30]
annual = [[16, 9, 5], [45, 41, 23], [31, 25, 18]]
rx = [[9, 7, 3], [15, 13, 2], [12, 10, 2]]
for c, a, r in zip(colors, annual, rx):
    ax.plot(x, a, "o-", color=c, markersize=3, lw=1.2)
    ax.plot(x, r, "o--", color=c, markersize=3, lw=1.2)
ax.set(xticks=x, ylim=(0, 50)); ax.set_xlabel("Minimum eligible years (of 37)", fontsize=7.3)
clean(ax, "d", "Station-selection sensitivity")
ax.legend(handles=[Line2D([0], [0], color=INK, label="Annual total"), Line2D([0], [0], color=INK, ls="--", label="Rx1day")],
          frameon=False, fontsize=7, loc="upper right", handlelength=1.8)

fig.savefig(OUT / "Figure_6_Trend_Comparison.pdf", facecolor="white")
fig.savefig(OUT / "Figure_6_Trend_Comparison.png", dpi=600, facecolor="white")
print("Figure 5 written")
