# Code Developer: Somnath Luitel (somnath.luitel416@topper.wku.edu)
"""Step 52: manuscript Figure 1 (workflow schematic), journal style: shows the mechanism (what happens to partly observed years) with year-strip glyphs,
one accent color, labeled arrows, short labels. Sentences go in the caption."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({"font.family": "Arial", "font.size": 8, "pdf.fonttype": 42, "svg.fonttype": "none"})
INK, MID, LIGHT = "#1a1a1a", "#8a8a8a", "#cfcfcf"
ACC = "#D55E00"          # single accent: partly observed / masked years (Okabe-Ito vermillion)
PANEL = "#f4f4f4"
W, H = 7.0, 4.35
fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0.24, H); ax.axis("off")


def box(x, y, w, h, lines, bold_first=True, fc="white"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.05", fc=fc, ec=INK, lw=0.7))
    n = len(lines); lh = 0.15
    y0 = y + h / 2 + (n - 1) * lh / 2
    for i, t in enumerate(lines):
        ax.text(x + w / 2, y0 - i * lh, t, ha="center", va="center", color=INK,
                fontweight="bold" if (i == 0 and bold_first) else "normal")


def arrow(x1, y1, x2, y2, label=None, lx=None, ly=None, ha="left"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=8, lw=0.8, color=INK,
                                 shrinkA=0, shrinkB=0))
    if label:
        ax.text(lx if lx is not None else (x1 + x2) / 2, ly if ly is not None else (y1 + y2) / 2 + 0.07, label,
                ha=ha, va="bottom" if ly is None else "center", fontsize=7.3, style="italic", color="#333333")


S, G = 0.14, 0.035   # square size, gap
NY = 12


def strip(x, y, kinds):
    """kinds: 'o' observed eligible, 'p' partly observed (incomplete total), 'c' corrected, 'x' excluded."""
    for i, k in enumerate(kinds):
        xx = x + i * (S + G)
        if k == "o":
            ax.add_patch(Rectangle((xx, y), S, S, fc=MID, ec=MID, lw=0.5))
        elif k == "x":
            ax.add_patch(Rectangle((xx, y), S, S, fc="white", ec=LIGHT, lw=0.6, ls=(0, (1.5, 1))))
        elif k == "p":
            ax.add_patch(Rectangle((xx, y), S, S, fc="white", ec=ACC, lw=0.7))
            ax.add_patch(Rectangle((xx, y), S, S * 0.55, fc=ACC, ec="none"))
        elif k == "c":
            ax.add_patch(Rectangle((xx, y), S, S, fc=ACC, ec=ACC, lw=0.7))


def row(x_label, x_strip, y, label, kinds):
    ax.text(x_label, y + S / 2, label, ha="left", va="center", color=INK, fontsize=7.6)
    strip(x_strip, y, kinds)

# ---- top row ----
yt, ht = 3.62, 0.62
box(0.15, yt, 1.80, ht, ["DHM daily archive", "277 gauges, 1980\u20132023", "checked vs GHCN-Daily, ERA5"])
box(2.375, yt, 2.25, ht, ["Trend sample", "219 gauges, 1981\u20132017", "Rx1day, Rx5day, PRCPTOT, R95pTOT"])
box(5.05, yt, 1.80, ht, ["Missingness diagnostics", "PIT rank vs masked years", "ERA5 rain on missing days"])
arrow(1.95, yt + ht / 2, 2.375, yt + ht / 2, "QC", lx=2.1625, ly=yt + ht / 2 + 0.12, ha="center")
arrow(4.625, yt + ht / 2, 5.05, yt + ht / 2, "test", lx=4.8375, ly=yt + ht / 2 + 0.12, ha="center")

# ---- panels ----
py, ph = 0.66, 2.68
pw = 3.35
for px, lab in ((0.10, "(a) Archive: three treatments"), (3.55, "(b) Experiment: imposed gap timing")):
    ax.add_patch(FancyBboxPatch((px, py), pw, ph, boxstyle="round,pad=0,rounding_size=0.06", fc=PANEL, ec="none"))
    ax.text(px + 0.1, py + ph - 0.13, lab, ha="left", va="center", fontweight="bold", color=INK)
# split arrows from trend sample
xc = 2.375 + 2.25 / 2
ax.plot([xc, xc], [yt, 3.47], color=INK, lw=0.8)
ax.plot([1.78, 5.23], [3.47, 3.47], color=INK, lw=0.8)
arrow(1.78, 3.47, 1.78, py + ph)
arrow(5.23, 3.47, 5.23, py + ph)

# panel (a) strips: partly observed years at 3, 7, 9, 10, 11 (more frequent late, as in the archive)
rec = list("oopooooopppo")
T1 = ["x" if k == "p" else k for k in rec]
T3 = ["c" if k == "p" else k for k in rec]
ax_l, ax_s = 0.22, 1.18
ys = [2.82, 2.50, 2.22, 1.94]
row(ax_l, ax_s, ys[0], "Station record", rec)
ax.plot([0.22, 3.33], [ys[0] - 0.09, ys[0] - 0.09], color=LIGHT, lw=0.6)
row(ax_l, ax_s, ys[1], "T1  eligible only", T1)
row(ax_l, ax_s, ys[2], "T2  uncorrected", rec)
row(ax_l, ax_s, ys[3], "T3  gap-corrected", T3)
ax.text(ax_s, ys[0] + S + 0.07, "1981", ha="left", va="bottom", fontsize=6.8, color="#555555")
ax.text(ax_s + NY * (S + G) - G, ys[0] + S + 0.07, "2017", ha="right", va="bottom", fontsize=6.8, color="#555555")
arrow(1.78, 1.88, 1.78, 1.42, "Sen's slope per treatment", lx=1.86, ly=1.65)
box(0.55, 0.8, 2.45, 0.62, ["Paired trend shifts", "T2 \u2212 T1 and T3 \u2212 T1", "(% of median per decade)"])

# panel (b) strips
bx_l, bx_s = 3.67, 4.63
comp = ["o"] * NY
rand = ["p" if i in (1, 5, 8) else "o" for i in range(NY)]
clus = ["p" if i in (8, 9, 10) else "o" for i in range(NY)]
row(bx_l, bx_s, ys[0], "Complete series", comp)
ax.plot([3.67, 6.78], [ys[0] - 0.09, ys[0] - 0.09], color=LIGHT, lw=0.6)
row(bx_l, bx_s, ys[1], "Random timing", rand)
row(bx_l, bx_s, ys[2], "Archive frequency", ["p" if i in (4, 9, 11) else "o" for i in range(NY)])
row(bx_l, bx_s, ys[3], "Clustered timing", clus)
ax.text(bx_s, ys[0] + S + 0.07, "gauge or ERA5", ha="left", va="bottom", fontsize=6.8, color="#555555")
arrow(5.23, 1.88, 5.23, 1.42, "re-estimate, \u00b1 correction", lx=5.31, ly=1.65)
box(4.0, 0.8, 2.45, 0.62, ["Trend bias", "relative to the undegraded series", "(% of median per decade)"])

# ---- legend ----
ly = 0.36
items = [("o", "eligible year"), ("p", "partly observed (incomplete total)"), ("c", "corrected"), ("x", "excluded")]
x = 0.55
for k, t in items:
    strip(x, ly, [k])
    ax.text(x + S + 0.06, ly + S / 2, t, ha="left", va="center", fontsize=7.3, color=INK)
    x += {"eligible year": 1.15, "partly observed (incomplete total)": 2.3, "corrected": 0.95}.get(t, 0)

from pathlib import Path
(Path(__file__).resolve().parents[1] / "figures").mkdir(exist_ok=True)
out = str(Path(__file__).resolve().parents[1] / "figures" / "Figure_1")
kw = dict(bbox_inches="tight", pad_inches=0.04)
fig.savefig(out + ".pdf", **kw); fig.savefig(out + ".png", dpi=600, **kw); fig.savefig(out + ".svg", **kw)
print("ok")
