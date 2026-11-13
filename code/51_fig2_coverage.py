# Code Developer: Somnath Luitel (somnath.luitel416@topper.wku.edu)
"""Step 51: manuscript Figure 2 (stations per year and the 2018-2019 dropout map).

Taken unchanged from the first figure block of the October step 11, which drew it as fig1_coverage.
Needs the Nepal local-unit boundary shapefile in gis/Local Unit/ (Survey Department of Nepal; not redistributed here).
"""
import numpy as np
import pandas as pd
import shapefile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from shapely.geometry import shape
from shapely.ops import unary_union
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT, FIG, GIS = ROOT / "outputs", ROOT / "figures", ROOT / "gis"
FIG.mkdir(exist_ok=True)

BLUE, ORANGE, GREEN, PURPLE, VERM, GREY = "#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00", "#8C8C8C"
INK, LIGHT = "#222222", "#D9D9D9"
MM = 1 / 25.4
plt.rcParams.update({"font.family": ["Arial", "DejaVu Sans"], "font.size": 8, "axes.labelsize": 9,
                     "axes.titlesize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
                     "legend.fontsize": 8, "axes.edgecolor": INK, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "text.color": INK,
                     "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
                     "pdf.fonttype": 42, "svg.fonttype": "none"})


def clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=LIGHT, lw=0.5)
    ax.set_axisbelow(True)


def label(ax, t):
    ax.text(-0.02, 1.04, t, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom", ha="right")


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png", dpi=600)
    plt.close(fig)


d = np.load(OUT / "daily_qc.npz", allow_pickle=True)
status, elig, years = d["status"], d["eligible"], d["years"]
dates = pd.to_datetime(d["dates"]); yr = dates.year.to_numpy()
sp = pd.read_csv(OUT / "station_province.csv", dtype={"station_id": str})
sp["region"] = np.where(sp.province.isin(["Sudur Pashchim", "Karnali", "Lumbini"]), "West",
                        np.where(sp.province == "Gandaki", "Central", "East"))

# ------------------------------------------------------------------ Figure 1
# coverage per year and 2017 -> 2018-19 dropout map
prov = {}
r = shapefile.Reader(str(GIS / "Local Unit" / "local_unit"))
recs = r.records()
shp = r.shapes()
for rec, s in zip(recs, shp):
    prov.setdefault(rec["Province"], []).append(shape(s.__geo_interface__))
prov_geom = {k: unary_union([g.buffer(0) for g in v]) for k, v in prov.items()}

a17 = np.array([(status[i, yr == 2017] == 0).mean() > 0.5 for i in range(len(sp))])
a1819 = np.array([(status[i, (yr >= 2018) & (yr <= 2019)] == 0).mean() > 0.5 for i in range(len(sp))])
cls = np.where(a17 & a1819, "Kept", np.where(a17 & ~a1819, "Silent 2018–19", "Other"))
sp["status_2018_19"] = cls

fig = plt.figure(figsize=(180 * MM, 78 * MM))
ax1 = fig.add_axes([0.075, 0.16, 0.40, 0.74])
ny = elig.sum(0)
anyobs = (status == 0).any(2).sum(0) if status.ndim == 3 else np.array(
    [((status[:, yr == y] == 0).any(1)).sum() for y in years])
ax1.axvspan(2017.5, 2020.5, color=LIGHT, alpha=0.6, lw=0)
ax1.plot(years, ny, color=BLUE, lw=1.6, label="Stations with an eligible year")
ax1.plot(years, anyobs, color=ORANGE, lw=1.6, ls="--", label="Stations with any observation")
ax1.set_xlim(1980, 2023); ax1.set_ylim(0, 290)
ax1.set_xlabel("Year"); ax1.set_ylabel("Number of stations")
ax1.legend(frameon=False, loc="lower left", bbox_to_anchor=(0.0, 0.0))
clean(ax1); label(ax1, "a")
pd.DataFrame({"year": years, "eligible": ny, "any_obs": anyobs}).to_csv(FIG / "fig1a_source.csv", index=False)

ax2 = fig.add_axes([0.575, 0.16, 0.41, 0.76])
for k, g in prov_geom.items():
    for poly in (g.geoms if hasattr(g, "geoms") else [g]):
        x, y = poly.exterior.xy
        ax2.fill(x, y, fc="#F2F2F2", ec=GREY, lw=0.3)
for name, col, mk, sz in (("Kept", BLUE, "o", 9), ("Silent 2018–19", VERM, "^", 11), ("Other", GREY, "s", 5)):
    m = sp.status_2018_19 == name
    ax2.scatter(sp.lon[m], sp.lat[m], s=sz, c=col, marker=mk, lw=0.3, ec="white", label=f"{name} (n = {int(m.sum())})")
ax2.set_aspect(1 / np.cos(np.radians(28)))
ax2.set_xlim(80, 88.3); ax2.set_ylim(26.3, 31.9)
ax2.set_xlabel("Longitude (°E)"); ax2.set_ylabel("Latitude (°N)")
ax2.legend(frameon=False, loc="upper left", handletextpad=0.2, borderaxespad=0.1)
label(ax2, "b")
sp[["station_id", "lon", "lat", "region", "status_2018_19"]].to_csv(FIG / "fig1b_source.csv", index=False)
save(fig, "fig1_coverage")

