# Code Developer: Somnath Luitel (somnath.luitel416@topper.wku.edu)
"""Step 37: composition-only placebo for the archive shift.

The paired shift (all years minus eligible years only) compares a longer series with a shorter one. Part of it could be genuine
signal carried by the extra (partly observed, mostly late) years, which no gap correction should remove. To measure that part, the
same two year sets are applied to series that contain no gaps and no gauge-specific gap bias:
  ERA5   : full ERA5 daily rain at the station cell (never masked)
  Region : mean of the normalised annual index of up to five complete neighbouring stations within 100 km (their eligible years only)
For each station with at least one partly observed year, the Sen trend (% per decade) on the all-years set minus that on the
eligible-years set is computed; the mean over stations has a 95% station-bootstrap interval. 1981-2017, the 219 trend stations.
Output: outputs/composition_placebo_pre2018.csv
"""
import numpy as np
import pandas as pd
from scipy import stats
from numpy.lib.stride_tricks import sliding_window_view
from pathlib import Path

rng = np.random.default_rng(37)
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
g = np.load(OUT / "daily_qc.npz", allow_pickle=True)
status, elig, years, sid = g["status"], g["eligible"], g["years"], g["station_id"].astype(str)
lat, lon = np.radians(g["lat"].astype(float)), np.radians(g["lon"].astype(float))
gd = pd.to_datetime(g["dates"]); gy = gd.year.to_numpy()
Y = np.arange(1981, 2018); T = len(Y)
m_idx = [list(years).index(y) for y in Y]
sel = np.where(elig[:, m_idx].sum(1) >= 26)[0]
ann = pd.read_csv(OUT / "annual_indices.csv", dtype={"station_id": str}).set_index(["station_id", "year"])
e = np.load(OUT / "era5_station_daily.npz", allow_pickle=True)
ER = e["era5_centre_mm"].astype(float); ed = pd.DatetimeIndex(e["dates"]); ey = ed.year.to_numpy(); keep_e = ~((ed.month == 2) & (ed.day == 29))
p95 = pd.read_csv(OUT / "p95_baseline.csv", dtype={"station_id": str}).p95_1981_2010.to_numpy()
NAMES = ["Rx1day", "Rx5day", "PRCPTOT", "R95pTOT"]
a = (np.sin((lat[:, None] - lat[None, :]) / 2) ** 2 + np.cos(lat[:, None]) * np.cos(lat[None, :]) * np.sin((lon[:, None] - lon[None, :]) / 2) ** 2)
DIST = 2 * 6371 * np.arcsin(np.sqrt(a))


def ind(x, p):
    with np.errstate(all="ignore"):
        rx1 = np.nanmax(x, -1)
        w = sliding_window_view(x, 5, axis=-1)
        rx5 = np.nanmax(w.sum(-1), -1)
        pt = np.nansum(np.where(x >= 1, x, 0), -1)
        r95 = np.nansum(np.where(x > p, x, 0), -1)
    return np.array([rx1, rx5, pt, r95])


def sen(v, t):
    ok = np.isfinite(v)
    if ok.sum() < 15 or np.nanmedian(v[ok]) <= 0:
        return np.nan
    return 100 * 10 * stats.theilslopes(v[ok], t[ok])[0] / np.nanmedian(v[ok])


# ERA5 annual indices at station cells
E = np.full((4, len(sid), T), np.nan)
for i in sel:
    wet = ER[i, (ey >= 1981) & (ey <= 2010)]
    pe = np.nanpercentile(wet[wet >= 1], 95)
    for t, y in enumerate(Y):
        E[:, i, t] = ind(ER[i, (ey == y) & keep_e], pe)

# gauge annual indices (eligible years only) as an array for neighbours
G = np.full((4, len(sid), T), np.nan)
for i in range(len(sid)):
    for t, y in enumerate(Y):
        if elig[i, m_idx[t]] and (sid[i], y) in ann.index:
            r = ann.loc[(sid[i], y)]
            G[:, i, t] = [r.Rx1day, r.Rx5day, r.PRCPTOT, r.R95pTOT]
Gn = G / np.nanmean(G, axis=2, keepdims=True)

rows = []
for k, name in enumerate(NAMES):
    d_era, d_reg = [], []
    for i in sel:
        elg = elig[i, m_idx]
        anyobs = np.array([(status[i, gy == y] == 0).any() for y in Y])
        if not (anyobs & ~elg).any():
            continue
        nb = [j for j in np.argsort(DIST[i]) if j != i and DIST[i, j] <= 100][:5]
        reg = np.full(T, np.nan)
        if len(nb) >= 2:
            cnt = np.isfinite(Gn[k, nb, :]).sum(0)
            with np.errstate(all="ignore"):
                reg = np.where(cnt >= 2, np.nanmean(Gn[k, nb, :], axis=0), np.nan)
        for series, out in ((E[k, i], d_era), (reg, d_reg)):
            s1 = np.where(elg, series, np.nan); s4 = np.where(anyobs, series, np.nan)
            t1, t4 = sen(s1, Y.astype(float)), sen(s4, Y.astype(float))
            if np.isfinite(t1) and np.isfinite(t4):
                out.append(t4 - t1)
    for lab, dv in (("ERA5 at the station cell", d_era), ("Neighbouring gauges", d_reg)):
        dv = np.array(dv)
        bs = [rng.choice(dv, len(dv)).mean() for _ in range(2000)]
        rows.append((name, lab, len(dv), dv.mean(), *np.percentile(bs, [2.5, 97.5])))
R = pd.DataFrame(rows, columns=["index", "series", "n", "mean_composition_shift", "lo95", "hi95"])
R.to_csv(OUT / "composition_placebo_pre2018.csv", index=False)
pd.set_option("display.width", 200)
print(R.round(2).to_string(index=False))
