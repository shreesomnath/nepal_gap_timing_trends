# Code Developer: Somnath Luitel (somnath.luitel416@topper.wku.edu)
"""Step 22: do ERA5 trends at the station cells resemble the gauge trends, 1981-2017?

For the 219 stations with at least 26 eligible gauge years, compare
  gauge  : Sen slope (% of median per decade) of the strict series (eligible years only)
  ERA5   : same statistic from ERA5 daily rain at the station cell, all years 1981-2017
Reported per index: Spearman correlation across stations, share of stations with the same sign,
and medians by Karki et al. (2017) longitude zone. ERA5 is a reanalysis and does not ingest DHM
gauges, so agreement is a partial independent check on the spatial pattern; it is not truth.
Output: outputs/era5_vs_gauge_trends.csv (stations), outputs/era5_vs_gauge_trends_summary.csv
"""
import numpy as np
import pandas as pd
from scipy import stats
from numpy.lib.stride_tricks import sliding_window_view
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
g = np.load(OUT / "daily_qc.npz", allow_pickle=True)
status, elig, years, sid, lon = g["status"], g["eligible"], g["years"], g["station_id"].astype(str), g["lon"].astype(float)
X = g["X"]
gd = pd.to_datetime(g["dates"]); gy = gd.year.to_numpy()
V = np.where(status == 0, X, np.nan)
e = np.load(OUT / "era5_station_daily.npz", allow_pickle=True)
ER = e["era5_centre_mm"].astype(float); ed = pd.DatetimeIndex(e["dates"]); ey = ed.year.to_numpy()
p95g = pd.read_csv(OUT / "p95_baseline.csv", dtype={"station_id": str}).p95_1981_2010.to_numpy()
keep_e = ~((ed.month == 2) & (ed.day == 29))
Y = np.arange(1981, 2018)
m_idx = [list(years).index(y) for y in Y]
cand = np.where(elig[:, m_idx].sum(1) >= 26)[0]
NAMES = ["Rx1day", "Rx5day", "PRCPTOT", "R95pTOT"]


def ind(a, p):
    with np.errstate(all="ignore"):
        rx1 = np.nanmax(a, -1)
        w = sliding_window_view(a, 5, axis=-1)
        s5 = np.where(np.isfinite(w).all(-1), w.sum(-1), np.nan)
        rx5 = np.nanmax(s5, -1)
        pt = np.nansum(np.where(a >= 1, a, 0), -1)
        r95 = np.nansum(np.where(a > p, a, 0), -1)
    return np.array([rx1, rx5, pt, r95])


def sen(vals, yrs):
    ok = np.isfinite(vals)
    if ok.sum() < 20 or np.nanmedian(vals[ok]) <= 0:
        return np.nan
    return 100 * 10 * stats.theilslopes(vals[ok], yrs[ok])[0] / np.nanmedian(vals[ok])


rows = []
for i in cand:
    wet = ER[i, (ey >= 1981) & (ey <= 2010)]
    pe = np.nanpercentile(wet[wet >= 1], 95)
    gm, em = [], []
    for y in Y:
        j = list(years).index(y)
        v = V[i, gy == y]; v = np.delete(v, 59) if v.size == 366 else v
        gm.append(ind(v, p95g[i]) if elig[i, j] else np.full(4, np.nan))
        ev = ER[i, (ey == y) & keep_e]
        em.append(ind(ev, pe))
    gm, em = np.array(gm), np.array(em)
    r = {"station_id": sid[i], "lon": lon[i]}
    for k, n in enumerate(NAMES):
        r["gauge_" + n] = sen(gm[:, k], Y.astype(float)); r["era5_" + n] = sen(em[:, k], Y.astype(float))
    rows.append(r)
D = pd.DataFrame(rows)
D["zone"] = np.where(D.lon < 83, "West", np.where(D.lon < 86, "Central", "East"))
D.to_csv(OUT / "era5_vs_gauge_trends.csv", index=False)
out = []
for n in NAMES:
    ok = D[["gauge_" + n, "era5_" + n]].dropna().index
    a, b = D.loc[ok, "gauge_" + n], D.loc[ok, "era5_" + n]
    rho, p = stats.spearmanr(a, b)
    row = {"index": n, "n": len(ok), "spearman": rho, "p": p, "same_sign": float((np.sign(a) == np.sign(b)).mean()),
           "median_gauge": a.median(), "median_era5": b.median()}
    for z in ("West", "Central", "East"):
        m = D.loc[ok].zone == z
        row["gauge_" + z] = D.loc[ok][m]["gauge_" + n].median(); row["era5_" + z] = D.loc[ok][m]["era5_" + n].median()
    out.append(row)
S_ = pd.DataFrame(out)
S_.to_csv(OUT / "era5_vs_gauge_trends_summary.csv", index=False)
pd.set_option("display.width", 220)
print(S_.round(3).to_string(index=False))
