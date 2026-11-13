# Code Developer: Somnath Luitel (somnath.luitel416@topper.wku.edu)
"""Step 42: how different is ERA5 from the gauges as a daily series? (1981-2017, 219 trend stations, days observed at the gauge only)

Per station: wet-day frequency (>= 1 mm), mean wet-day intensity, P(wet | wet the day before), annual-maximum 1-day (median over eligible years),
annual PRCPTOT; for the cell value and the 3x3 mean. Summary: median across stations of ERA5 / gauge ratio (frequency, intensity, Rx1day, PRCPTOT) and difference
of P(wet|wet), with 95% station-bootstrap intervals. Output: outputs/era5_vs_gauge_distribution.csv
"""
import numpy as np
import pandas as pd
from pathlib import Path

rng = np.random.default_rng(42)
OUT = Path(__file__).resolve().parents[1] / "outputs"
g = np.load(OUT / "daily_qc.npz", allow_pickle=True)
status, elig, years, sid, X = g["status"], g["eligible"], g["years"], g["station_id"].astype(str), g["X"]
gd = pd.to_datetime(g["dates"]); gy = gd.year.to_numpy()
e = np.load(OUT / "era5_station_daily.npz", allow_pickle=True)
ed = pd.DatetimeIndex(e["dates"]); esid = e["station_id"].astype(str)
ER = {"cell": e["era5_centre_mm"].astype(float), "3x3 mean": e["era5_3x3_mm"].astype(float)}
# common dates 1981-2017
gmask = (gy >= 1981) & (gy <= 2017); ekey = {d: k for k, d in enumerate(ed)}
gdates = gd[gmask]; gi = np.where(gmask)[0]
pairs = [(a, ekey[d]) for a, d in zip(gi, gdates) if d in ekey]
ga = np.array([p[0] for p in pairs]); ea = np.array([p[1] for p in pairs])
m_idx = [list(years).index(y) for y in range(1981, 2018)]
sel = [k for k in range(len(sid)) if elig[k, m_idx].sum() >= 26 and sid[k] in set(esid)]
epos = {s: k for k, s in enumerate(esid)}
yy = gy[ga]
rows = []
for i in sel:
    gv = np.where(status[i, ga] == 0, X[i, ga], np.nan)
    ok = np.isfinite(gv)
    ok1 = ok[1:] & ok[:-1]
    for lab, A in ER.items():
        ev = A[epos[sid[i]], ea]
        gw, ew = gv >= 1, ev >= 1
        f = lambda w, v: (w[ok].mean(), v[ok & w].mean() if (ok & w).any() else np.nan,
                          (w[1:][ok1 & w[:-1]].mean() if (ok1 & w[:-1]).any() else np.nan))
        gf, gi_, gp = f(gw, gv); ef, ei_, ep = f(ew, ev)
        # annual maxima and totals in eligible years
        gr, er, gt, et = [], [], [], []
        for y in range(1981, 2018):
            if not elig[i, list(years).index(y)]:
                continue
            m = (yy == y) & ok
            gr.append(np.nanmax(gv[m])); er.append(np.nanmax(ev[m])); gt.append(np.nansum(np.where(gv[m] >= 1, gv[m], 0))); et.append(np.nansum(np.where(ev[m] >= 1, ev[m], 0)))
        rows.append((sid[i], lab, ef / gf, ei_ / gi_, ep - gp, np.median(er) / np.median(gr), np.median(et) / np.median(gt), gf, ef))
D = pd.DataFrame(rows, columns=["station_id", "era5", "freq_ratio", "intensity_ratio", "persist_diff", "rx1day_ratio", "prcptot_ratio", "gauge_freq", "era5_freq"])
out = []
for lab, g_ in D.groupby("era5"):
    for c in ("freq_ratio", "intensity_ratio", "persist_diff", "rx1day_ratio", "prcptot_ratio"):
        v = g_[c].dropna().to_numpy()
        bs = [np.median(rng.choice(v, len(v))) for _ in range(2000)]
        out.append((lab, c, len(v), np.median(v), *np.percentile(bs, [2.5, 97.5])))
R = pd.DataFrame(out, columns=["era5", "quantity", "n_stations", "median", "lo95", "hi95"])
R.to_csv(OUT / "era5_vs_gauge_distribution.csv", index=False)
print(R.round(3).to_string(index=False))
