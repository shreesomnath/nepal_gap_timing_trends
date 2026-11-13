# Code Developer: Somnath Luitel (somnath.luitel416@topper.wku.edu)
"""Step 38: validate the gap-pattern correction on the REAL partly observed years against a neighbour-based prediction.

For each station i and partly observed year y (1981-2017; the station has at least 15 eligible years; at least 3 complete neighbours,
i.e. neighbours with an eligible year y, within 100 km), a prediction of the station's complete-year index is built from the neighbours:
    P = (station median of its eligible-year index) x median over neighbours of (neighbour index in y / neighbour median).
The observed index of the partly observed year (O, uncorrected) and the gap-corrected index (C, the same correction as step 17) are compared
with P through the log ratio ln(X / P). The same log ratio for the station's eligible years (X = observed complete value, leaving that year
out of the medians) gives the reference: the neighbour prediction is itself imperfect and may be biased. A correction that works should
give a median ln(C / P) equal to the reference median, and an uncorrected O should sit below it by about the expected loss.
Reported by index and by the observed fraction of days (bins), with 95% station-bootstrap intervals for the median.
Output: outputs/correction_validation_real_years.csv
"""
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from pathlib import Path

rng = np.random.default_rng(38)
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
d = np.load(OUT / "daily_qc.npz", allow_pickle=True)
X, status, elig, years, sid = d["X"], d["status"], d["eligible"], d["years"], d["station_id"].astype(str)
lat, lon = np.radians(d["lat"].astype(float)), np.radians(d["lon"].astype(float))
dates = pd.to_datetime(d["dates"]); yr = dates.year.to_numpy()
V = np.where(status == 0, X, np.nan)
p95 = pd.read_csv(OUT / "p95_baseline.csv", dtype={"station_id": str}).p95_1981_2010.to_numpy()
PER = (years >= 1981) & (years <= 2017)
yrs = years[PER]; T = len(yrs); S = V.shape[0]
NAMES = ["Rx1day", "Rx5day", "PRCPTOT", "R95pTOT"]
a = (np.sin((lat[:, None] - lat[None, :]) / 2) ** 2 + np.cos(lat[:, None]) * np.cos(lat[None, :]) * np.sin((lon[:, None] - lon[None, :]) / 2) ** 2)
DIST = 2 * 6371 * np.arcsin(np.sqrt(a))
m_idx = [list(years).index(y) for y in yrs]


def yv(i, y):
    v = V[i, yr == y]
    return np.delete(v, 59) if v.size == 366 else v


def ind(x, p):
    with np.errstate(all="ignore"):
        rx1 = np.nanmax(x, -1)
        w = sliding_window_view(x, 5, axis=-1)
        s5 = np.where(np.isfinite(w).all(-1), w.sum(-1), np.nan)
        rx5 = np.nanmax(s5, -1)
        pt = np.nansum(np.where(x >= 1, x, 0), -1)
        r95 = np.nansum(np.where(x > p, x, 0), -1) if np.isfinite(p) else np.full(pt.shape, np.nan)
    return np.stack([rx1, rx5, pt, r95])


G = np.full((4, S, T), np.nan)       # complete (eligible) years only
O = np.full((4, S, T), np.nan)       # observed index in partly observed years
C = np.full((4, S, T), np.nan)       # corrected index
FR = np.full((S, T), np.nan)         # observed fraction of days
for i in range(S):
    ey = [t for t in range(T) if elig[i, m_idx[t]]]
    if len(ey) < 15:
        continue
    comp = np.array([yv(i, yrs[t]) for t in ey]); base = ind(comp, p95[i])
    for k, t in enumerate(ey):
        G[:, i, t] = base[:, k]
    for t, y in enumerate(yrs):
        if elig[i, m_idx[t]] or (status[i, yr == y] == 2).all():
            continue
        v = yv(i, y)
        if not np.isfinite(v).any():
            continue
        FR[i, t] = np.isfinite(v).mean()
        O[:, i, t] = ind(v, p95[i])
        cm = comp.copy(); cm[:, ~np.isfinite(v)] = np.nan
        with np.errstate(all="ignore"):
            r = ind(cm, p95[i]) / base
        for kk in range(4):
            rr = r[kk][np.isfinite(r[kk]) & (r[kk] > 0)]
            g = np.median(rr) if rr.size else np.nan
            C[kk, i, t] = O[kk, i, t] / g if (np.isfinite(g) and g >= 0.2) else np.nan

rows = []
for k, name in enumerate(NAMES):
    Gk = G[k]
    med = np.nanmedian(Gk, axis=1)
    with np.errstate(all="ignore"):
        Gn = Gk / med[:, None]
    ref, ob, co, st_o, fr_o = [], [], [], [], []
    for i in range(S):
        nb = [j for j in np.argsort(DIST[i]) if j != i and DIST[i, j] <= 100 and np.isfinite(med[j]) and med[j] > 0][:8]
        if len(nb) < 3 or not np.isfinite(med[i]) or med[i] <= 0:
            continue
        for t in range(T):
            cnt = np.isfinite(Gn[nb, t]).sum()
            if cnt < 3:
                continue
            P = med[i] * np.nanmedian(Gn[nb, t])
            if not (np.isfinite(P) and P > 0):
                continue
            if np.isfinite(Gk[i, t]) and Gk[i, t] > 0:       # eligible-year reference
                ref.append((i, np.log(Gk[i, t] / P)))
            elif np.isfinite(O[k, i, t]) and O[k, i, t] > 0 and np.isfinite(C[k, i, t]) and C[k, i, t] > 0:
                ob.append((i, np.log(O[k, i, t] / P), FR[i, t])); co.append((i, np.log(C[k, i, t] / P)))
    ref = np.array(ref); ob = np.array(ob); co = np.array(co)

    def boot(arr):
        st = np.unique(arr[:, 0]); groups = {s: arr[arr[:, 0] == s, 1] for s in st}
        bs = []
        for _ in range(1000):
            pick = rng.choice(st, len(st))
            bs.append(np.median(np.concatenate([groups[s] for s in pick])))
        return np.percentile(bs, [2.5, 97.5])

    r_med = np.median(ref[:, 1]); r_ci = boot(ref)
    rows.append((name, "eligible years (reference)", "all", len(ref), r_med, *r_ci))
    bins = [("all", np.ones(len(ob), bool)), ("observed < 70% of days", ob[:, 2] < 0.7), ("observed 70-95%", (ob[:, 2] >= 0.7) & (ob[:, 2] < 0.95))]
    for lab, m in bins:
        if m.sum() < 15:
            continue
        for tag, arr in (("partly observed, uncorrected", np.column_stack([ob[m, 0], ob[m, 1]])), ("partly observed, corrected", np.column_stack([co[m, 0], co[m, 1]]))):
            rows.append((name, tag, lab, int(m.sum()), np.median(arr[:, 1]), *boot(arr)))
R = pd.DataFrame(rows, columns=["index", "series", "observed_fraction", "n", "median_log_ratio_to_neighbour_prediction", "lo95", "hi95"])
R["median_ratio"] = np.exp(R.median_log_ratio_to_neighbour_prediction)
R.to_csv(OUT / "correction_validation_real_years.csv", index=False)
pd.set_option("display.width", 220)
print(R.round(3).to_string(index=False))
