# Code Developer: Somnath Luitel (somnath.luitel416@topper.wku.edu)
"""Step 43: systematic neighbour-based screen of the archive for gross monthly errors (1981-2017, all stations).

Monthly screen: for every station-month with at least 28 observed days (>= 90% of the month), the monthly total is compared with a neighbour prediction
(station's own median for that calendar month x median over up to 8 neighbours within 100 km of (neighbour total / neighbour calendar-month median), neighbours
needing the same month fully observed; at least 3 neighbours; station and neighbour medians > 5 mm). Flag if the log ratio is outside the central 99.9% range
expected from the pooled distribution of the station's calendar-month log ratios (robust: median +/- 4.5 x 1.4826 MAD of the pooled ratios of that calendar month)
AND the monthly total is at least 3x or at most 1/3 of the prediction, so only large departures count.
Also counts monsoon (JJAS) station-months with zero observed rain although all neighbours had > 100 mm.
Output: outputs/neighbour_screen_pre2018.csv (flagged station-months), outputs/neighbour_screen_summary_pre2018.csv
"""
import numpy as np
import pandas as pd
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "outputs"
d = np.load(OUT / "daily_qc.npz", allow_pickle=True)
X, status, years, sid = d["X"], d["status"], d["years"], d["station_id"].astype(str)
lat, lon = np.radians(d["lat"].astype(float)), np.radians(d["lon"].astype(float))
dt = pd.to_datetime(d["dates"]); gy, gm = dt.year.to_numpy(), dt.month.to_numpy()
V = np.where(status == 0, X, np.nan)
S = len(sid)
a = (np.sin((lat[:, None] - lat[None, :]) / 2) ** 2 + np.cos(lat[:, None]) * np.cos(lat[None, :]) * np.sin((lon[:, None] - lon[None, :]) / 2) ** 2)
DIST = 2 * 6371 * np.arcsin(np.sqrt(a))
Y = np.arange(1981, 2018)
TOT = np.full((S, len(Y), 12), np.nan); ND = np.zeros((S, len(Y), 12), int)
for t, y in enumerate(Y):
    for m in range(1, 13):
        sel = (gy == y) & (gm == m)
        v = V[:, sel]; n = np.isfinite(v).sum(1); ND[:, t, m - 1] = n
        full = n >= 0.9 * sel.sum()
        TOT[:, t, m - 1] = np.where(full, np.nansum(v, 1), np.nan)
MED = np.nanmedian(TOT, axis=1)                       # (S, 12)
NB = [[j for j in np.argsort(DIST[i]) if j != i and DIST[i, j] <= 100][:8] for i in range(S)]
rows, lr_all = [], {m: [] for m in range(12)}
cand = []
for i in range(S):
    for m in range(12):
        if not (MED[i, m] > 5):
            continue
        for t in range(len(Y)):
            x = TOT[i, t, m]
            if not np.isfinite(x):
                continue
            nbv = [TOT[j, t, m] / MED[j, m] for j in NB[i] if np.isfinite(TOT[j, t, m]) and MED[j, m] > 5]
            if len(nbv) < 3:
                continue
            P = MED[i, m] * np.median(nbv)
            if P <= 0:
                continue
            lr = np.log((x + 1) / (P + 1)); lr_all[m].append(lr)
            cand.append((i, m, t, x, P, lr, len(nbv), min(nbv) if nbv else np.nan, np.median(nbv) * 0))
res = pd.DataFrame(cand, columns=["si", "m", "t", "total", "pred", "lr", "n_nb", "x1", "x2"])
sig = {m: (np.median(lr_all[m]), 1.4826 * np.median(np.abs(np.array(lr_all[m]) - np.median(lr_all[m])))) for m in range(12)}
res["z"] = [(r.lr - sig[r.m][0]) / sig[r.m][1] for r in res.itertuples()]
res["flag"] = (res.z.abs() > 6) & (((res.total + 1) / (res.pred + 1) > 5) | ((res.total + 1) / (res.pred + 1) < 1 / 5))
res["station_id"] = sid[res.si]; res["year"] = Y[res.t]; res["month"] = res.m + 1
fl = res[res.flag][["station_id", "year", "month", "total", "pred", "z", "n_nb"]].sort_values(["station_id", "year", "month"])
fl.to_csv(OUT / "neighbour_screen_pre2018.csv", index=False)
mons = res[res.m.isin([5, 6, 7, 8])]
zero_mon = int(((mons.total == 0) & (mons.pred > 100)).sum())
summ = pd.DataFrame([("station-months screened", len(res)), ("flagged station-months", int(res.flag.sum())),
                     ("stations with at least one flag", fl.station_id.nunique()), ("stations with at least 3 flags", int((fl.station_id.value_counts() >= 3).sum())),
                     ("stations with at least 6 flags", int((fl.station_id.value_counts() >= 6).sum())),
                     ("monsoon months with 0 mm and prediction > 100 mm", zero_mon),
                     ("flagged low (total < prediction)", int((res.flag & (res.total < res.pred)).sum())),
                     ("flagged high (total > prediction)", int((res.flag & (res.total > res.pred)).sum()))], columns=["quantity", "value"])
summ.to_csv(OUT / "neighbour_screen_summary_pre2018.csv", index=False)
print(summ.to_string(index=False))
print(fl.station_id.value_counts().head(12).to_string())
print(fl.head(15).round(1).to_string(index=False))
