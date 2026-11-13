# Code Developer: Somnath Luitel (somnath.luitel416@topper.wku.edu)
"""Step 2: annual extreme-precipitation indices per station-year.

Indices follow ETCCDI definitions with a 1 mm wet-day threshold:
  Rx1day   annual maximum daily precipitation (mm)
  Rx5day   annual maximum consecutive 5-day total (mm); windows need all five days
  PRCPTOT  annual total on wet days (mm)
  R95pTOT  annual total on days above the station's 1981-2010 wet-day 95th percentile (mm)

Each index is also stored with the fraction of the year observed, so later steps can
apply different completeness rules. Suspect and missing days count as unobserved.
Output: outputs/annual_indices.csv (long format).
"""
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
d = np.load(OUT / "daily_qc.npz", allow_pickle=True)
X, status = d["X"], d["status"]
dates = pd.to_datetime(d["dates"])
sid = d["station_id"]
yr = dates.year.to_numpy()
years = np.unique(yr)

V = np.where(status == 0, X, np.nan)          # quality-controlled observed only
WET = 1.0

# 1981-2010 wet-day 95th percentile per station, requiring at least 1000 wet days
base = (yr >= 1981) & (yr <= 2010)
p95 = np.full(len(sid), np.nan)
for i in range(len(sid)):
    w = V[i, base]
    w = w[w >= WET]
    if w.size >= 1000:
        p95[i] = np.percentile(w, 95)

rows = []
for i in range(len(sid)):
    v = V[i]
    # 5-day running sums, valid only when all five days are observed
    c = np.convolve(np.nan_to_num(v), np.ones(5), "valid")
    n = np.convolve((~np.isnan(v)).astype(int), np.ones(5, int), "valid")
    s5 = np.where(n == 5, c, np.nan)
    s5_end = np.concatenate([np.full(4, np.nan), s5])   # aligned to window end day
    for y in years:
        sel = yr == y
        vy = v[sel]
        nobs = np.isfinite(vy).sum()
        if nobs == 0:
            continue
        frac = nobs / sel.sum()
        wet = vy[vy >= WET]
        r95 = np.nan if np.isnan(p95[i]) else wet[wet > p95[i]].sum()
        rows.append((sid[i], y, round(frac, 4), np.nanmax(vy),
                     np.nanmax(s5_end[sel]) if np.isfinite(s5_end[sel]).any() else np.nan,
                     wet.sum(), r95))

ann = pd.DataFrame(rows, columns=["station_id", "year", "obs_frac", "Rx1day", "Rx5day",
                                  "PRCPTOT", "R95pTOT"])
ann.to_csv(OUT / "annual_indices.csv", index=False)
pd.DataFrame({"station_id": sid, "p95_1981_2010": p95}).to_csv(OUT / "p95_baseline.csv", index=False)
print(ann.describe().round(2).to_string())
print("stations with baseline p95:", np.isfinite(p95).sum())
