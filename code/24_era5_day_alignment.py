# Code Developer: Somnath Luitel (somnath.luitel416@topper.wku.edu)
"""Step 24: which ERA5 day definition best matches the DHM daily gauge dates?

DHM reads its daily gauges at 08:45 Nepal time (UTC+5:45), i.e. 03:00 UTC, and the archive may label a
reading with the date of the reading or the day before. Here nothing is assumed: ERA5 hourly rain at
each station cell is summed over the 24 h ending at UTC hour h (h = 0..23) and compared with the gauge
on date D + s (s = -1, 0, +1 days) for all gauge-observed days 1981-2023. Score per (h, s): median
across stations of the daily Pearson correlation, and the median wet-day (>= 1 mm) hit rate and
false-alarm ratio. The best cell shows the alignment the data support.
Inputs: era5_final/*.nc (hourly, one grid), outputs/era5_station_daily.npz (cell coordinates),
outputs/daily_qc.npz (gauge values and status). Output: outputs/era5_day_alignment.csv
"""
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
e = np.load(OUT / "era5_station_daily.npz", allow_pickle=True)
clat, clon = e["cell_lat"].astype(float), e["cell_lon"].astype(float)
g = np.load(OUT / "daily_qc.npz", allow_pickle=True)
X, status = g["X"], g["status"]
gd = pd.DatetimeIndex(g["dates"])
G = np.where(status == 0, X, np.nan)                     # (S, days)
S = G.shape[0]
parts, times = [], []
for p in sorted((ROOT / "era5_final").glob("*.nc")):
    ds = xr.open_dataset(p)
    da = ds["total_precipitation"]
    tdim = "valid_time" if "valid_time" in da.dims else "time"
    li = [int(np.abs(ds.latitude.values - a).argmin()) for a in clat]
    lo = [int(np.abs(ds.longitude.values - b).argmin()) for b in clon]
    v = da.values[:, li, lo].astype("float32") * 1000     # (hours, S) mm
    parts.append(v); times.append(pd.DatetimeIndex(ds[tdim].values)); ds.close()
H = np.concatenate(parts); T = times[0].append(times[1:])
H = H[~T.duplicated()]; T = T[~T.duplicated()]
print("hourly array", H.shape, T[0], T[-1])
c = np.cumsum(np.vstack([np.zeros((1, S), "float32"), H]), axis=0, dtype=np.float64)
S24 = np.full(H.shape, np.nan, dtype="float32")
S24[23:] = c[24:] - c[:-24]                               # element k = sum of hours k..k+23, stored at t = k+23
# S24[t] = sum of hours t-23..t  (stamp t is the hour ending at t)
dates = pd.date_range("1981-01-01", "2023-12-31")
tpos = pd.Series(np.arange(len(T)), index=T)
gi = pd.Series(np.arange(len(gd)), index=gd)
rows = []
for h in range(24):
    idx = tpos.reindex(dates + pd.Timedelta(hours=h)).values
    ok = ~np.isnan(idx); idx = idx[ok].astype(int); dd = dates[ok]
    E = S24[idx]                                          # (days, S)
    for s in (-1, 0, 1):
        gdates = dd + pd.Timedelta(days=s)
        okg = gi.reindex(gdates).values
        m = ~np.isnan(okg); Gm = G[:, okg[m].astype(int)].T; Em = E[m]
        cors, hits, fas = [], [], []
        for i in range(S):
            a, b = Gm[:, i], Em[:, i]
            k = np.isfinite(a) & np.isfinite(b)
            if k.sum() < 2000:
                continue
            cors.append(np.corrcoef(a[k], b[k])[0, 1])
            wg, we = a[k] >= 1, b[k] >= 1
            hits.append((wg & we).sum() / max(wg.sum(), 1)); fas.append((we & ~wg).sum() / max(we.sum(), 1))
        rows.append((h, s, len(cors), np.median(cors), np.median(hits), np.median(fas)))
R = pd.DataFrame(rows, columns=["end_hour_utc", "gauge_date_shift_days", "n_stations", "median_corr", "median_hit_rate", "median_false_alarm_ratio"])
R.to_csv(OUT / "era5_day_alignment.csv", index=False)
pd.set_option("display.width", 160)
print(R.pivot(index="end_hour_utc", columns="gauge_date_shift_days", values="median_corr").round(3).to_string())
best = R.sort_values("median_corr", ascending=False).head(5)
print("\nbest five:\n", best.round(3).to_string(index=False))
