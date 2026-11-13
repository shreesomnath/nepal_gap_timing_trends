# Code Developer: Somnath Luitel (somnath.luitel416@topper.wku.edu)
"""Step 1: station inventory, status coding and quality-control flags.

Reads the original wide CSV (never modified) and writes:
  outputs/station_inventory.csv   one row per station with coverage statistics
  outputs/qc_flags.csv            one row per flagged station-day or station-month
  outputs/daily_qc.npz            value array, status codes and dates for later steps

Status codes in daily_qc.npz:
  0 observed, 1 missing (inside operating period), 2 outside operation,
  3 suspect (observed value set aside for the primary analysis)
"""
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "melted_ppt_from_1980.csv"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

OBS, MISS, OUTSIDE, SUSPECT = 0, 1, 2, 3

df = pd.read_csv(SRC, dtype={"Station_ID": str})
meta = df.iloc[:, :4].copy()
X = df.iloc[:, 4:].to_numpy(dtype=float)
dates = pd.to_datetime(df.columns[4:], format="%b_%d_%Y")
yr, mo = dates.year.to_numpy(), dates.month.to_numpy()
ns, nt = X.shape

# Operating period: first to last observed day. Days outside it are not "missing".
ok = ~np.isnan(X)
first = ok.argmax(1)
last = nt - 1 - ok[:, ::-1].argmax(1)
status = np.full(X.shape, MISS, dtype=np.int8)
status[ok] = OBS
idx = np.arange(nt)
status[(idx[None, :] < first[:, None]) | (idx[None, :] > last[:, None])] = OUTSIDE

flags = []

# Flag A: zero-coded missing. A fully observed calendar month in Jun-Sep with a total
# of exactly 0 mm, at a station whose median total for that month in other complete
# years exceeds 50 mm. Monsoon months with no rain at all are implausible there.
month_keys = pd.MultiIndex.from_arrays([yr, mo])
ukeys = month_keys.unique()
mtot = np.full((ns, len(ukeys)), np.nan)
for k, (y, m) in enumerate(ukeys):
    sel = (yr == y) & (mo == m)
    blk = X[:, sel]
    complete = ~np.isnan(blk).any(1)
    mtot[complete, k] = blk[complete].sum(1)
key_mo = np.array([m for _, m in ukeys])
key_yr = np.array([y for y, _ in ukeys])
for m in (6, 7, 8, 9):
    cols = np.where(key_mo == m)[0]
    for i in range(ns):
        vals = mtot[i, cols]
        zero = np.where(vals == 0)[0]
        if len(zero) == 0:
            continue
        others = vals[(vals > 0)]
        if len(others) >= 5 and np.median(others) > 50:
            for z in zero:
                y = key_yr[cols[z]]
                sel = (yr == y) & (mo == m)
                status[i, sel] = SUSPECT
                flags.append((meta.Station_ID[i], f"{y}-{m:02d}", "zero_monsoon_month",
                              0.0, f"median other years {np.median(others):.0f} mm"))

# Flag B: stuck gauge. Five or more consecutive identical positive values.
for i in range(ns):
    v = X[i]
    run = 1
    for t in range(1, nt):
        if v[t] > 0 and v[t] == v[t - 1]:
            run += 1
        else:
            if run >= 5:
                status[i, t - run:t] = SUSPECT
                flags.append((meta.Station_ID[i], str(dates[t - run].date()), "repeated_value",
                              v[t - 1], f"{run} days"))
            run = 1

# Flag C: isolated very large value. >= 200 mm while every observed neighbour within
# 50 km (at least 3 observing) recorded < 20 mm on days D-1 to D+1. Flagged for manual
# review only; values are kept as observed because a local storm can be genuine.
lat, lon = np.radians(meta.Latitude.to_numpy()), np.radians(meta.Longitude.to_numpy())
dlat = lat[:, None] - lat[None, :]
dlon = lon[:, None] - lon[None, :]
a = np.sin(dlat / 2) ** 2 + np.cos(lat[:, None]) * np.cos(lat[None, :]) * np.sin(dlon / 2) ** 2
dist = 2 * 6371 * np.arcsin(np.sqrt(a))
big = np.argwhere(np.nan_to_num(X) >= 200)
n_iso = 0
for i, t in big:
    nb = np.where((dist[i] > 0) & (dist[i] <= 50))[0]
    win = X[nb][:, max(t - 1, 0):t + 2]
    obs_nb = ~np.isnan(win).all(1)
    if obs_nb.sum() >= 3 and np.nanmax(win[obs_nb]) < 20:
        n_iso += 1
        flags.append((meta.Station_ID[i], str(dates[t].date()), "isolated_large_value",
                      X[i, t], f"max of {obs_nb.sum()} neighbours {np.nanmax(win[obs_nb]):.1f} mm"))

fl = pd.DataFrame(flags, columns=["station_id", "date_or_month", "flag", "value_mm", "evidence"])
fl.to_csv(OUT / "qc_flags.csv", index=False)

# Inventory
active = status != OUTSIDE
years = np.unique(yr)
ndays = np.array([(yr == y).sum() for y in years])
good = status == OBS
obs_y = np.stack([good[:, yr == y].sum(1) for y in years], 1)
elig = obs_y / ndays >= 0.95
# eligible year also requires no month with more than three missing days
for j, y in enumerate(years):
    for m in range(1, 13):
        sel = (yr == y) & (mo == m)
        elig[:, j] &= (~good[:, sel]).sum(1) <= 3
inv = meta.copy()
inv["first_obs"] = dates[first].date
inv["last_obs"] = dates[last].date
inv["active_days"] = active.sum(1)
inv["observed_days"] = good.sum(1)
inv["active_completeness"] = (good.sum(1) / active.sum(1)).round(3)
inv["suspect_days"] = (status == SUSPECT).sum(1)
inv["eligible_years"] = elig.sum(1)
inv["eligible_1981_1990"] = elig[:, (years >= 1981) & (years <= 1990)].sum(1)
inv["eligible_2014_2023"] = elig[:, (years >= 2014) & (years <= 2023)].sum(1)
inv["zero_fraction"] = np.round(np.nanmean(np.where(good, X, np.nan) == 0, 1), 3)
inv["max_mm"] = np.nanmax(np.where(good, X, np.nan), 1)
inv.to_csv(OUT / "station_inventory.csv", index=False)

np.savez_compressed(OUT / "daily_qc.npz", X=X, status=status, dates=dates.to_numpy(),
                    station_id=meta.Station_ID.to_numpy(), lat=meta.Latitude.to_numpy(),
                    lon=meta.Longitude.to_numpy(), elev=meta.Elevation.to_numpy(),
                    eligible=elig, years=years)

print("stations", ns)
print("status counts", {k: int((status == c).sum()) for k, c in
                        [("observed", OBS), ("missing", MISS), ("outside", OUTSIDE), ("suspect", SUSPECT)]})
print("flags by type", fl.flag.value_counts().to_dict())
print("isolated large values", n_iso)
print("missing fraction within operating period %.3f" % ((status == MISS).sum() / active.sum()))
core = (inv.eligible_years >= 35) & (inv.eligible_1981_1990 >= 8) & (inv.eligible_2014_2023 >= 8)
print("long-record core candidates (>=35 eligible yrs, >=8 in each endpoint decade):", int(core.sum()))
print("eligible stations per year:", dict(zip(years.tolist(), elig.sum(0).tolist())))
