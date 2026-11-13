# Code Developer: Somnath Luitel (somnath.luitel416@topper.wku.edu)
"""Step 29 (PIT test with a local reference: only eligible years within 6 years of the target year form the
reference distribution, so a long-term trend cannot make late partly observed years look low; at least 8 reference
years are needed; output outputs/partial_year_pit_local.csv).

Step 17b (step 7 restricted to 1981-2017; output outputs/partial_year_pit_pre2018.csv).

Step 7: are partly observed years lower than their gaps alone can explain?

For every partly observed station-year (1981-2023) at a station with >= 15 complete
years, apply that year's own missingness mask to each of the station's complete years.
This gives the distribution of the index we would expect from those gaps if nothing
else were different. The observed value's position in that distribution (PIT) should be
uniform on (0, 1) if missingness is uninformative and the year is climatologically
ordinary.

A year can also be genuinely dry. To separate this, the same calculation is done for
neighbouring stations (within 50 km, complete in the same year, same mask applied to
their own complete years). The station-specific deficit is target PIT minus mean
neighbour PIT.

Outputs: outputs/partial_year_pit.csv, printed summary.
"""
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
d = np.load(OUT / "daily_qc.npz", allow_pickle=True)
X, status, elig, years = d["X"], d["status"], d["eligible"], d["years"]
sid, lat, lon = d["station_id"], np.radians(d["lat"].astype(float)), np.radians(d["lon"].astype(float))
dates = pd.to_datetime(d["dates"])
yr = dates.year.to_numpy()
V = np.where(status == 0, X, np.nan)
reg = pd.read_csv(OUT / "station_province.csv", dtype={"station_id": str})
reg["region"] = np.where(reg.province.isin(["Sudur Pashchim", "Karnali", "Lumbini"]), "West",
                         np.where(reg.province == "Gandaki", "Central", "East"))
PER = (years >= 1981) & (years <= 2017)

a = (np.sin((lat[:, None] - lat[None, :]) / 2) ** 2 +
     np.cos(lat[:, None]) * np.cos(lat[None, :]) * np.sin((lon[:, None] - lon[None, :]) / 2) ** 2)
dist = 2 * 6371 * np.arcsin(np.sqrt(a))

cache = {}


def full_years(i):
    if i not in cache:
        ys = [y for j, y in enumerate(years) if PER[j] and elig[i, j]]
        arr = []
        for y in ys:
            v = V[i, yr == y]
            arr.append(np.delete(v, 59) if v.size == 366 else v)
        cache[i] = (ys, np.array(arr) if arr else np.zeros((0, 365)))
    return cache[i]


def stats_of(arr):
    """Rx1day and PRCPTOT for each row (NaN ignored)."""
    with np.errstate(all="ignore"):
        rx = np.nanmax(arr, 1)
        pt = np.nansum(np.where(arr >= 1, arr, 0), 1)
    return rx, pt


def pit(value, ref):
    ref = ref[np.isfinite(ref)]
    return ((ref < value).sum() + 0.5 * (ref == value).sum()) / len(ref)


rows = []
for i in range(len(sid)):
    ys_i, comp_i = full_years(i)
    if len(ys_i) < 15:
        continue
    for j, y in enumerate(years):
        if not PER[j] or elig[i, j]:
            continue
        v = V[i, yr == y]
        v = np.delete(v, 59) if v.size == 366 else v
        m = ~np.isfinite(v)
        f = 1 - m.mean()
        if f == 0 or f >= 1 or (status[i, yr == y] == 2).any():
            continue
        rx_o, pt_o = stats_of(v[None, :])
        others = [k for k, yy in enumerate(ys_i) if 0 < abs(yy - y) <= 6]
        if len(others) < 8:
            continue
        masked = comp_i[others].copy(); masked[:, m] = np.nan
        rx_e, pt_e = stats_of(masked)
        p_rx, p_pt = pit(rx_o[0], rx_e), pit(pt_o[0], pt_e)
        # neighbours complete in year y
        nb_rx, nb_pt = [], []
        for k in np.where((dist[i] > 0) & (dist[i] <= 50))[0]:
            if not elig[k, j]:
                continue
            ys_k, comp_k = full_years(k)
            if len(ys_k) < 15:
                continue
            vk = V[k, yr == y]; vk = np.delete(vk, 59) if vk.size == 366 else vk
            vk = vk.copy(); vk[m] = np.nan
            rk, pk = stats_of(vk[None, :])
            sel_k = [q for q, yy in enumerate(ys_k) if 0 < abs(yy - y) <= 6]
            if len(sel_k) < 8:
                continue
            ck = comp_k[sel_k].copy(); ck[:, m] = np.nan
            rke, pke = stats_of(ck)
            nb_rx.append(pit(rk[0], rke)); nb_pt.append(pit(pk[0], pke))
        rows.append((sid[i], y, round(f, 3), p_rx, p_pt, len(nb_rx),
                     np.mean(nb_rx) if nb_rx else np.nan, np.mean(nb_pt) if nb_pt else np.nan))

res = pd.DataFrame(rows, columns=["station_id", "year", "obs_frac", "pit_rx1day", "pit_prcptot",
                                  "n_neighbours", "nb_pit_rx1day", "nb_pit_prcptot"])
res = res.merge(reg[["station_id", "region", "province"]], on="station_id")
res.to_csv(OUT / "partial_year_pit_local.csv", index=False)

print("partial station-years tested:", len(res))
for c in ("pit_rx1day", "pit_prcptot"):
    x = res[c].dropna()
    print(f"{c}: mean {x.mean():.3f} (uniform 0.5), share below 0.1 {np.mean(x < 0.1):.3f} (0.1), "
          f"KS p = {stats.kstest(x, 'uniform').pvalue:.2g}")
hn = res[res.n_neighbours >= 2]
print(f"\nwith >= 2 complete neighbours (n={len(hn)}):")
for c in ("rx1day", "prcptot"):
    diff = hn[f"pit_{c}"] - hn[f"nb_pit_{c}"]
    w = stats.wilcoxon(diff)
    print(f"{c}: target PIT {hn[f'pit_{c}'].mean():.3f}, neighbour PIT {hn[f'nb_pit_{c}'].mean():.3f}, "
          f"median difference {diff.median():.3f}, Wilcoxon p = {w.pvalue:.2g}")
res["frac_bin"] = pd.cut(res.obs_frac, [0, 0.5, 0.8, 0.9, 1.0])
print("\nmean PIT by observed fraction:")
print(res.groupby("frac_bin", observed=True)[["pit_rx1day", "pit_prcptot", "nb_pit_prcptot"]]
      .agg(["mean", "count"]).round(3).to_string())
print("\nmean PIT by period:")
res["period"] = pd.cut(res.year, [1980, 1990, 2000, 2010, 2017])
print(res.groupby("period", observed=True)[["pit_rx1day", "pit_prcptot", "nb_pit_prcptot"]]
      .mean().round(3).to_string())
print("\nmean PIT by region:")
print(res.groupby("region")[["pit_rx1day", "pit_prcptot", "nb_pit_prcptot"]].mean().round(3).to_string())
