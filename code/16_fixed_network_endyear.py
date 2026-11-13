# Code Developer: Somnath Luitel (somnath.luitel416@topper.wku.edu)
"""Step 16: fixed-network and end-year sensitivity of the strict station trends.

Question. The 2018-2020 half-network outage removes mostly eastern and central stations.
Do the regional trend conclusions change (a) when the period ends in 2017, before the
outage, (b) when only stations that kept reporting through 2018-2019 are used, or
(c) when the dropped-out stations are analysed on their own?

Design. Strict series (eligible years only, as in step 3) for the 184 stations of the main
trend set, Sen slope in % of the station median per decade, Mann-Kendall p. Groups are
all / kept (observed on at least 80% of days in each of 2018 and 2019) / dropped (observed
on less than 20% of days in 2018-2019 after reporting in 2017). Periods: 1981-2017 and
1981-2023. A station needs at least 25 eligible years inside the period. Regional
medians use the regions of step 6. Central-minus-East differences have 95% intervals from
resampling stations within region (stations treated as independent, so intervals are
narrow). Gauge data are never modified.
Outputs: outputs/fixed_network_endyear.csv, outputs/fixed_network_paired.csv
"""
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path

rng = np.random.default_rng(16)
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
d = np.load(OUT / "daily_qc.npz", allow_pickle=True)
elig, years, sid = d["eligible"], d["years"], d["station_id"].astype(str)
status = d["status"]
yr = pd.to_datetime(d["dates"]).year.to_numpy()
ai = pd.read_csv(OUT / "annual_indices.csv", dtype={"station_id": str})
tr = pd.read_csv(OUT / "trend_by_rule.csv", dtype={"station_id": str})
used = sorted(set(tr[(tr.rule == "R1_strict") & (tr["index"] == "Rx1day")].station_id))
pos = {s: k for k, s in enumerate(sid)}
prov = pd.read_csv(OUT / "station_province.csv", dtype={"station_id": str}).set_index("station_id")
WEST = {"Sudur Pashchim", "Karnali", "Lumbini"}
region = {s: ("West" if prov.loc[s, "province"] in WEST else "Central" if prov.loc[s, "province"] == "Gandaki" else "East") for s in used}

obs = status == 0
def frac(i, y):
    m = yr == y
    return obs[i, m].mean()
group = {}
for s in used:
    i = pos[s]
    f18, f19 = frac(i, 2018), frac(i, 2019)
    if f18 >= 0.8 and f19 >= 0.8:
        group[s] = "kept"
    elif obs[i, (yr == 2018) | (yr == 2019)].mean() < 0.2 and obs[i, yr == 2017].any():
        group[s] = "dropped"
    else:
        group[s] = "other"

INDICES = ["Rx1day", "Rx5day", "PRCPTOT", "R95pTOT"]
ai = ai.set_index(["station_id", "year"])


def trend(s, idx, y0, y1):
    i = pos[s]
    ys = [int(y) for j, y in enumerate(years) if y0 <= y <= y1 and elig[i, j]]
    if len(ys) < 25:
        return np.nan
    v = np.array([ai.loc[(s, y), idx] if (s, y) in ai.index else np.nan for y in ys], float)
    ok = np.isfinite(v)
    if ok.sum() < 25 or np.nanmedian(v[ok]) <= 0:
        return np.nan
    sl = stats.theilslopes(v[ok], np.array(ys)[ok])[0]
    return 100 * 10 * sl / np.nanmedian(v[ok])


rows, paired = [], []
for idx in INDICES:
    T = {}
    for per in [(1981, 2017), (1981, 2023)]:
        T[per] = {s: trend(s, idx, *per) for s in used}
    for per in T:
        for g in ["all", "kept", "dropped"]:
            ss = [s for s in used if (g == "all" or group[s] == g) and np.isfinite(T[per][s])]
            vals = np.array([T[per][s] for s in ss])
            r = {"index": idx, "period": f"{per[0]}-{per[1]}", "group": g, "n": len(ss),
                 "median": np.median(vals) if len(ss) else np.nan,
                 "share_declining": (vals < 0).mean() if len(ss) else np.nan}
            by = {k: np.array([T[per][s] for s in ss if region[s] == k]) for k in ["East", "Central", "West"]}
            for k, v in by.items():
                r[f"n_{k}"] = len(v); r[f"med_{k}"] = np.median(v) if len(v) else np.nan
            if len(by["East"]) >= 5 and len(by["Central"]) >= 5:
                bs = [np.median(rng.choice(by["Central"], len(by["Central"]))) -
                      np.median(rng.choice(by["East"], len(by["East"]))) for _ in range(2000)]
                r["C_minus_E"] = np.median(by["Central"]) - np.median(by["East"])
                r["CmE_lo"], r["CmE_hi"] = np.percentile(bs, [2.5, 97.5])
            rows.append(r)
    # paired change in each station's trend when 2018-2023 is added
    for g in ["kept", "dropped", "all"]:
        ss = [s for s in used if (g == "all" or group[s] == g) and np.isfinite(T[(1981, 2017)][s]) and np.isfinite(T[(1981, 2023)][s])]
        dv = np.array([T[(1981, 2023)][s] - T[(1981, 2017)][s] for s in ss])
        if len(dv):
            bs = [np.mean(rng.choice(dv, len(dv))) for _ in range(2000)]
            paired.append({"index": idx, "group": g, "n": len(dv), "mean_change_2017_to_2023": dv.mean(),
                           "lo": np.percentile(bs, 2.5), "hi": np.percentile(bs, 97.5)})

res = pd.DataFrame(rows); pr = pd.DataFrame(paired)
res.to_csv(OUT / "fixed_network_endyear.csv", index=False)
pr.to_csv(OUT / "fixed_network_paired.csv", index=False)
print(pd.Series(group).value_counts().to_string())
print(pd.crosstab(pd.Series(region), pd.Series(group)).to_string())
pd.set_option("display.width", 250)
print(res.round(2).to_string())
print(pr.round(2).to_string())
