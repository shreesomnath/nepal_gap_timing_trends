# Code Developer: Somnath Luitel (somnath.luitel416@topper.wku.edu)
"""Step 41: (a) do RHtests breaks cluster in the same years?  (b) flagged vs unflagged stations compared WITHIN region x elevation strata.

(a) RHtests significant break dates (1981-2017, 219 trend stations) are counted by year. Null: each station's breaks fall in random years
    within 1981-2017 (same number per station). Statistic: the largest annual count and the sum of squared annual counts; one-sided P by 5000 draws.
    Also counted for the 1990s-2000s transition years (2001-2005) reported by the archive history if any.
(b) Station Sen trends (eligible years, % per decade; step 17 output) of flagged minus unflagged stations, averaged over region x elevation
    strata (East/Central/West x <500, 500-1500, >1500 m; strata needing at least 3 stations of each kind), weighted by the smaller group size,
    with a station bootstrap interval. Same for the paired shift (corrected - eligible) among affected stations.
Output: outputs/homogeneity_clustering_pre2018.csv, outputs/flagged_vs_unflagged_strata_pre2018.csv
"""
import numpy as np
import pandas as pd
from pathlib import Path

rng = np.random.default_rng(41)
OUT = Path(__file__).resolve().parents[1] / "outputs"
rh = pd.read_csv(OUT / "rhtests_pre2018.csv", dtype={"station_id": str})
st = pd.read_csv(OUT / "station_trends_pre2018.csv", dtype={"station_id": str})
pv = pd.read_csv(OUT / "station_province.csv", dtype={"station_id": str})
stations = st.station_id.unique()
rh = rh[rh.station_id.isin(stations)]
Y0, Y1 = 1981, 2017
yrs = np.arange(Y0, Y1 + 1)

# (a) clustering of break years
dates = []
per_station = {}
for _, r in rh.iterrows():
    ds = [int(str(x)[:4]) for x in str(r.significant_dates).split(";") if str(x) not in ("nan", "")]
    ds = [y for y in ds if Y0 <= y <= Y1]
    if ds:
        per_station[r.station_id] = ds
        dates += ds
obs_counts = np.array([dates.count(y) for y in yrs])
obs_max, obs_ss = obs_counts.max(), (obs_counts ** 2).sum()
nmax, nss = [], []
for _ in range(5000):
    c = np.zeros(len(yrs))
    for s, ds in per_station.items():
        for y in rng.choice(yrs, len(ds), replace=True):
            c[y - Y0] += 1
    nmax.append(c.max()); nss.append((c ** 2).sum())
rows = [("breaks", len(dates), len(per_station), int(obs_max), int(yrs[obs_counts.argmax()]), float((np.array(nmax) >= obs_max).mean()),
         float((np.array(nss) >= obs_ss).mean()))]
A = pd.DataFrame(rows, columns=["what", "n_breaks", "n_stations", "max_annual_count", "year_of_max", "p_max", "p_dispersion"])
cnt = pd.DataFrame({"year": yrs, "count": obs_counts})
A.to_csv(OUT / "homogeneity_clustering_pre2018.csv", index=False)
cnt.to_csv(OUT / "homogeneity_break_counts_by_year_pre2018.csv", index=False)
print(A.round(3).to_string(index=False)); print(cnt[cnt["count"] >= 8].to_string(index=False))

# (b) within-stratum comparison
elev = pv.set_index("station_id").elev
prov = pv.set_index("station_id").province
reg = prov.map(lambda p: "West" if p in {"Sudur Pashchim", "Karnali", "Lumbini"} else "Central" if p == "Gandaki" else "East")
band = pd.cut(elev, [-1, 500, 1500, 9000], labels=["<500", "500-1500", ">1500"])
flag = rh.set_index("station_id").rhtests_flag
rows = []
for idx in ("Rx1day", "Rx5day", "PRCPTOT", "R95pTOT"):
    s = st[st["index"] == idx].set_index("station_id")
    df = pd.DataFrame({"R1": s.R1, "shift": s.Rcorr - s.R1, "reg": reg.reindex(s.index), "band": band.reindex(s.index), "flag": flag.reindex(s.index)}).dropna(subset=["R1", "reg", "band", "flag"])
    for var in ("R1", "shift"):
        d_ = df.dropna(subset=[var]) if var == "R1" else df.dropna(subset=[var])
        if var == "shift":
            d_ = d_[d_["shift"] != 0]

        def est(dd):
            num = den = 0.0
            for (r_, b_), g in dd.groupby(["reg", "band"], observed=True):
                a, b = g[g.flag][var], g[~g.flag][var]
                if len(a) >= 3 and len(b) >= 3:
                    w = min(len(a), len(b)); num += w * (a.median() - b.median()); den += w
            return num / den if den else np.nan
        pt = est(d_)
        bs = [est(d_.sample(len(d_), replace=True, random_state=int(rng.integers(1e9)))) for _ in range(1000)]
        rows.append((idx, var, int(d_.flag.sum()), int((~d_.flag).sum()), pt, *np.nanpercentile(bs, [2.5, 97.5]),
                     d_[d_.flag][var].median(), d_[~d_.flag][var].median()))
B = pd.DataFrame(rows, columns=["index", "variable", "n_flagged", "n_unflagged", "within_stratum_diff", "lo95", "hi95", "median_flagged", "median_unflagged"])
B.to_csv(OUT / "flagged_vs_unflagged_strata_pre2018.csv", index=False)
print(B.round(2).to_string(index=False))
