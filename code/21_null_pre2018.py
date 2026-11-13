# Code Developer: Somnath Luitel (somnath.luitel416@topper.wku.edu)
"""Step 21 (null experiment of step 6 restricted to 1981-2017, stations with at least 26 eligible
years in that period, at least 22 usable years per series; output outputs/null_trends_pre2018.csv).

Step 6: (a) can the archive's gap pattern alone create significant trends?
            (b) does the east-west contrast survive completeness rules and a fixed network?

(a) Null experiment. For each station used in step 3 (R1 rule), the complete years are
resampled at random into every year 1981-2023 in which the station operated, which
removes any real trend. Years that were partly observed in the archive then receive
that station's actual missingness pattern for that year, so the timing and shape of
gaps are real while the climate signal is absent. Trends are estimated under R1
(complete years only) and R4 (every year with data). Repeated 100 times.

(b) Province is assigned by point-in-polygon on the official local-unit boundary after
transforming station coordinates (assumed WGS84) to the boundary's Everest 1830 datum.

Outputs: outputs/null_trends.csv, outputs/regional_trends.csv, gis/stations_wgs84.shp
"""
import numpy as np
import pandas as pd
# import shapefile (not needed here)
from scipy import stats
from pathlib import Path

rng = np.random.default_rng(11)
ROOT = Path(__file__).resolve().parents[1]
OUT, GIS = ROOT / "outputs", ROOT / "gis"
d = np.load(OUT / "daily_qc.npz", allow_pickle=True)
X, status, elig, years = d["X"], d["status"], d["eligible"], d["years"]
sid = d["station_id"]
dates = pd.to_datetime(d["dates"])
yr = dates.year.to_numpy()
V = np.where(status == 0, X, np.nan)
p95 = pd.read_csv(OUT / "p95_baseline.csv", dtype={"station_id": str}).p95_1981_2010.to_numpy()
tr = pd.read_csv(OUT / "trend_by_rule.csv", dtype={"station_id": str})
_m = [list(years).index(y) for y in range(1981, 2018)]
used = set(sid[elig[:, _m].sum(1) >= 26].astype(str))
PER = (years >= 1981) & (years <= 2017)


def yv(i, y):
    v = V[i, yr == y]
    return np.delete(v, 59) if v.size == 366 else v


def idx(v, p):
    wet = v[v >= 1]
    return np.nanmax(v), wet.sum(), wet[wet > p].sum()


def mk_p(y):
    n = len(y)
    s = np.sum(np.sign(y[None, :] - y[:, None])[np.triu_indices(n, 1)])
    var = n * (n - 1) * (2 * n + 5) / 18
    z = (s - np.sign(s)) / np.sqrt(var) if s else 0.0
    return 2 * (1 - stats.norm.cdf(abs(z)))


def fdr(p, q=0.10):
    p = np.asarray(p); o = np.argsort(p)
    ok = p[o] <= q * np.arange(1, len(p) + 1) / len(p)
    k = np.where(ok)[0].max() + 1 if ok.any() else 0
    s = np.zeros(len(p), bool); s[o[:k]] = True
    return s


# ---------------- (a) null experiment ----------------
st_idx = [i for i in range(len(sid)) if str(sid[i]) in used and np.isfinite(p95[i])]
info = []
for i in st_idx:
    comp, part = [], {}
    for j, y in enumerate(years):
        if not PER[j] or (status[i, yr == y] == 2).all():
            continue
        v = yv(i, y)
        if elig[i, j]:
            comp.append(y)
        elif np.isfinite(v).any():
            part[y] = ~np.isfinite(v)
    pool = np.array([yv(i, y) for y in comp])
    info.append((i, comp, part, pool))

NAMES = ["Rx1day", "PRCPTOT", "R95pTOT"]
rows = []
for rep in range(100):
    res = {(r, n): [] for r in ("R1", "R4") for n in NAMES}
    for i, comp, part, pool in info:
        allyears = sorted(comp + list(part))
        draw = pool[rng.integers(len(pool), size=len(allyears))]
        vals, isc = [], []
        for y, v in zip(allyears, draw):
            v = v.copy()
            if y in part:
                v[part[y]] = np.nan
            vals.append(idx(v, p95[i])); isc.append(y not in part)
        vals, isc, t = np.array(vals), np.array(isc), np.array(allyears, float)
        for k, n in enumerate(NAMES):
            for r, m in (("R1", isc), ("R4", np.ones_like(isc))):
                if m.sum() >= 22:
                    yy = vals[m, k]
                    res[(r, n)].append((stats.theilslopes(yy, t[m])[0] * 10 / np.median(yy) * 100 if np.median(yy) > 0 else 0.0,
                                        mk_p(yy)))
    for (r, n), lst in res.items():
        a = np.array(lst)
        sig = fdr(a[:, 1])
        rows.append((rep, r, n, len(a), np.median(a[:, 0]), (a[:, 0] < 0).mean(),
                     int(((a[:, 1] < 0.05) & (a[:, 0] < 0)).sum()),
                     int(((a[:, 1] < 0.05) & (a[:, 0] > 0)).sum()),
                     int((sig & (a[:, 0] < 0)).sum())))
nul = pd.DataFrame(rows, columns=["rep", "rule", "index", "n", "median_pct_dec", "share_neg",
                                  "sig_neg_local", "sig_pos_local", "sig_neg_fdr"])
nul.to_csv(OUT / "null_trends_pre2018.csv", index=False)
print("NULL experiment (no real trend), mean over 100 replicates:")
print(nul.groupby(["index", "rule"])[["n", "median_pct_dec", "share_neg", "sig_neg_local",
                                      "sig_pos_local", "sig_neg_fdr"]].mean().round(2).to_string())
q = nul.groupby(["index", "rule"]).sig_neg_fdr.quantile(0.95)
print("95th percentile of FDR-significant negative trends under null:\n", q.to_string())

