# Code Developer: Somnath Luitel (somnath.luitel416@topper.wku.edu)
"""Step 17: the step 12 analysis restricted to 1981-2017 (before the 2018-2020 archive gap).

Differences from step 12: period 1981-2017; stations are those with at least 26 eligible
years (70% of the 37-year period) in 1981-2017, with no requirement for data after 2014,
so stations lost in the outage are kept; adds Benjamini-Hochberg counts (q = 0.10),
share of declining stations and regional medians per treatment.
Outputs: outputs/*_pre2018.csv.

Original step 12 docstring follows.

Step 12: all four indices through the paired bootstrap and field-significance tests.

Same design as step 10 (strict / all years uncorrected / all years corrected; 1000
whole-year bootstrap replicates with fresh gap-ratio draws; 1000 block permutations of
whole years), now for Rx1day, Rx5day, PRCPTOT and R95pTOT, all 184 stations.
Also reports the lag-1 autocorrelation of the strict station series, because the
permutation null breaks serial correlation and understates variance if it is large.
Outputs: outputs/bootstrap_shift_all.csv, outputs/field_significance_all.csv,
         outputs/autocorrelation.csv
"""
import numpy as np
import pandas as pd
from scipy import stats
from numpy.lib.stride_tricks import sliding_window_view
from pathlib import Path

rng = np.random.default_rng(99)
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
d = np.load(OUT / "daily_qc.npz", allow_pickle=True)
X, status, elig, years = d["X"], d["status"], d["eligible"], d["years"]
sid = d["station_id"]
dates = pd.to_datetime(d["dates"]); yr = dates.year.to_numpy()
V = np.where(status == 0, X, np.nan)
p95 = pd.read_csv(OUT / "p95_baseline.csv", dtype={"station_id": str}).p95_1981_2010.to_numpy()
tr = pd.read_csv(OUT / "trend_by_rule.csv", dtype={"station_id": str})
_ys = list(years)
_m = [_ys.index(y) for y in range(1981, 2018)]
used = [s for k, s in enumerate(sid) if elig[k, _m].sum() >= 26]
pos = {s: k for k, s in enumerate(sid)}
PER = (years >= 1981) & (years <= 2017)
yrs = years[PER]; T = len(yrs); S = len(used)
NAMES = ["Rx1day", "Rx5day", "PRCPTOT", "R95pTOT"]
_prov = pd.read_csv(OUT / "station_province.csv", dtype={"station_id": str}).set_index("station_id")
REG = np.array(["West" if _prov.loc[s, "province"] in {"Sudur Pashchim", "Karnali", "Lumbini"} else "Central" if _prov.loc[s, "province"] == "Gandaki" else "East" for s in used])
rows_r = []
STN = []
_lon = d["lon"].astype(float)[[list(sid).index(s) for s in used]]
ZONE = np.where(_lon < 83, "West", np.where(_lon < 86, "Central", "East"))   # Karki et al. (2017) longitude zones
def bh(p, q=0.10):
    ok = np.isfinite(p); pv = p[ok]; o = np.argsort(pv); m = len(pv)
    thr = (np.arange(1, m + 1) / m) * q
    passed = pv[o] <= thr
    k = np.where(passed)[0].max() + 1 if passed.any() else 0
    sig = np.zeros(m, bool); sig[o[:k]] = True
    out = np.zeros(len(p), bool); out[np.where(ok)[0][sig]] = True
    return out



def yv(i, y):
    v = V[i, yr == y]
    return np.delete(v, 59) if v.size == 366 else v


def ind(a, p):
    """Four indices for an array (..., 365). Rx5day needs all five days observed."""
    with np.errstate(all="ignore"):
        rx1 = np.nanmax(a, -1)
        w = sliding_window_view(a, 5, axis=-1)
        s5 = np.where(np.isfinite(w).all(-1), w.sum(-1), np.nan)
        rx5 = np.nanmax(s5, -1)
        wet = np.where(a >= 1, a, 0)
        pt = np.nansum(wet, -1)
        r95 = np.nansum(np.where(a > p, a, 0), -1) if np.isfinite(p) else np.full(pt.shape, np.nan)
    return np.stack([rx1, rx5, pt, r95])


RAW = np.full((4, S, T), np.nan)
COMP = np.zeros((S, T), bool)
RATIOS = {}
for a_, s in enumerate(used):
    i = pos[s]
    cy = [y for j, y in enumerate(years) if PER[j] and elig[i, j]]
    comp = np.array([yv(i, y) for y in cy])
    base = ind(comp, p95[i])                       # (4, n_complete)
    for t, y in enumerate(yrs):
        j = int(np.where(years == y)[0][0])
        if (status[i, yr == y] == 2).all():
            continue
        v = yv(i, y)
        if not np.isfinite(v).any():
            continue
        RAW[:, a_, t] = ind(v, p95[i])
        if elig[i, j]:
            COMP[a_, t] = True
        else:
            m = ~np.isfinite(v)
            cm = comp.copy(); cm[:, m] = np.nan
            with np.errstate(all="ignore"):
                RATIOS[(a_, t)] = ind(cm, p95[i]) / base
RAW[2][RAW[2] == 0] = np.nan                          # zero annual total = suspect
I, J = np.triu_indices(T, 1)


def corrected(k, mode):
    M = RAW[k].copy()
    for (a_, t), r in RATIOS.items():
        rr = r[k][np.isfinite(r[k]) & (r[k] > 0)]
        if rr.size == 0:
            M[a_, t] = np.nan
            continue
        g = np.median(rr) if mode == "median" else rr[rng.integers(rr.size)]
        M[a_, t] = M[a_, t] / g if g >= 0.2 else np.nan
    return M


def sen_pct(M, tt):
    dy = M[:, J] - M[:, I]
    dx = (tt[J] - tt[I])[None, :]
    with np.errstate(all="ignore"):
        sl = np.where(dx != 0, dy / dx, np.nan)     # resampled years are unsorted: keep pairs with either sign of dx (audit fix)
        med = np.nanmedian(M, 1)
        out = np.nanmedian(sl, 1) * 10 / med * 100
        return np.where(med > 0, out, np.nan)      # % of a zero median is undefined


def mk_local(M, tt):
    dy = np.sign(M[:, J] - M[:, I])
    valid = np.isfinite(dy) & ((tt[J] - tt[I]) > 0)[None, :]
    s = np.nansum(np.where(valid, dy, 0), 1)
    n = np.isfinite(M).sum(1)
    var = n * (n - 1) * (2 * n + 5) / 18
    z = np.where(s != 0, (s - np.sign(s)) / np.sqrt(var), 0)
    return 2 * (1 - stats.norm.cdf(np.abs(z))), s


t0 = yrs.astype(float)
blocks = [np.arange(s, min(s + 3, T)) for s in range(0, T, 3)]
rows_b, rows_f, rows_a = [], [], []
for k, name in enumerate(NAMES):
    R1 = np.where(COMP, RAW[k], np.nan)
    R4 = RAW[k]
    point = corrected(k, "median")
    affected = (~COMP & np.isfinite(RAW[k])).any(1)
    # lag-1 autocorrelation of strict series
    ac = []
    for r in R1:
        x = r[np.isfinite(r)]
        if len(x) >= 20:
            ac.append(np.corrcoef(x[:-1], x[1:])[0, 1])
    rows_a.append((name, len(ac), np.median(ac), np.percentile(ac, 90)))
    shifts = {"R4-R1": [], "Rcorr-R1": []}
    for b in range(1000):
        idx = rng.integers(T, size=T); tt = t0[idx]
        cb = corrected(k, "draw")
        e1, e4, ec = (sen_pct(M[:, idx], tt) for M in (R1, R4, cb))
        shifts["R4-R1"].append(np.nanmean((e4 - e1)[affected]))
        shifts["Rcorr-R1"].append(np.nanmean((ec - e1)[affected]))
    pe = {"R1": sen_pct(R1, t0), "R4": sen_pct(R4, t0), "Rcorr": sen_pct(point, t0)}
    STN.append(pd.DataFrame({"station_id": used, "index": name, "R1": pe["R1"], "R4": pe["R4"], "Rcorr": pe["Rcorr"]}))
    for q, (a1, b1) in {"R4-R1": ("R4", "R1"), "Rcorr-R1": ("Rcorr", "R1")}.items():
        lo, hi = np.percentile(shifts[q], [2.5, 97.5])
        rows_b.append((name, int(affected.sum()), q, np.nanmean((pe[a1] - pe[b1])[affected]), lo, hi))
    for tname, M in (("R1", R1), ("R4", R4), ("R4_corr", point)):
        p, s_ = mk_local(M, t0)
        on, op = int(((p < 0.05) & (s_ < 0)).sum()), int(((p < 0.05) & (s_ > 0)).sum())
        nn, npos = [], []
        for _ in range(1000):
            order = np.concatenate([blocks[j] for j in rng.permutation(len(blocks))])
            pp, ss = mk_local(M[:, order], t0)
            nn.append(((pp < 0.05) & (ss < 0)).sum()); npos.append(((pp < 0.05) & (ss > 0)).sum())
        nn, npos = np.array(nn), np.array(npos)
        _pe = pe["R1" if tname == "R1" else ("R4" if tname == "R4" else "Rcorr")]
        _sig = bh(p)
        rows_r.append((name, tname, int(np.isfinite(_pe).sum()), np.nanmedian(_pe), float((_pe < 0)[np.isfinite(_pe)].mean()),
                       int((_sig & (s_ < 0)).sum()), int((_sig & (s_ > 0)).sum()),
                       *[np.nanmedian(_pe[REG == r]) for r in ("East", "Central", "West")],
                       *[np.nanmedian(_pe[ZONE == r]) for r in ("East", "Central", "West")]))
        rows_f.append((name, tname, on, (nn >= on).mean(), np.percentile(nn, 95),
                       op, (npos >= op).mean(), np.nanmedian(pe["R1" if tname == "R1" else ("R4" if tname == "R4" else "Rcorr")])))

b = pd.DataFrame(rows_b, columns=["index", "n_affected", "quantity", "point", "ci_lo", "ci_hi"])
f = pd.DataFrame(rows_f, columns=["index", "treatment", "obs_local_neg", "p_field_neg", "null95_neg",
                                  "obs_local_pos", "p_field_pos", "median_trend_pct_dec"])
a = pd.DataFrame(rows_a, columns=["index", "n_stations", "median_lag1", "p90_lag1"])
b.to_csv(OUT / "bootstrap_shift_pre2018.csv", index=False)
f.to_csv(OUT / "field_significance_pre2018.csv", index=False)
a.to_csv(OUT / "autocorrelation_pre2018.csv", index=False)
r_ = pd.DataFrame(rows_r, columns=["index", "treatment", "n_stations", "median_trend", "share_declining", "fdr_declines", "fdr_increases", "med_East", "med_Central", "med_West",
                                  "zone_East", "zone_Central", "zone_West"])
r_.to_csv(OUT / "trend_summary_pre2018.csv", index=False)
_st = pd.concat(STN)
_st["lon"] = np.tile(_lon, 4); _st["elev"] = np.tile(d["elev"].astype(float)[[list(sid).index(s) for s in used]], 4)
_st["zone"] = np.tile(ZONE, 4); _st["province_region"] = np.tile(REG, 4)
_st.to_csv(OUT / "station_trends_pre2018.csv", index=False)
print("zone counts:", {r: int((ZONE == r).sum()) for r in ("East", "Central", "West")})
print("stations:", len(used), {r: int((REG == r).sum()) for r in ("East", "Central", "West")})
print(r_.round(2).to_string(index=False)); print()
print(a.round(3).to_string(index=False)); print()
print(b.round(2).to_string(index=False)); print()
print(f.round(3).to_string(index=False))
