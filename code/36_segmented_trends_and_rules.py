# Code Developer: Somnath Luitel (somnath.luitel416@topper.wku.edu)
"""Step 36: (a) break-aware trends and (b) robustness of the paired shift to the station-inclusion rule (1981-2017).

(a) For each station, the RHtests_dlyPrcp changepoints listed as significant (outputs/rhtests_pre2018.csv) split the record into segments.
    The Sen slope is then the median of year-pair slopes taken only inside one segment (years containing a changepoint are left out), so a
    step change between segments cannot create or hide a trend. A station needs at least 8 within-segment pairs, otherwise it has no
    segmented trend. Reported for eligible years only (R1), all years uncorrected (R4) and all years corrected (R4corr).
(b) Mean paired shift (R4 - R1 and R4corr - R1, % per decade) over stations with at least one partly observed year, for stations having at
    least 22, 26, 30 or 32 eligible years in 1981-2017, with 95% station-bootstrap intervals.
Outputs: outputs/segmented_trends_pre2018.csv (summary), outputs/shift_by_station_rule_pre2018.csv
"""
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from pathlib import Path

rng = np.random.default_rng(36)
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
d = np.load(OUT / "daily_qc.npz", allow_pickle=True)
X, status, elig, years, sid = d["X"], d["status"], d["eligible"], d["years"], d["station_id"].astype(str)
dates = pd.to_datetime(d["dates"]); yr = dates.year.to_numpy()
V = np.where(status == 0, X, np.nan)
p95 = pd.read_csv(OUT / "p95_baseline.csv", dtype={"station_id": str}).p95_1981_2010.to_numpy()
PER = (years >= 1981) & (years <= 2017)
yrs = years[PER]; T = len(yrs); S = V.shape[0]
m_idx = [list(years).index(y) for y in range(1981, 2018)]
n_elig = elig[:, m_idx].sum(1)
NAMES = ["Rx1day", "Rx5day", "PRCPTOT", "R95pTOT"]
RH = pd.read_csv(OUT / "rhtests_pre2018.csv", dtype={"station_id": str}, keep_default_na=False).set_index("station_id")


def yv(i, y):
    v = V[i, yr == y]
    return np.delete(v, 59) if v.size == 366 else v


def ind(a, p):
    with np.errstate(all="ignore"):
        rx1 = np.nanmax(a, -1)
        w = sliding_window_view(a, 5, axis=-1)
        s5 = np.where(np.isfinite(w).all(-1), w.sum(-1), np.nan)
        rx5 = np.nanmax(s5, -1)
        pt = np.nansum(np.where(a >= 1, a, 0), -1)
        r95 = np.nansum(np.where(a > p, a, 0), -1) if np.isfinite(p) else np.full(pt.shape, np.nan)
    return np.stack([rx1, rx5, pt, r95])


RAW = np.full((4, S, T), np.nan); COMP = np.zeros((S, T), bool); RAT = {}
for i in np.where(n_elig >= 22)[0]:
    cy = [y for j, y in enumerate(years) if PER[j] and elig[i, j]]
    comp = np.array([yv(i, y) for y in cy]); base = ind(comp, p95[i])
    for t, y in enumerate(yrs):
        j = int(np.where(years == y)[0][0])
        if (status[i, yr == y] == 2).all():
            continue
        v = yv(i, y)
        if not np.isfinite(v).any():
            continue
        RAW[:, i, t] = ind(v, p95[i])
        if elig[i, j]:
            COMP[i, t] = True
        else:
            cm = comp.copy(); cm[:, ~np.isfinite(v)] = np.nan
            with np.errstate(all="ignore"):
                RAT[(i, t)] = ind(cm, p95[i]) / base
RAW[2][RAW[2] == 0] = np.nan


def corrected(k, cut=0.2):
    M = RAW[k].copy()
    for (i, t), r in RAT.items():
        rr = r[k][np.isfinite(r[k]) & (r[k] > 0)]
        if rr.size == 0:
            M[i, t] = np.nan; continue
        g = np.median(rr)
        M[i, t] = M[i, t] / g if g >= cut else np.nan
    return M


I_, J_ = np.triu_indices(T, 1)
dx = (yrs[J_] - yrs[I_]).astype(float)[None, :]


def sen_pct(M, same=None):
    dy = M[:, J_] - M[:, I_]
    with np.errstate(all="ignore"):
        sl = np.where(np.isfinite(dy), dy / dx, np.nan)
        if same is not None:
            sl = np.where(same, sl, np.nan)
        med = np.nanmedian(M, 1)
        o = np.nanmedian(sl, 1) * 10 / med * 100
        o = np.where(med > 0, o, np.nan)
        npair = np.isfinite(sl).sum(1)
    n_ok = np.isfinite(M).sum(1)
    o[(n_ok < 20)] = np.nan
    if same is not None:
        o[npair < 8] = np.nan
    return o


# same-segment pair mask per station
SAME = np.ones((S, len(I_)), bool)
for i in range(S):
    sid_i = sid[i]
    if sid_i in RH.index and RH.loc[sid_i, "significant_dates"]:
        brk = [int(x[:4]) for x in RH.loc[sid_i, "significant_dates"].split(";")]
        seg = np.array([sum(b < y for b in brk) for y in yrs], float)
        seg[np.isin(yrs, brk)] = np.nan                 # a year containing a changepoint is left out
        SAME[i] = (seg[I_] == seg[J_]) & np.isfinite(seg[I_]) & np.isfinite(seg[J_])

sel = n_elig >= 26
flag = np.array([bool(sid[i] in RH.index and RH.loc[sid[i], "significant_dates"]) for i in range(S)])
rows, rows2 = [], []
for k, name in enumerate(NAMES):
    M = {"R1": np.where(COMP, RAW[k], np.nan), "R4": RAW[k], "R4corr": corrected(k)}
    full = {t: sen_pct(M[t]) for t in M}
    seg_ = {t: sen_pct(M[t], SAME) for t in M}
    for grp, gm in (("all stations", sel & np.isin(sid, RH.index)), ("changepoint flagged", sel & flag),
                    ("no changepoint", sel & ~flag & np.isin(sid, RH.index))):
        v0, v1 = full["R1"][gm], seg_["R1"][gm]
        ok = np.isfinite(v0) & np.isfinite(v1)
        bs0 = [np.median(rng.choice(v0[np.isfinite(v0)], np.isfinite(v0).sum())) for _ in range(1000)]
        v1f = v1[np.isfinite(v1)]
        bs1 = [np.median(rng.choice(v1f, len(v1f))) for _ in range(1000)]
        rows.append((name, grp, int(np.isfinite(v0).sum()), np.median(v0[np.isfinite(v0)]), *np.percentile(bs0, [2.5, 97.5]),
                     len(v1f), np.median(v1f), *np.percentile(bs1, [2.5, 97.5]), float((v1f < 0).mean())))
        # paired shift on segmented trends over stations with at least one partly observed year
        aff = gm & np.isfinite(seg_["R4"]) & np.isfinite(seg_["R1"]) & ((M["R4"] != M["R1"]) & np.isfinite(M["R4"])).any(1)
        du = (seg_["R4"] - seg_["R1"])[aff]; dc = (seg_["R4corr"] - seg_["R1"])[aff]
        rows2.append((name, grp, int(aff.sum()), du.mean(), *np.percentile([rng.choice(du, len(du)).mean() for _ in range(1000)], [2.5, 97.5]),
                      dc.mean(), *np.percentile([rng.choice(dc, len(dc)).mean() for _ in range(1000)], [2.5, 97.5])))
A = pd.DataFrame(rows, columns=["index", "group", "n_orig", "median_orig", "o_lo", "o_hi", "n_seg", "median_segmented", "s_lo", "s_hi", "share_declining_seg"])
B = pd.DataFrame(rows2, columns=["index", "group", "n_affected", "shift_uncorr_seg", "u_lo", "u_hi", "shift_corr_seg", "c_lo", "c_hi"])
A.merge(B, on=["index", "group"]).to_csv(OUT / "segmented_trends_pre2018.csv", index=False)
pd.set_option("display.width", 220)
print(A.round(2).to_string(index=False)); print(B.round(2).to_string(index=False))

# (b) shift by station rule
rows = []
for k, name in enumerate(NAMES):
    M = {"R1": np.where(COMP, RAW[k], np.nan), "R4": RAW[k], "R4corr": corrected(k)}
    t1, t4, tc = sen_pct(M["R1"]), sen_pct(M["R4"]), sen_pct(M["R4corr"])
    for th in (22, 26, 30, 32):
        gm = n_elig >= th
        aff = gm & np.isfinite(t4) & np.isfinite(t1) & ((M["R4"] != M["R1"]) & np.isfinite(M["R4"])).any(1)
        du, dc = (t4 - t1)[aff], (tc - t1)[aff]
        rows.append((name, th, int(aff.sum()), du.mean(), *np.percentile([rng.choice(du, len(du)).mean() for _ in range(1000)], [2.5, 97.5]),
                     dc.mean(), *np.percentile([rng.choice(dc, len(dc)).mean() for _ in range(1000)], [2.5, 97.5])))
R = pd.DataFrame(rows, columns=["index", "min_eligible_years", "n_affected", "shift_uncorrected", "u_lo", "u_hi", "shift_corrected", "c_lo", "c_hi"])
R.to_csv(OUT / "shift_by_station_rule_pre2018.csv", index=False)
print(R.round(2).to_string(index=False))
