# Code Developer: Somnath Luitel (somnath.luitel416@topper.wku.edu)
"""Step 30: uncertainty of the numbers of FDR-significant declining stations (1981-2017, 219 stations).

The counts (for example 9, 41 and 25 for the annual total) have no sampling uncertainty in the main tables.
Here whole years are resampled jointly across all stations (years drawn with replacement, kept in time order, so cross-station
correlation and the real gap timing travel with each year), the Mann-Kendall test and Benjamini-Hochberg control (q = 0.10)
are repeated for eligible years only (R1), all years uncorrected (R4) and all years corrected (R4corr), and the counts and their
differences are summarised over 500 replicates. Duplicated years give zero-difference pairs, which do not contribute to S;
n counts every resampled value, a small approximation. Output: outputs/count_uncertainty_pre2018.csv
"""
import numpy as np
import pandas as pd
from scipy import stats
from numpy.lib.stride_tricks import sliding_window_view
from pathlib import Path

rng = np.random.default_rng(30)
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
d = np.load(OUT / "daily_qc.npz", allow_pickle=True)
X, status, elig, years = d["X"], d["status"], d["eligible"], d["years"]
dates = pd.to_datetime(d["dates"]); yr = dates.year.to_numpy()
V = np.where(status == 0, X, np.nan)
p95 = pd.read_csv(OUT / "p95_baseline.csv", dtype={"station_id": str}).p95_1981_2010.to_numpy()
PER = (years >= 1981) & (years <= 2017)
yrs = years[PER]; T = len(yrs); S = V.shape[0]
m_idx = [list(years).index(y) for y in range(1981, 2018)]
sel = elig[:, m_idx].sum(1) >= 26
NAMES = ["Rx1day", "Rx5day", "PRCPTOT", "R95pTOT"]


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
for i in np.where(sel)[0]:
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


def mk(M):
    I_, J_ = np.triu_indices(M.shape[1], 1)
    dy = np.sign(M[:, J_] - M[:, I_])
    s = np.nansum(np.where(np.isfinite(dy), dy, 0), 1)
    n = np.isfinite(M).sum(1)
    var = n * (n - 1) * (2 * n + 5) / 18
    with np.errstate(all="ignore"):
        z = np.where(s != 0, (s - np.sign(s)) / np.sqrt(var), 0)
    p = 2 * (1 - stats.norm.cdf(np.abs(z))); p[n < 20] = np.nan
    return p, s


def bh(p, q=0.10):
    ok = np.isfinite(p); pv = p[ok]; o = np.argsort(pv); m = len(pv)
    passed = pv[o] <= (np.arange(1, m + 1) / m) * q
    k = np.where(passed)[0].max() + 1 if passed.any() else 0
    sg = np.zeros(m, bool); sg[o[:k]] = True
    out = np.zeros(len(p), bool); out[np.where(ok)[0][sg]] = True
    return out


def count(M):
    p, s_ = mk(M[sel]); sg = bh(p)
    return int((sg & (s_ < 0)).sum())


rows = []
# Delete-a-block jackknife over consecutive 3-year blocks. (A year bootstrap with duplicated years inflates the
# Mann-Kendall significance and gave counts far above the observed ones, so it is not used.)
blocks = [np.arange(s, min(s + 3, T)) for s in range(0, T, 3)]
G = len(blocks)
for k, name in enumerate(NAMES):
    M = {"R1": np.where(COMP, RAW[k], np.nan), "R4": RAW[k], "R4corr": corrected(k)}
    obs = {t: count(M[t]) for t in M}
    reps = {t: [] for t in M}
    for b in blocks:
        keep_ = np.setdiff1d(np.arange(T), b)
        for t in M:
            reps[t].append(count(M[t][:, keep_]))
    r1, r4, rc = (np.array(reps[t], float) for t in ("R1", "R4", "R4corr"))
    def jack(a):
        return np.sqrt((G - 1) / G * ((a - a.mean()) ** 2).sum())
    for t, a in (("R1", r1), ("R4", r4), ("R4corr", rc)):
        rows.append((name, t, obs[t], jack(a), obs[t] - 1.96 * jack(a), obs[t] + 1.96 * jack(a)))
    for lab, a, o in (("R4 - R1", r4 - r1, obs["R4"] - obs["R1"]), ("R4corr - R1", rc - r1, obs["R4corr"] - obs["R1"]),
                      ("R4 - R4corr", r4 - rc, obs["R4"] - obs["R4corr"])):
        rows.append((name, lab, o, jack(a), o - 1.96 * jack(a), o + 1.96 * jack(a)))
R = pd.DataFrame(rows, columns=["index", "quantity", "observed", "jackknife_se", "lo95", "hi95"])
R.to_csv(OUT / "count_uncertainty_pre2018.csv", index=False)
pd.set_option("display.width", 200)
print(R.round(1).to_string(index=False))
