# Code Developer: Somnath Luitel (somnath.luitel416@topper.wku.edu)
"""Step 20: serial-correlation-aware tests and sensitivity of the 1981-2017 result.

Question. Does the number of stations with a significant decline under the three treatments
(strict R1, all years uncorrected R4, all years gap-corrected R4corr) depend on
  (a) the trend test: Mann-Kendall (MK), Hamed-Rao modified MK (MMK), trend-free pre-whitened MK (TFPW)?
  (b) the station-inclusion threshold (at least 22, 26 or 30 eligible years in 1981-2017)?
  (c) removing the stations flagged by the homogeneity screen (outputs/homogeneity.csv)?
  (d) the correction's minimum expected-ratio cut-off (0.2 in step 12; here also 0.1 and 0.4)?
FDR control: Benjamini-Hochberg, q = 0.10, within each index and treatment.
Series are built as in step 17 (annual indices from daily_qc.npz; correction = division by the
station's median masked/unmasked ratio over its eligible years). Missing years are dropped and
the series treated as consecutive for the autocorrelation terms (a caveat, as in step 13).
Outputs: outputs/sens_tests_pre2018.csv (a), outputs/sens_threshold_pre2018.csv (b, c, d).
"""
import numpy as np
import pandas as pd
from scipy import stats
from numpy.lib.stride_tricks import sliding_window_view
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
d = np.load(OUT / "daily_qc.npz", allow_pickle=True)
X, status, elig, years, sid = d["X"], d["status"], d["eligible"], d["years"], d["station_id"].astype(str)
dates = pd.to_datetime(d["dates"]); yr = dates.year.to_numpy()
V = np.where(status == 0, X, np.nan)
p95 = pd.read_csv(OUT / "p95_baseline.csv", dtype={"station_id": str}).p95_1981_2010.to_numpy()
hom = pd.read_csv(OUT / "homogeneity.csv", dtype={"station_id": str}).set_index("station_id")
PER = (years >= 1981) & (years <= 2017)
yrs = years[PER]; T = len(yrs); S = len(sid)
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
for i in range(S):
    cy = [y for j, y in enumerate(years) if PER[j] and elig[i, j]]
    if len(cy) < 5:
        continue
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


def mk_stat(x):
    n = len(x); s = 0.0
    for a in range(n - 1):
        s += np.sign(x[a + 1:] - x[a]).sum()
    return s


def p_from(s, var):
    z = (s - np.sign(s)) / np.sqrt(var) if s != 0 else 0.0
    return 2 * (1 - stats.norm.cdf(abs(z)))


def test_mk(t, x, kind):
    ok = np.isfinite(x); x = x[ok]; t = t[ok].astype(float); n = len(x)
    if n < 20:
        return np.nan, 0.0
    s = mk_stat(x); var = n * (n - 1) * (2 * n + 5) / 18
    if kind == "MK":
        return p_from(s, var), s
    sl = np.median([(x[j] - x[i]) / (t[j] - t[i]) for i in range(n - 1) for j in range(i + 1, n)])
    det = x - sl * t
    if kind == "MMK":
        rk = stats.rankdata(det); rk = rk - rk.mean()
        den = (rk ** 2).sum(); fac = 1.0
        corr = 0.0
        for k in range(1, n - 2):                     # all lags; Hamed-Rao (1998) significance limits (audit fix)
            r = (rk[:-k] * rk[k:]).sum() / den
            lo_, hi_ = (-1 - 1.96 * np.sqrt(n - k - 1)) / (n - k), (-1 + 1.96 * np.sqrt(n - k - 1)) / (n - k)
            if r < lo_ or r > hi_:
                corr += (n - k) * (n - k - 1) * (n - k - 2) * r
        fac = 1 + 2 / (n * (n - 1) * (n - 2)) * corr
        fac = max(fac, 1e-3)
        return p_from(s, var * fac), s
    if kind == "TFPW":
        r1 = (det[:-1] - det.mean()) @ (det[1:] - det.mean()) / ((det - det.mean()) ** 2).sum()
        if abs(r1) > 1.96 / np.sqrt(n):
            pw = det[1:] - r1 * det[:-1] + sl * t[1:]
            s2 = mk_stat(pw); m = len(pw)
            return p_from(s2, m * (m - 1) * (2 * m + 5) / 18), s2
        return p_from(s, var), s


def bh(p, q=0.10):
    ok = np.isfinite(p); pv = p[ok]; o = np.argsort(pv); m = len(pv)
    passed = pv[o] <= (np.arange(1, m + 1) / m) * q
    k = np.where(passed)[0].max() + 1 if passed.any() else 0
    sg = np.zeros(m, bool); sg[o[:k]] = True
    out = np.zeros(len(p), bool); out[np.where(ok)[0][sg]] = True
    return out


m_idx = [list(years).index(y) for y in range(1981, 2018)]
n_elig = elig[:, m_idx].sum(1)
flag = np.array([hom.loc[s, "result"] == "flag" if s in hom.index else False for s in sid])


def counts(M, sel, kind):
    p = np.full(S, np.nan); s_ = np.zeros(S)
    for i in np.where(sel)[0]:
        p[i], s_[i] = test_mk(yrs, M[i], kind)
    sg = bh(np.where(sel, p, np.nan))
    return int((sg & (s_ < 0)).sum()), int((sg & (s_ > 0)).sum()), int(((p < 0.05) & (s_ < 0)).sum()), int(np.isfinite(p).sum())


rows_a, rows_b = [], []
sel26 = n_elig >= 26
CUT = {}
for k, name in enumerate(NAMES):
    R1 = np.where(COMP, RAW[k], np.nan); R4 = RAW[k]; RC = corrected(k, 0.2)
    for tname, M in (("R1", R1), ("R4", R4), ("R4corr", RC)):
        for kind in ("MK", "MMK", "TFPW"):
            fd, fi, ld, n = counts(M, sel26, kind)
            rows_a.append((name, tname, kind, n, fd, fi, ld))
    if name in ("Rx1day", "PRCPTOT"):
        for th in (22, 26, 30):
            for tname, M in (("R1", R1), ("R4", R4), ("R4corr", RC)):
                fd, fi, ld, n = counts(M, n_elig >= th, "MK")
                rows_b.append((name, "threshold>=%d" % th, tname, n, fd, fi, ld))
        for tname, M in (("R1", R1), ("R4", R4), ("R4corr", RC)):
            fd, fi, ld, n = counts(M, sel26 & ~flag, "MK")
            rows_b.append((name, "excl_homogeneity_flagged", tname, n, fd, fi, ld))
        for cut in (0.1, 0.4):
            fd, fi, ld, n = counts(corrected(k, cut), sel26, "MK")
            rows_b.append((name, "ratio_cutoff_%.1f" % cut, "R4corr", n, fd, fi, ld))
cols = ["index", "treatment", "test", "n_stations", "fdr_declines", "fdr_increases", "local_declines_p05"]
A = pd.DataFrame(rows_a, columns=cols); A.to_csv(OUT / "sens_tests_pre2018.csv", index=False)
B = pd.DataFrame(rows_b, columns=["index", "variant", "treatment", "n_stations", "fdr_declines", "fdr_increases", "local_declines_p05"])
B.to_csv(OUT / "sens_threshold_pre2018.csv", index=False)
pd.set_option("display.width", 200)
print(A.pivot_table(index=["index", "test"], columns="treatment", values="fdr_declines").to_string())
print(B.to_string(index=False))
