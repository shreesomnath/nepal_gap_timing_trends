# Code Developer: Somnath Luitel (somnath.luitel416@topper.wku.edu)
"""Step 25: a gauge-based truth experiment for the three treatments (1981-2017).

ERA5 (step 19) gives a gap-free truth but is a model. Here the truth is real gauge rain. For each
station with at least 26 eligible years in 1981-2017, the eligible years are complete gauge
years. Each replicate degrades a random subset of them, each year with probability q_i equal to the
station's own share of partly observed years, by a REAL missingness mask drawn at random from the pool
of all partly observed station-years of the archive in 1981-2017 (so seasonal and block structure is
real). Treatments, compared with the unmasked series of the same years (truth):
  R1      years left undegraded only (what a strict rule keeps)
  R4      all years, degraded years used as if complete
  R4corr  as R4, each degraded year divided by the station's median masked/unmasked ratio over its
          other eligible years (omitted if the ratio is below 0.2)
Outputs per index and treatment: bias of the station trend (% per decade; mean across stations, then
mean and 95% range across replicates), and the number of stations with an FDR (q = 0.10) significant
decline and with a local (p < 0.05) decline, for truth and each treatment.
Trend = Sen slope in % of the station median per decade; test = Mann-Kendall (no pre-whitening).
Three scenarios (station_timing: degradation probability of each year follows a kernel of exp(-|t - p| / 2) around the station's own real partly observed years p, so gaps cluster in time at the stations that really have them): random_timing (degraded years chosen at random) and real_timing (probability of degradation
follows the archive's real share of partly observed years in each calendar year, which rises late in the period).
Output: outputs/gauge_truth_experiment.csv. Usage: python code/25_gauge_truth_experiment.py [nrep]
"""
import sys
import numpy as np
import pandas as pd
from scipy import stats
from numpy.lib.stride_tricks import sliding_window_view
from pathlib import Path

NREP = int(sys.argv[1]) if len(sys.argv) > 1 else 60
rng = np.random.default_rng(25)
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
d = np.load(OUT / "daily_qc.npz", allow_pickle=True)
X, status, elig, years, sid = d["X"], d["status"], d["eligible"], d["years"], d["station_id"].astype(str)
dates = pd.to_datetime(d["dates"]); yr = dates.year.to_numpy()
V = np.where(status == 0, X, np.nan)
p95 = pd.read_csv(OUT / "p95_baseline.csv", dtype={"station_id": str}).p95_1981_2010.to_numpy()
Y = np.arange(1981, 2018); T = len(Y)
m_idx = [list(years).index(y) for y in Y]
cand = np.where(elig[:, m_idx].sum(1) >= 26)[0]
cand = np.array([i for i in cand if np.isfinite(p95[i])])
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
        r95 = np.nansum(np.where(a > p, a, 0), -1)
    return np.stack([rx1, rx5, pt, r95])


# station data and the pool of real masks
ST = {}
pool, pshare = [], {}
for i in cand:
    ey, comp, npart, ptime = [], [], 0, []
    for t, y in enumerate(Y):
        j = m_idx[t]
        if (status[i, yr == y] == 2).all():
            continue
        v = yv(i, y)
        if elig[i, j]:
            ey.append(t); comp.append(v)
        elif np.isfinite(v).any():
            npart += 1; ptime.append(t)
    ST[i] = (np.array(ey), np.array(comp), npart / (npart + len(ey)), np.array(ptime, int))
for i in range(len(sid)):                                  # the pool uses every station
    for t, y in enumerate(Y):
        j = m_idx[t]
        if elig[i, j] or (status[i, yr == y] == 2).all():
            continue
        v = yv(i, y)
        if np.isfinite(v).any():
            pool.append(~np.isfinite(v))
pool = np.array(pool)
# real timing: share of partly observed years among operating station-years, by year
W = np.zeros(T)
for t, y in enumerate(Y):
    j = m_idx[t]; npart = nop = 0
    for i in range(len(sid)):
        if (status[i, yr == y] == 2).all():
            continue
        v = yv(i, y)
        if not np.isfinite(v).any():
            continue
        nop += 1; npart += int(not elig[i, j])
    W[t] = npart / max(nop, 1)
W = W / W.mean()
print("stations", len(cand), "mask pool", pool.shape[0], "mean q", np.mean([ST[i][2] for i in cand]).round(3),
      "| relative frequency of partial years, first and last 5 yrs:", W[:5].mean().round(2), W[-5:].mean().round(2))

I_, J_ = np.triu_indices(T, 1)
dx = (Y[J_] - Y[I_]).astype(float)[None, :]


def sen_pct(M):
    dy = M[:, J_] - M[:, I_]
    with np.errstate(all="ignore"):
        sl = np.where(np.isfinite(dy), dy / dx, np.nan)
        med = np.nanmedian(M, 1)
        o = np.nanmedian(sl, 1) * 10 / med * 100
        o = np.where(med > 0, o, np.nan)
    o[np.isfinite(M).sum(1) < 20] = np.nan
    return o


def mk(M):
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


S = len(cand)
rows = []
OMIT = []
# neighbours for infilling: up to three nearest stations within 100 km, scaled by the ratio of mean daily rain
_lat, _lon = np.radians(d["lat"].astype(float)), np.radians(d["lon"].astype(float))
_a = (np.sin((_lat[:, None] - _lat[None, :]) / 2) ** 2 + np.cos(_lat[:, None]) * np.cos(_lat[None, :]) * np.sin((_lon[:, None] - _lon[None, :]) / 2) ** 2)
DIST = 2 * 6371 * np.arcsin(np.sqrt(_a))
MEAN = np.nanmean(V, axis=1)
NB = {}
for i in cand:
    order = [j for j in np.argsort(DIST[i]) if j != i and DIST[i, j] <= 100 and np.isfinite(MEAN[j]) and MEAN[j] > 0][:3]
    NB[i] = [(j, MEAN[i] / MEAN[j]) for j in order]
print("stations with at least one neighbour within 100 km:", sum(1 for i in cand if NB[i]), "of", len(cand))
for rep, scen in [(r, s_) for s_ in ("random_timing", "real_timing", "station_timing") for r in range(NREP)]:
    TR = np.full((4, S, T), np.nan); R1 = TR.copy(); R4 = TR.copy(); RC = TR.copy(); INF = TR.copy(); HY = TR.copy()
    for a, i in enumerate(cand):
        ey, comp, q, ptime = ST[i]
        full = ind(comp, p95[i])                           # (4, n)
        if scen == "random_timing":
            qq = q
        elif scen == "real_timing":
            qq = np.minimum(1.0, q * W[ey])
        else:                                              # station_timing: gaps cluster around this station's own partial years
            Kt = np.array([np.exp(-np.abs(t - ptime) / 2.0).sum() for t in ey]) if len(ptime) else np.zeros(len(ey))
            qq = np.minimum(1.0, q * Kt / Kt.mean()) if Kt.mean() > 0 else np.zeros(len(ey))
        deg = rng.random(len(ey)) < qq
        masked = full.copy(); corr = full.copy(); infl = full.copy(); hy = full.copy()
        q40 = np.nanpercentile(full[:2], 40, axis=1)       # HYADES: 40th percentile of the station's annual maxima
        for k in np.where(deg)[0]:
            mk_ = pool[rng.integers(len(pool))]
            cm = np.where(mk_[None, :], np.nan, comp)
            mv = ind(cm, p95[i])                           # masked indices of every eligible year
            masked[:, k] = mv[:, k]
            with np.errstate(all="ignore"):
                ratio = mv / full
            for kk in range(4):
                # ratio only from years that are complete in this replicate (no leakage from degraded years; audit fix)
                rr = ratio[kk][~deg]; rr = rr[np.isfinite(rr) & (rr > 0)]
                g = np.median(rr) if rr.size >= 5 else np.nan
                corr[kk, k] = mv[kk, k] / g if (np.isfinite(g) and g >= 0.2) else np.nan
            # competitor 1: normal-ratio infilling from up to three neighbours within 100 km (real gauge days)
            fill = np.full(365, np.nan)
            if NB[i]:
                yy_ = int(Y[ey[k]])
                nbv = np.array([yv(nb, yy_) * sc for nb, sc in NB[i]])
                with np.errstate(all="ignore"):
                    fill = np.nanmean(nbv, axis=0)
            newx = comp[k].copy(); newx[mk_] = fill[mk_]
            infl[:, k] = ind(newx[None, :], p95[i])[:, 0]
            # competitor 2: HYADES rule for annual maxima: drop the maximum if more than 33% of days are missing
            # and the masked maximum is below the station's 40th percentile; otherwise keep it as if complete
            for kk in range(2):
                if mk_.mean() > 0.33 and mv[kk, k] < q40[kk]:
                    hy[kk, k] = np.nan
                else:
                    hy[kk, k] = mv[kk, k]
            hy[2:, k] = np.nan                              # not defined for totals
        for kk in range(4):
            TR[kk, a, ey] = full[kk]; R4[kk, a, ey] = masked[kk]; RC[kk, a, ey] = corr[kk]
            R1[kk, a, ey[~deg]] = full[kk, ~deg]
            INF[kk, a, ey] = infl[kk]; HY[kk, a, ey] = hy[kk]
            if deg.any():
                OMIT.append((scen, rep, NAMES[kk], float(np.isnan(corr[kk, deg]).mean())))
    for kk, name in enumerate(NAMES):
        tt = sen_pct(TR[kk]); pt, st_ = mk(TR[kk])
        ok = np.isfinite(tt)
        sg = bh(np.where(ok, pt, np.nan))
        rows.append((scen, rep, name, "truth", np.nan, int((sg & (st_ < 0)).sum()), int(((pt < 0.05) & (st_ < 0) & ok).sum()), np.nanmedian(tt)))
        for tname, M in (("R1", R1[kk]), ("R4", R4[kk]), ("R4corr", RC[kk]), ("INFILL", INF[kk]), ("HYADES", HY[kk])):
            if tname == "HYADES" and kk >= 2:
                continue
            tr = sen_pct(M); p, s_ = mk(M)
            ok2 = ok & np.isfinite(tr)
            sg2 = bh(np.where(ok2, p, np.nan))
            rows.append((scen, rep, name, tname, float(np.mean(tr[ok2] - tt[ok2])), int((sg2 & (s_ < 0)).sum()),
                         int(((p < 0.05) & (s_ < 0) & ok2).sum()), np.nanmedian(tr[ok2])))
    if rep % 10 == 0:
        print("rep", rep, flush=True)
R = pd.DataFrame(rows, columns=["scenario", "rep", "index", "treatment", "mean_bias_pct_dec", "fdr_declines", "local_declines_p05", "median_trend"])
R.to_csv(OUT / "gauge_truth_experiment_reps.csv", index=False)
g = R.groupby(["scenario", "index", "treatment"])
S_ = g.agg(bias_mean=("mean_bias_pct_dec", "mean"), bias_lo=("mean_bias_pct_dec", lambda x: x.quantile(0.025)),
           bias_hi=("mean_bias_pct_dec", lambda x: x.quantile(0.975)), fdr_declines=("fdr_declines", "mean"),
           local_declines=("local_declines_p05", "mean"), median_trend=("median_trend", "mean")).reset_index()
S_.to_csv(OUT / "gauge_truth_experiment.csv", index=False)
om = pd.DataFrame(OMIT, columns=["scenario", "rep", "index", "share_degraded_years_omitted"]).groupby(["scenario", "index"]).share_degraded_years_omitted.mean().reset_index()
om.to_csv(OUT / "gauge_truth_omitted_share.csv", index=False)
print(om.round(3).to_string(index=False))
pd.set_option("display.width", 200)
print(S_.round(2).to_string(index=False))
