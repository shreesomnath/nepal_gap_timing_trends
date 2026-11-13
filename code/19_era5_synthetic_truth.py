# Code Developer: Somnath Luitel (somnath.luitel416@topper.wku.edu)
"""Step 19: ERA5 as a gap-free truth for the partly-observed-year problem.

Idea. ERA5 daily rain at each station cell has no gaps. If we impose each station's REAL gauge
gap pattern on it, we can ask directly what the three treatments of partly observed years do to
trend estimates when the truth is known:
  truth   all operating years, full ERA5 days (years in which the gauge had at least one observation)
  R1      eligible years only (gauge eligibility flags), ERA5 masked with the year's real mask
  R4      every year with any observation, masked ERA5, used as if complete
  R4corr  as R4, each partial year divided by the station's median masked/unmasked ratio
          (ERA5 values of the station's eligible years under that year's mask; omitted if < 0.2)
ERA5 is a model, so this says how the estimators behave under realistic gaps, not how large
real rain trends are. ERA5 is not used to judge gauge accuracy.

Scenarios
  real    each year uses the gauge's real mask (all years 1981-2017 with ERA5 coverage)
  replay  all years complete except 2008-2010, which receive the real masks of 2018-2020
          (an outage replayed on a period without it); isolates a block outage from ordinary gaps.
Outputs (per index and treatment): bias of the station trend relative to the truth trend
(mean with 95% station-bootstrap CI, median, RMSE), counts of local (p < 0.05) and FDR (q = 0.10)
declines against the truth counts, and regional medians (Karki et al. 2017 longitude zones).
Writes outputs/era5_truth_<scenario>.csv and outputs/era5_truth_<scenario>_counts.csv.

Usage: python code/19_era5_synthetic_truth.py [--era5-npz outputs/era5_station_daily.npz]
       [--out-dir outputs] [--cell centre|3x3] [--min-years 20] [--nboot 1000]
"""
import argparse
import numpy as np
import pandas as pd
from scipy import stats
from numpy.lib.stride_tricks import sliding_window_view
from pathlib import Path

ap = argparse.ArgumentParser()
ROOT = Path(__file__).resolve().parents[1]
ap.add_argument("--era5-npz", default=str(ROOT / "outputs" / "era5_station_daily.npz"))
ap.add_argument("--out-dir", default=str(ROOT / "outputs"))
ap.add_argument("--cell", default="centre", choices=["centre", "3x3"])
ap.add_argument("--min-years", type=int, default=20)
ap.add_argument("--nboot", type=int, default=1000)
ap.add_argument("--seed", type=int, default=19)
a = ap.parse_args()
rng = np.random.default_rng(a.seed)
OUT = ROOT / "outputs"
OD = Path(a.out_dir)
OD.mkdir(parents=True, exist_ok=True)

g = np.load(OUT / "daily_qc.npz", allow_pickle=True)
status, elig, years_all, sid = g["status"], g["eligible"], g["years"], g["station_id"].astype(str)
lon = g["lon"].astype(float)
e = np.load(a.era5_npz, allow_pickle=True)
assert (e["station_id"].astype(str) == sid).all()
ER = e["era5_centre_mm" if a.cell == "centre" else "era5_3x3_mm"].astype(float)
dates = pd.DatetimeIndex(e["dates"])
yr, mo, dd = dates.year.to_numpy(), dates.month.to_numpy(), dates.day.to_numpy()
keep = ~((mo == 2) & (dd == 29))

# ---- arrays (station, year, 365) for years 1981-2017 with ERA5 coverage
Y = []
for y in range(1981, 2018):
    m = (yr == y) & keep
    if m.sum() == 365 and np.isfinite(ER[:, m]).mean() > 0.999:
        Y.append(y)
Y = np.array(Y)
T = len(Y)
print("ERA5 years usable:", T, Y.tolist())
E = np.stack([ER[:, (yr == y) & keep] for y in Y], 1)                    # (S, T, 365)
OBS = np.stack([status[:, (yr == y) & keep] == 0 for y in Y], 1)         # observed by gauge
MON = np.array([mo[(yr == Y[0]) & keep]])[0]
ELG = np.stack([elig[:, list(years_all).index(y)] for y in Y], 1)        # real eligibility flags
S = len(sid)
ZONE = np.where(lon < 83, "West", np.where(lon < 86, "Central", "East"))
CELLID = pd.factorize(pd.Series(list(zip(e["cell_lat"].round(3), e["cell_lon"].round(3)))))[0]
# stations: at least 26 eligible gauge years in 1981-2017 (as step 17)
m_all = [list(years_all).index(y) for y in range(1981, 2018)]
cand = elig[:, m_all].sum(1) >= 26
# p95 of ERA5 wet days, 1981-2010 years available
wetvals = np.where(E[:, Y <= 2010] >= 1, E[:, Y <= 2010], np.nan).reshape(S, -1)
with np.errstate(all="ignore"):
    P95 = np.nanpercentile(wetvals, 95, axis=1)


def ind(x, p):
    """Rx1day, Rx5day, PRCPTOT, R95pTOT for an array (..., 365); NaN days ignored; Rx5day needs 5 observed days."""
    with np.errstate(all="ignore"):
        rx1 = np.nanmax(x, -1)
        w = sliding_window_view(x, 5, axis=-1)
        s5 = np.where(np.isfinite(w).all(-1), w.sum(-1), np.nan)
        rx5 = np.nanmax(s5, -1)
        pt = np.nansum(np.where(x >= 1, x, 0), -1)
        r95 = np.nansum(np.where(x > p[..., None], x, 0), -1)
        allnan = ~np.isfinite(x).any(-1)
    out = np.stack([rx1, rx5, pt, r95])
    out[:, allnan] = np.nan
    return out


NAMES = ["Rx1day", "Rx5day", "PRCPTOT", "R95pTOT"]
I, J = np.triu_indices(T, 1)


def sen_pct(M):
    dy = M[:, J] - M[:, I]
    dx = (Y[J] - Y[I]).astype(float)[None, :]
    with np.errstate(all="ignore"):
        sl = np.where(np.isfinite(dy), dy / dx, np.nan)
        med = np.nanmedian(M, 1)
        o = np.nanmedian(sl, 1) * 10 / med * 100
        o = np.where(med > 0, o, np.nan)
    o[np.isfinite(M).sum(1) < a.min_years] = np.nan
    return o


def mk(M):
    dy = np.sign(M[:, J] - M[:, I])
    s = np.nansum(np.where(np.isfinite(dy), dy, 0), 1)
    n = np.isfinite(M).sum(1)
    var = n * (n - 1) * (2 * n + 5) / 18
    with np.errstate(all="ignore"):
        z = np.where(s != 0, (s - np.sign(s)) / np.sqrt(var), 0)
    p = 2 * (1 - stats.norm.cdf(np.abs(z)))
    p[n < a.min_years] = np.nan
    return p, s


def bh(p, q=0.10):
    ok = np.isfinite(p); pv = p[ok]; o = np.argsort(pv); m = len(pv)
    passed = pv[o] <= (np.arange(1, m + 1) / m) * q
    k = np.where(passed)[0].max() + 1 if passed.any() else 0
    sig = np.zeros(m, bool); sig[o[:k]] = True
    out = np.zeros(len(p), bool); out[np.where(ok)[0][sig]] = True
    return out


def elig_from_mask(obs):
    """95% of days observed and no month with more than 3 missing days (as step 1)."""
    ok = obs.mean(-1) >= 0.95
    for k in range(1, 13):
        ok &= (~obs[..., MON == k]).sum(-1) <= 3
    return ok


def run(scenario):
    if scenario == "real":
        obs = OBS.copy(); elg = ELG.copy()
    else:
        obs = np.ones_like(OBS)
        for t, y in enumerate(Y):
            if 2008 <= y <= 2010:
                j = list(years_all).index(y + 10)
                src = status[:, (np.isin(dates.year, [y + 10])) & keep] == 0
                obs[:, t] = src
        elg = elig_from_mask(obs)
    anyobs = obs.any(-1)
    # operating-year truth, masked data
    truth = np.full((4, S, T), np.nan); r1 = truth.copy(); r4 = truth.copy(); r4c = truth.copy()
    full = ind(E, np.broadcast_to(P95[:, None], (S, T)))
    masked = ind(np.where(obs, E, np.nan), np.broadcast_to(P95[:, None], (S, T)))
    for i in np.where(cand)[0]:
        for t in range(T):
            if not anyobs[i, t]:
                continue
            truth[:, i, t] = full[:, i, t]
            r4[:, i, t] = masked[:, i, t]
            if elg[i, t]:
                r1[:, i, t] = masked[:, i, t]
        ce = np.where(elg[i] & anyobs[i])[0]
        if len(ce) < 5:
            continue
        base = full[:, i, ce]
        for t in np.where(anyobs[i] & ~elg[i])[0]:
            cm = np.where(obs[i, t][None, :], E[i, ce], np.nan)
            ratio = ind(cm, np.full(len(ce), P95[i])) / base
            for k in range(4):
                rr = ratio[k][np.isfinite(ratio[k]) & (ratio[k] > 0)]
                if rr.size == 0:
                    continue
                gm = np.median(rr)
                r4c[k, i, t] = r4[k, i, t] / gm if gm >= 0.2 else np.nan
        # eligible years in R4corr keep their masked value
        for t in ce:
            r4c[:, i, t] = masked[:, i, t]
    rows, cnt = [], []
    for k, name in enumerate(NAMES):
        tt = sen_pct(truth[k]); pt, st_ = mk(truth[k])
        use = cand & np.isfinite(tt)
        sig_t = bh(np.where(use, pt, np.nan))
        cnt.append((name, "truth", int(use.sum()), int(((pt < 0.05) & (st_ < 0) & use).sum()),
                    int((sig_t & (st_ < 0)).sum()), np.nanmedian(tt[use]),
                    *[np.nanmedian(tt[use & (ZONE == z)]) for z in ("East", "Central", "West")]))
        for tname, M in (("R1", r1[k]), ("R4", r4[k]), ("R4corr", r4c[k])):
            tr = sen_pct(M); p, s_ = mk(M)
            ok = use & np.isfinite(tr)
            d = tr[ok] - tt[ok]
            cid = CELLID[ok]                                   # resample ERA5 cells (several stations share a cell): audit fix
            uc = np.unique(cid)
            sums = np.array([d[cid == u].sum() for u in uc]); cnts = np.array([(cid == u).sum() for u in uc])
            bs = []
            for _ in range(a.nboot):
                pick = rng.integers(len(uc), size=len(uc))
                bs.append(sums[pick].sum() / cnts[pick].sum())
            rows.append((name, tname, int(ok.sum()), d.mean(), *np.percentile(bs, [2.5, 97.5]), np.median(d),
                         np.sqrt(np.mean(d ** 2)), np.corrcoef(tr[ok], tt[ok])[0, 1]))
            sg = bh(np.where(ok, p, np.nan))
            cnt.append((name, tname, int(ok.sum()), int(((p < 0.05) & (s_ < 0) & ok).sum()),
                        int((sg & (s_ < 0)).sum()), np.nanmedian(tr[ok]),
                        *[np.nanmedian(tr[ok & (ZONE == z)]) for z in ("East", "Central", "West")]))
    R = pd.DataFrame(rows, columns=["index", "treatment", "n_stations", "mean_bias_pct_dec", "ci_lo", "ci_hi",
                                    "median_bias", "rmse", "corr_with_truth"])
    C = pd.DataFrame(cnt, columns=["index", "treatment", "n_stations", "local_declines_p05", "fdr_declines",
                                   "median_trend", "East", "Central", "West"])
    R.to_csv(OD / f"era5_truth_{scenario}.csv", index=False)
    C.to_csv(OD / f"era5_truth_{scenario}_counts.csv", index=False)
    pd.set_option("display.width", 220)
    print(f"\n===== scenario: {scenario} (stations {int(cand.sum())}, years {T}) =====")
    print(R.round(3).to_string(index=False))
    print(C.round(2).to_string(index=False))


for sc in ("real", "replay"):
    run(sc)
