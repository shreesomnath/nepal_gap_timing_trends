# Code Developer: Somnath Luitel (somnath.luitel416@topper.wku.edu)
"""Step 27 (step 5 restricted to 1981-2017; stations with at least 29 complete years of the 37; output outputs/gev_experiment_pre2018.csv).

Step 5: return levels from records with real gap patterns, five treatments.

Reference: for each long-record station (>= 35 complete years, 1981-2023) a GEV fitted
by maximum likelihood to Rx1day of all complete years gives reference 10- and 20-year
return levels.
Degraded record: the same years, but a share of years receives a real partial-year
missingness pattern drawn from the archive (pattern placed by day of year), and a share
of years is removed entirely, at the rates observed in the archive.
Treatments on the degraded record:
  strict   : use complete years only
  naive    : use every year's observed maximum as if complete
  hyades   : naive, but drop a year if > 33% days missing and its maximum lies below the
             40th percentile of the station's maxima (Vargas Godoy et al. 2024 rule,
             exceedance probability > 60%)
  sn_raw   : Simpson and Northrop (2026) likelihood, block maximum of a partial year has
             cdf G(x)^f with f = observed fraction of days
  sn_season: same likelihood but f = seasonal weight of the observed days, where the
             weight of a day is the climatological probability that the annual maximum
             falls on it (estimated from the training years only, smoothed by +/-15 days)
Repeated over R random degradations per station. Outputs: outputs/gev_experiment.csv.
"""
import numpy as np
import pandas as pd
from scipy import optimize
from pathlib import Path

rng = np.random.default_rng(7)
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
d = np.load(OUT / "daily_qc.npz", allow_pickle=True)
X, status, elig, years = d["X"], d["status"], d["eligible"], d["years"]
dates = pd.to_datetime(d["dates"])
yr = dates.year.to_numpy()
V = np.where(status == 0, X, np.nan)
R = 20
PERIOD = (years >= 1981) & (years <= 2017)


def yv(i, y):
    v = V[i, yr == y]
    return np.delete(v, 59) if v.size == 366 else v


def gev_nll(p, x, f):
    mu, ls, xi = p
    s = np.exp(ls)
    z = 1 + xi * (x - mu) / s
    if np.any(z <= 0) or abs(xi) > 0.5:
        return 1e10
    t = z ** (-1 / xi) if abs(xi) > 1e-6 else np.exp(-(x - mu) / s)
    # log density of max with cdf G^f: log f + log g + (f-1) log G
    logg = -np.log(s) - (1 + 1 / xi) * np.log(z) - t if abs(xi) > 1e-6 else -np.log(s) - (x - mu) / s - t
    return -np.sum(np.log(f) + logg - (f - 1) * t)


def fit(x, f=None):
    x = np.asarray(x, float)
    f = np.ones_like(x) if f is None else np.clip(np.asarray(f, float), 1e-3, 1)
    best = None
    for xi0 in (0.0, 0.1, -0.1):
        p0 = [np.mean(x) - 0.45 * np.std(x), np.log(0.78 * np.std(x) + 1e-6), xi0]
        r = optimize.minimize(gev_nll, p0, args=(x, f), method="Nelder-Mead",
                              options={"maxiter": 4000, "xatol": 1e-6, "fatol": 1e-8})
        if best is None or r.fun < best.fun:
            best = r
    return best.x


def rl(p, T):
    mu, ls, xi = p
    s = np.exp(ls)
    yp = -np.log(1 - 1 / T)
    return mu + s / xi * (yp ** (-xi) - 1) if abs(xi) > 1e-6 else mu - s * np.log(yp)


# Archive rates and patterns
masks, n_part, n_absent, n_tot = [], 0, 0, 0
for i in range(V.shape[0]):
    for j, y in enumerate(years[PERIOD]):
        st = status[i, yr == y]
        if (st == 2).all():
            continue
        n_tot += 1
        v = yv(i, y)
        fr = np.isfinite(v).mean()
        if fr == 0:
            n_absent += 1
        elif fr < 0.95 or not elig[i, np.where(years == y)[0][0]]:
            n_part += 1
            masks.append(~np.isfinite(v))
p_part, p_abs = n_part / n_tot, n_absent / n_tot
masks = np.array(masks)
print(f"archive rates: partial years {p_part:.3f}, absent years {p_abs:.3f}, patterns {len(masks)}")

rows = []
cand = [i for i in range(V.shape[0]) if elig[i, PERIOD].sum() >= 29]
print("stations:", len(cand))
for i in cand:
    ys = years[PERIOD][elig[i, PERIOD]]
    full = np.array([yv(i, y) for y in ys])            # complete years, 365 days each
    amax = np.nanmax(full, 1)
    ref = fit(amax)
    ref10, ref20 = rl(ref, 10), rl(ref, 20)
    for r in range(R):
        u = rng.random(len(ys))
        absent = u < p_abs
        partial = (u >= p_abs) & (u < p_abs + p_part)
        # seasonal weights from years that remain complete in this degraded record
        keep = ~absent & ~partial
        argm = np.nanargmax(full[keep], 1)
        h = np.bincount(argm, minlength=365).astype(float) + 0.01
        k = np.exp(-0.5 * (np.arange(-30, 31) / 15) ** 2)
        w = np.convolve(np.concatenate([h[-30:], h, h[:30]]), k / k.sum(), "valid")
        w /= w.sum()
        xs, fr, fw, comp = [], [], [], []
        for t in np.where(~absent)[0]:
            v = full[t].copy()
            if partial[t]:
                m = masks[rng.integers(len(masks))]
                if m.all():
                    continue
                v[m] = np.nan
            obs = np.isfinite(v)
            xs.append(np.nanmax(v)); fr.append(obs.mean()); fw.append(w[obs].sum()); comp.append(not partial[t])
        xs, fr, fw, comp = map(np.array, (xs, fr, fw, comp))
        q40 = np.quantile(xs, 0.40)
        drop_h = (fr < 0.67) & (xs < q40)
        est = {"strict": fit(xs[comp]), "naive": fit(xs), "hyades": fit(xs[~drop_h]),
               "sn_raw": fit(xs, fr), "sn_season": fit(xs, fw)}
        for kname, p in est.items():
            rows.append((d["station_id"][i], r, kname, len(ys), int(partial.sum()),
                         100 * (rl(p, 10) - ref10) / ref10, 100 * (rl(p, 20) - ref20) / ref20))

res = pd.DataFrame(rows, columns=["station_id", "rep", "method", "n_years", "n_partial",
                                  "err10_pct", "err20_pct"])
res.to_csv(OUT / "gev_experiment_pre2018.csv", index=False)
g = res.groupby("method")
print(pd.DataFrame({"bias10_median": g.err10_pct.median(), "bias10_mean": g.err10_pct.mean(),
                    "rmse10": g.err10_pct.apply(lambda e: np.sqrt((e ** 2).mean())),
                    "bias20_mean": g.err20_pct.mean(),
                    "rmse20": g.err20_pct.apply(lambda e: np.sqrt((e ** 2).mean()))}).round(2).to_string())
