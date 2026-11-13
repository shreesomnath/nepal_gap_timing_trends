# Code Developer: Somnath Luitel (somnath.luitel416@topper.wku.edu)
"""Step 32 (step 9 restricted to 1981-2017 and the 219 stations with at least 26 eligible years;
output outputs/homogeneity_pre2018.csv).

Step 9: relative homogeneity screening of annual precipitation totals.

Substitute for RHtests_dlyPrcp (R is not installed). For each of the 184 stations used
in the trend analysis:
  reference = mean of log annual wet-day totals (PRCPTOT) of the best-correlated
              neighbours (up to 5, within 150 km, >= 25 common complete years,
              correlation of log totals >= 0.5), using complete years only;
  Q_t       = log(target_t) - log(reference_t), standardised;
  SNHT      = max_k [ k z1^2 + (n-k) z2^2 ] (Alexandersson 1986); critical value from
              20,000 Monte Carlo series of the same length.
Only the single most significant break per station is reported. A flagged break is a
candidate, not proof of an instrument change: without metadata it can reflect real
regional change or a reference station's own break.
Outputs: outputs/homogeneity.csv
"""
import numpy as np
import pandas as pd
from pathlib import Path

rng = np.random.default_rng(5)
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
d = np.load(OUT / "daily_qc.npz", allow_pickle=True)
elig, years, sid = d["eligible"], d["years"], d["station_id"]
lat, lon = np.radians(d["lat"].astype(float)), np.radians(d["lon"].astype(float))
ann = pd.read_csv(OUT / "annual_indices.csv", dtype={"station_id": str})
_m = [list(years).index(y) for y in range(1981, 2018)]
used = set(np.asarray(sid)[elig[:, _m].sum(1) >= 26].astype(str))

PER = (years >= 1981) & (years <= 2017)
yrs = years[PER]
tot = np.full((len(sid), len(yrs)), np.nan)
pos = {s: k for k, s in enumerate(sid)}
for r in ann.itertuples():
    if 1981 <= r.year <= 2017 and elig[pos[r.station_id], list(years).index(r.year)] and r.PRCPTOT > 0:
        tot[pos[r.station_id], r.year - 1981] = np.log(r.PRCPTOT)

a = (np.sin((lat[:, None] - lat[None, :]) / 2) ** 2 +
     np.cos(lat[:, None]) * np.cos(lat[None, :]) * np.sin((lon[:, None] - lon[None, :]) / 2) ** 2)
dist = 2 * 6371 * np.arcsin(np.sqrt(a))


def snht(z):
    n = len(z)
    k = np.arange(1, n)
    c = np.cumsum(z)
    m1 = c[:-1] / k
    m2 = (c[-1] - c[:-1]) / (n - k)
    T = k * m1 ** 2 + (n - k) * m2 ** 2
    return T.max(), int(T.argmax())


crit = {}


def critical(n, level=0.95):
    if n not in crit:
        sims = []
        for _ in range(20000):
            z = rng.standard_normal(n)
            z = (z - z.mean()) / z.std(ddof=1)
            sims.append(snht(z)[0])
        crit[n] = (np.quantile(sims, level), np.sort(sims))
    return crit[n]


rows = []
for i in range(len(sid)):
    if str(sid[i]) not in used:
        continue
    cand = []
    for k in np.where((dist[i] > 0) & (dist[i] <= 150))[0]:
        m = np.isfinite(tot[i]) & np.isfinite(tot[k])
        if m.sum() >= 25:
            c = np.corrcoef(tot[i, m], tot[k, m])[0, 1]
            if c >= 0.5:
                cand.append((c, k))
    cand = [k for c, k in sorted(cand, reverse=True)[:5]]
    if len(cand) < 3:
        rows.append((sid[i], len(cand), np.nan, np.nan, np.nan, np.nan, np.nan, "no_reference"))
        continue
    ref = np.nanmean(tot[cand], 0)
    q = tot[i] - ref
    ok = np.isfinite(q)
    qq = q[ok]
    if len(qq) < 25:
        rows.append((sid[i], len(cand), len(qq), np.nan, np.nan, np.nan, np.nan, "short"))
        continue
    z = (qq - qq.mean()) / qq.std(ddof=1)
    T, k = snht(z)
    c95, sims = critical(len(qq))
    p = 1 - np.searchsorted(sims, T) / len(sims)
    yy = yrs[ok]
    shift = 100 * (np.exp(qq[k + 1:].mean() - qq[:k + 1].mean()) - 1)
    rows.append((sid[i], len(cand), len(qq), T, p, int(yy[k]) + 1, shift,
                 "flag" if T > c95 else "ok"))

h = pd.DataFrame(rows, columns=["station_id", "n_ref", "n_years", "T", "p", "break_year_start",
                                "shift_pct", "result"])
h.to_csv(OUT / "homogeneity_pre2018.csv", index=False)
print(h.result.value_counts().to_string())
f = h[h.result == "flag"]
print("flagged break years:", f.break_year_start.value_counts().sort_index().to_dict())
print("median |shift| of flagged (%):", round(f.shift_pct.abs().median(), 1))
# With 5% level and ~170 tests, about 8 false flags are expected by chance
print("tests:", int((h.result.isin(['ok', 'flag'])).sum()), "expected false flags at 5%:",
      round(0.05 * h.result.isin(['ok', 'flag']).sum(), 1))
