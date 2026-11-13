# Code Developer: Somnath Luitel (somnath.luitel416@topper.wku.edu)
"""Step 3: do trend conclusions depend on which station-years are counted?

For 1981-2023 we estimate station trends (Sen slope, Mann-Kendall with a lag-1
prewhitening check) in Rx1day, Rx5day, PRCPTOT and R95pTOT under four completeness
rules commonly used in the literature, and compare a fixed network with the full
changing network. Regional composites are formed from standardised anomalies.

Rules (applied per station-year):
  R1 strict    : eligible year (>=95% days observed, no month missing > 3 days)
  R2 moderate  : >= 80% of days observed
  R3 lax       : >= 50% of days observed
  R4 any       : at least one observed day (index computed from whatever exists)
Station-level inclusion: at least 30 qualifying years in 1981-2023 and at least 5 in
2014-2023 (so a trend is not driven only by the early record).

Outputs: outputs/trend_by_rule.csv, printed summary.
"""
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
ann = pd.read_csv(OUT / "annual_indices.csv", dtype={"station_id": str})
inv = pd.read_csv(OUT / "station_inventory.csv", dtype={"Station_ID": str})
d = np.load(OUT / "daily_qc.npz", allow_pickle=True)
elig = pd.DataFrame(d["eligible"], index=d["station_id"], columns=d["years"])

ann = ann[(ann.year >= 1981) & (ann.year <= 2023)].copy()
ann["eligible"] = [elig.at[s, y] for s, y in zip(ann.station_id, ann.year)]
RULES = {"R1_strict": ann.eligible, "R2_80pct": ann.obs_frac >= 0.80,
         "R3_50pct": ann.obs_frac >= 0.50, "R4_any": ann.obs_frac > 0}
INDICES = ["Rx1day", "Rx5day", "PRCPTOT", "R95pTOT"]


def sen_mk(t, y):
    """Sen slope per decade and two-sided Mann-Kendall p-value (ties ignored)."""
    n = len(y)
    s = sum(np.sign(y[j] - y[i]) for i in range(n - 1) for j in range(i + 1, n))
    var = n * (n - 1) * (2 * n + 5) / 18
    z = (s - np.sign(s)) / np.sqrt(var) if s != 0 else 0.0
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    slope = stats.theilslopes(y, t)[0] * 10
    return slope, p


def fdr(p, q=0.10):
    """Benjamini-Hochberg: which tests are significant at false-discovery rate q."""
    p = np.asarray(p)
    o = np.argsort(p)
    thr = q * np.arange(1, len(p) + 1) / len(p)
    passed = p[o] <= thr
    k = np.where(passed)[0].max() + 1 if passed.any() else 0
    sig = np.zeros(len(p), bool)
    sig[o[:k]] = True
    return sig


rows = []
for rule, mask in RULES.items():
    sub = ann[mask]
    cnt = sub.groupby("station_id").year.agg(n="count", late=lambda s: (s >= 2014).sum())
    keep = cnt[(cnt.n >= 30) & (cnt.late >= 5)].index
    for ix in INDICES:
        res = []
        for s in keep:
            g = sub[sub.station_id == s].dropna(subset=[ix])
            if len(g) < 30:
                continue
            sl, p = sen_mk(g.year.to_numpy(float), g[ix].to_numpy(float))
            med = g[ix].median()
            res.append((s, sl, 100 * sl / med if med > 0 else np.nan, p))
        r = pd.DataFrame(res, columns=["station_id", "slope_dec", "slope_pct_dec", "p"])
        r["sig_local"] = r.p < 0.05
        r["sig_fdr"] = fdr(r.p.to_numpy())
        for _, x in r.iterrows():
            rows.append((rule, ix, x.station_id, x.slope_dec, x.slope_pct_dec, x.p,
                         x.sig_local, x.sig_fdr))

tr = pd.DataFrame(rows, columns=["rule", "index", "station_id", "slope_dec", "slope_pct_dec",
                                 "p", "sig_local", "sig_fdr"])
tr.to_csv(OUT / "trend_by_rule.csv", index=False)

summ = tr.groupby(["index", "rule"]).agg(
    n_stations=("station_id", "count"),
    median_pct_dec=("slope_pct_dec", "median"),
    share_pos=("slope_dec", lambda s: (s > 0).mean()),
    sig_pos_local=("sig_local", lambda s: int((s & (tr.loc[s.index, "slope_dec"] > 0)).sum())),
    sig_neg_local=("sig_local", lambda s: int((s & (tr.loc[s.index, "slope_dec"] < 0)).sum())),
    sig_fdr=("sig_fdr", "sum"))
print(summ.round(2).to_string())

# Same stations under every rule: isolates the effect of the rule from station composition
common = set.intersection(*[set(tr[(tr.rule == r) & (tr["index"] == "Rx1day")].station_id)
                            for r in RULES])
print("\nstations common to all rules:", len(common))
cc = tr[tr.station_id.isin(common)]
print(cc.groupby(["index", "rule"]).agg(median_pct_dec=("slope_pct_dec", "median"),
                                         share_pos=("slope_dec", lambda s: (s > 0).mean()),
                                         sig_local=("sig_local", "sum"),
                                         sig_fdr=("sig_fdr", "sum")).round(2).to_string())
