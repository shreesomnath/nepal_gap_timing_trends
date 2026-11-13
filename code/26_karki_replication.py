# Code Developer: Somnath Luitel (somnath.luitel416@topper.wku.edu)
"""Step 26: like-for-like replication of Karki et al. (2017) on their 76 stations.

Same stations (outputs/karki_station_check.csv), same indices (Rx1day, Rx5day, R95pTOT = their R95, annual total),
same sub-regions (WL, WM, CL, CM, CH, EL, EM), station-level trend test = Mann-Kendall with trend-free
pre-whitening (Yue et al. 2002), field significance by 1000-fold bootstrap of whole years across all
stations of a sub-region (preserves cross-correlation) at the 0.1 level (their Eqs 1-3), plain MK inside the
bootstrap. Differences that remain: period 1981-2012 (our archive starts in 1980; theirs 1970-2012), our quality
control, eligible years only (at least 20), annual indices from this archive.
Output: outputs/karki_replication.csv. Their reported field-significant signs are listed in the printed comparison.
"""
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
rng = np.random.default_rng(26)
K = pd.read_csv(OUT / "karki_station_check.csv", dtype={"station_id": str})
d = np.load(OUT / "daily_qc.npz", allow_pickle=True)
elig = pd.DataFrame(d["eligible"], index=d["station_id"].astype(str), columns=d["years"])
ai = pd.read_csv(OUT / "annual_indices.csv", dtype={"station_id": str}).set_index(["station_id", "year"])
Y = np.arange(1981, 2013); T = len(Y)
IND = {"Rx1day": "Rx1day", "Rx5day": "Rx5day", "R95": "R95pTOT", "PRCPTOT": "PRCPTOT"}
SUB = ["WL", "WM", "CL", "CM", "CH", "EL", "EM"]
REPORTED = {"Rx1day": {"WM": "+", "CL": "+", "CH": "+", "CM": "-", "EM": "-"},
            "Rx5day": {"WM": "+", "CL": "+", "CH": "+", "CM": "-"},
            "R95": {"WM": "+", "CH": "+", "CM": "-", "EM": "-"}}
I_, J_ = np.triu_indices(T, 1)


def mk_vec(M, tt):
    dy = np.sign(M[:, J_] - M[:, I_])
    valid = np.isfinite(dy) & ((tt[J_] - tt[I_]) > 0)[None, :]
    s = np.nansum(np.where(valid, dy, 0), 1)
    n = np.isfinite(M).sum(1)
    var = n * (n - 1) * (2 * n + 5) / 18
    with np.errstate(all="ignore"):
        z = np.where(s != 0, (s - np.sign(s)) / np.sqrt(var), 0)
    p = 2 * (1 - stats.norm.cdf(np.abs(z))); p[n < 15] = np.nan
    return p, s


def tfpw(x, t):
    ok = np.isfinite(x)
    if ok.sum() < 20:
        return np.nan, np.nan, np.nan
    x, t = x[ok], t[ok].astype(float)
    n = len(x)
    sl = stats.theilslopes(x, t)[0]
    det = x - sl * t
    r1 = np.corrcoef(det[:-1], det[1:])[0, 1]
    y = x
    if abs(r1) > 1.96 / np.sqrt(n):
        y = det[1:] - r1 * det[:-1] + sl * t[1:]; tt = t[1:]
    else:
        tt = t
    s = sum(np.sign(y[j] - y[i]) for i in range(len(y) - 1) for j in range(i + 1, len(y)))
    m = len(y); var = m * (m - 1) * (2 * m + 5) / 18
    z = (s - np.sign(s)) / np.sqrt(var) if s else 0
    return 2 * (1 - stats.norm.cdf(abs(z))), s, 100 * 10 * sl / np.median(x)


rows = []
for key, col in IND.items():
    M = np.full((len(K), T), np.nan)
    for a, s in enumerate(K.station_id):
        for t, y in enumerate(Y):
            if s in elig.index and elig.loc[s, y] and (s, y) in ai.index:
                M[a, t] = ai.loc[(s, y), col]
    M[M == 0] = np.nan if key == "PRCPTOT" else M[M == 0]
    keep = np.isfinite(M).sum(1) >= 20
    p_obs, s_obs = mk_vec(M, Y.astype(float))
    tf = [tfpw(M[a], Y) if keep[a] else (np.nan, np.nan, np.nan) for a in range(len(K))]
    p_tf = np.array([x[0] for x in tf]); s_tf = np.array([x[1] for x in tf]); pct = np.array([x[2] for x in tf])
    boots = np.zeros((1000, len(K), 2), bool)
    for b in range(1000):
        idx = rng.integers(T, size=T)
        pb, sb = mk_vec(M[:, idx], Y.astype(float))      # resampled values against the fixed time axis: a no-trend null
        boots[b, :, 0] = (pb < 0.1) & (sb < 0); boots[b, :, 1] = (pb < 0.1) & (sb > 0)
    for sub in SUB:
        m = (K.kreg == sub).values & keep
        if m.sum() < 2:
            continue
        o_neg = int(((p_obs < 0.1) & (s_obs < 0) & m).sum()); o_pos = int(((p_obs < 0.1) & (s_obs > 0) & m).sum())
        c_neg = boots[:, m, 0].sum(1); c_pos = boots[:, m, 1].sum(1)
        def pf(obs, c):
            return ((c >= obs).sum() + 1) / (len(c) + 1)      # P(bootstrap count >= observed), ties included (audit fix)
        pfn, pfp = pf(o_neg, c_neg), pf(o_pos, c_pos)
        field = "none"
        if o_pos > 0 and pfp <= 0.1:
            field = "+"
        if o_neg > 0 and pfn <= 0.1:
            field = "-" if field == "none" else field + "/-"
        rows.append((key, sub, int(m.sum()), int(((p_tf < 0.1) & (s_tf < 0) & m).sum()), int(((p_tf < 0.1) & (s_tf > 0) & m).sum()),
                     float(np.nanmedian(pct[m])), float((s_tf[m] < 0).mean()), o_neg, o_pos, round(pfn, 3), round(pfp, 3), field,
                     REPORTED.get(key, {}).get(sub, "n/r")))
R = pd.DataFrame(rows, columns=["index", "subregion", "n", "st_sig_dec_tfpw", "st_sig_inc_tfpw", "median_pct_dec", "share_declining",
                                "field_obs_neg", "field_obs_pos", "Pf_neg", "Pf_pos", "field_ours", "field_Karki_reported"])
R.to_csv(OUT / "karki_replication.csv", index=False)
pd.set_option("display.width", 220)
print(R.round(2).to_string(index=False))
