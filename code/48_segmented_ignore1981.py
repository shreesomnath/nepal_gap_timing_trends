# Code Developer: Somnath Luitel (somnath.luitel416@topper.wku.edu)
"""Step 48: break-aware trends when RHtests changepoints in 1981, the first analysis year, are ignored (referee check).

Reuses the set-up of 36_segmented_trends_and_rules.py (indices, treatments, within-segment Sen slope) and recomputes
the all-station segmented strict medians and segmented paired shifts (stations with at least 26 eligible years) with
(a) all significant changepoints, as in step 36, and (b) changepoints dated 1981 removed.
Output: outputs/segmented_ignore1981_pre2018.csv
"""
from pathlib import Path

SRC = Path(__file__).with_name("36_segmented_trends_and_rules.py").read_text(encoding="utf-8")
HEAD = SRC.split("# same-segment pair mask per station")[0]
exec(compile(HEAD, "step36_head", "exec"))


def same_mask(drop_years=()):
    M = np.ones((S, len(I_)), bool)
    n_brk = 0
    for i in range(S):
        if sid[i] in RH.index and RH.loc[sid[i], "significant_dates"]:
            brk = [int(x[:4]) for x in RH.loc[sid[i], "significant_dates"].split(";")]
            brk = [b for b in brk if b not in drop_years]
            n_brk += len(brk)
            if not brk:
                continue
            seg = np.array([sum(b < y for b in brk) for y in yrs], float)
            seg[np.isin(yrs, brk)] = np.nan
            M[i] = (seg[I_] == seg[J_]) & np.isfinite(seg[I_]) & np.isfinite(seg[J_])
    return M, n_brk


sel = n_elig >= 26
gm = sel & np.isin(sid, RH.index)
rows = []
for label, drop in (("all changepoints", ()), ("1981 changepoints ignored", (1981,))):
    SAME, nb = same_mask(drop)
    rng = np.random.default_rng(48)
    for k, name in enumerate(NAMES):
        M = {"R1": np.where(COMP, RAW[k], np.nan), "R4": RAW[k], "R4corr": corrected(k)}
        seg_ = {t: sen_pct(M[t], SAME) for t in M}
        v1 = seg_["R1"][gm]; v1f = v1[np.isfinite(v1)]
        bs1 = [np.median(rng.choice(v1f, len(v1f))) for _ in range(1000)]
        aff = gm & np.isfinite(seg_["R4"]) & np.isfinite(seg_["R1"]) & ((M["R4"] != M["R1"]) & np.isfinite(M["R4"])).any(1)
        du = (seg_["R4"] - seg_["R1"])[aff]; dc = (seg_["R4corr"] - seg_["R1"])[aff]
        rows.append((label, nb, name, len(v1f), np.median(v1f), *np.percentile(bs1, [2.5, 97.5]), int(aff.sum()),
                     du.mean(), *np.percentile([rng.choice(du, len(du)).mean() for _ in range(1000)], [2.5, 97.5]),
                     dc.mean(), *np.percentile([rng.choice(dc, len(dc)).mean() for _ in range(1000)], [2.5, 97.5])))
R = pd.DataFrame(rows, columns=["version", "n_changepoints", "index", "n_seg", "median_segmented", "s_lo", "s_hi",
                                "n_affected", "shift_uncorr_seg", "u_lo", "u_hi", "shift_corr_seg", "c_lo", "c_hi"])
R.to_csv(OUT / "segmented_ignore1981_pre2018.csv", index=False)
pd.set_option("display.width", 240)
print(R.round(2).to_string(index=False))
