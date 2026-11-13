# Code Developer: Somnath Luitel (somnath.luitel416@topper.wku.edu)
"""Step 47: checks requested by the pre-submission review (ESS).

1. Common denominator: paired shifts with every treatment normalized by the eligible-year (T1) station median,
   so the shift is a pure slope difference.
2. Serial dependence: two-way bootstrap with circular moving blocks of whole years (lengths 3 and 5) instead of
   independent years, plus 1-degree cell resampling, as in step 39.
3. Adjustment versus selection: T3 without the 0.2 correction-ratio exclusion, and T2 restricted to the years
   that T3 retains.
Reuses the data preparation of step 17 (as step 39 does). Output: outputs/review_checks_pre2018.csv
"""
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
src = (ROOT / "code" / "17_pre2018_trends.py").read_text(encoding="utf-8")
head = src.split("t0 = yrs.astype(float)")[0].replace("__file__", repr(str(ROOT / "code" / "17_pre2018_trends.py")))
ns = {"__name__": "prep"}
exec(compile(head, "prep17", "exec"), ns)
RAW, COMP, T, NAMES, used, yrs = ns["RAW"], ns["COMP"], ns["T"], ns["NAMES"], ns["used"], ns["yrs"]
RATIOS, I, J, OUT, d = ns["RATIOS"], ns["I"], ns["J"], ns["OUT"], ns["d"]
rng = np.random.default_rng(47)
B = 1000

sidl = list(d["station_id"])
la = d["lat"].astype(float)[[sidl.index(s) for s in used]]; lo = d["lon"].astype(float)[[sidl.index(s) for s in used]]
cell = np.array([f"{int(np.floor(a))}_{int(np.floor(b))}" for a, b in zip(la, lo)])
cells = np.unique(cell); members = {c: np.where(cell == c)[0] for c in cells}
t0 = yrs.astype(float)


def corrected(k, mode, exclude=True):
    M = RAW[k].copy()
    for (a_, t), r in RATIOS.items():
        rr = r[k][np.isfinite(r[k]) & (r[k] > 0)]
        if rr.size == 0:
            M[a_, t] = np.nan
            continue
        g = np.median(rr) if mode == "median" else rr[rng.integers(rr.size)]
        M[a_, t] = M[a_, t] / g if (g >= 0.2 or not exclude) else np.nan
    return M


def sen_raw(M, tt):
    dy = M[:, J] - M[:, I]
    dx = (tt[J] - tt[I])[None, :]
    with np.errstate(all="ignore"):
        sl = np.where(dx != 0, dy / dx, np.nan)
        return np.nanmedian(sl, 1) * 10


def norm(slope, M):
    with np.errstate(all="ignore"):
        med = np.nanmedian(M, 1)
        return np.where(med > 0, slope / med * 100, np.nan)


def block_idx(L):
    if L == 1:
        return rng.integers(T, size=T)
    starts = rng.integers(T, size=int(np.ceil(T / L)))
    return np.concatenate([(s + np.arange(L)) % T for s in starts])[:T]


def cell_mean(e, affected, pick):
    return np.nanmean(np.concatenate([e[members[c]][affected[members[c]]] for c in pick]))


rows = []
for k, name in enumerate(NAMES):
    R1 = np.where(COMP, RAW[k], np.nan); R2 = RAW[k]
    affected = (~COMP & np.isfinite(RAW[k])).any(1)
    aff_cells = np.array([c for c in cells if affected[members[c]].any()])
    R3 = corrected(k, "median")
    R3n = corrected(k, "median", exclude=False)
    R2r = np.where(COMP | np.isfinite(R3), R2, np.nan)          # T2 on the years T3 retains
    s = {n: sen_raw(M, t0) for n, M in (("R1", R1), ("R2", R2), ("R3", R3), ("R3n", R3n), ("R2r", R2r))}
    own = {n: norm(s[n], M) for n, M in (("R1", R1), ("R2", R2), ("R3", R3), ("R3n", R3n), ("R2r", R2r))}
    com = {n: norm(s[n], R1) for n in s}
    point = {
        "own: T2-T1": own["R2"] - own["R1"], "own: T3-T1": own["R3"] - own["R1"],
        "common: T2-T1": com["R2"] - com["R1"], "common: T3-T1": com["R3"] - com["R1"],
        "own: T3 no exclusion-T1": own["R3n"] - own["R1"], "own: T2 on T3 years-T1": own["R2r"] - own["R1"],
    }
    reps = {(q, L): [] for q in point for L in (1, 3, 5)}
    for L in (1, 3, 5):
        for b in range(B):
            idx = block_idx(L); tt = t0[idx]
            pick = rng.choice(aff_cells, len(aff_cells))
            Mb = {"R1": R1[:, idx], "R2": R2[:, idx], "R3": corrected(k, "draw")[:, idx],
                  "R3n": corrected(k, "draw", exclude=False)[:, idx]}
            Mb["R2r"] = np.where(np.isfinite(Mb["R1"]) | np.isfinite(Mb["R3"]), Mb["R2"], np.nan)
            sb = {n: sen_raw(M, tt) for n, M in Mb.items()}
            ob = {n: norm(sb[n], Mb[n]) for n in sb}
            cb = {n: norm(sb[n], Mb["R1"]) for n in sb}
            vals = {
                "own: T2-T1": ob["R2"] - ob["R1"], "own: T3-T1": ob["R3"] - ob["R1"],
                "common: T2-T1": cb["R2"] - cb["R1"], "common: T3-T1": cb["R3"] - cb["R1"],
                "own: T3 no exclusion-T1": ob["R3n"] - ob["R1"], "own: T2 on T3 years-T1": ob["R2r"] - ob["R1"],
            }
            for q, e in vals.items():
                reps[(q, L)].append(cell_mean(e, affected, pick))
    for q, e in point.items():
        row = [name, q, int(affected.sum()), np.nanmean(e[affected])]
        for L in (1, 3, 5):
            row += list(np.nanpercentile(reps[(q, L)], [2.5, 97.5]))
        rows.append(row)
    print(name, "done", flush=True)

R = pd.DataFrame(rows, columns=["index", "quantity", "n_affected", "point", "lo_iid", "hi_iid", "lo_block3", "hi_block3",
                                "lo_block5", "hi_block5"])
R.to_csv(OUT / "review_checks_pre2018.csv", index=False)
pd.set_option("display.width", 220)
print(R.round(2).to_string(index=False))
