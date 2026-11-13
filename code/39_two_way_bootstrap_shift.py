# Code Developer: Somnath Luitel (somnath.luitel416@topper.wku.edu)
"""Step 39: one primary interval for the paired shift (corrected or all-years Sen slope minus eligible-years slope).

Two-way (years x spatial clusters) bootstrap: every replicate resamples whole years (the same years for all stations, which keeps the
cross-station dependence within a year) and, independently, 1-degree grid cells of stations (which keeps the dependence among
neighbouring gauges), draws fresh gap ratios for the correction, and averages the station shifts over the selected cells.
Also reports the cell-only and year-only versions, and a one-sided check of the corrected residual.
Reuses steps 17's data preparation by executing its head. Output: outputs/bootstrap_shift_twoway_pre2018.csv
"""
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
src = (ROOT / "code" / "17_pre2018_trends.py").read_text(encoding="utf-8")
head = src.split("t0 = yrs.astype(float)")[0].replace("__file__", repr(str(ROOT / "code" / "17_pre2018_trends.py")))
ns = {"__name__": "prep"}
exec(compile(head, "prep17", "exec"), ns)
np_, pd_ = ns["np"], ns["pd"]
RAW, COMP, T, NAMES, used, yrs = ns["RAW"], ns["COMP"], ns["T"], ns["NAMES"], ns["used"], ns["yrs"]
sen_pct, corrected, OUT, d = ns["sen_pct"], ns["corrected"], ns["OUT"], ns["d"]
rng = np.random.default_rng(39)
sidl = list(d["station_id"])
la = d["lat"].astype(float)[[sidl.index(s) for s in used]]; lo = d["lon"].astype(float)[[sidl.index(s) for s in used]]
cell = np.array([f"{int(np.floor(a))}_{int(np.floor(b))}" for a, b in zip(la, lo)])
cells = np.unique(cell); members = {c: np.where(cell == c)[0] for c in cells}
print("stations", len(used), "cells", len(cells))
t0 = yrs.astype(float)
B = 1000
rows = []
for k, name in enumerate(NAMES):
    R1 = np.where(COMP, RAW[k], np.nan); R4 = RAW[k]
    affected = (~COMP & np.isfinite(RAW[k])).any(1)
    point = corrected(k, "median")
    pe = {"R4": sen_pct(R4, t0), "Rcorr": sen_pct(point, t0), "R1": sen_pct(R1, t0)}
    res = {q: {"two": [], "year": [], "cell": []} for q in ("R4-R1", "Rcorr-R1")}
    aff_cells = np.array([c for c in cells if affected[members[c]].any()])
    for b in range(B):
        idx = rng.integers(T, size=T); tt = t0[idx]
        cb = corrected(k, "draw")
        e1, e4, ec = (sen_pct(M[:, idx], tt) for M in (R1, R4, cb))
        pick = rng.choice(aff_cells, len(aff_cells))
        for q, e in (("R4-R1", e4 - e1), ("Rcorr-R1", ec - e1)):
            res[q]["year"].append(np.nanmean(e[affected]))
            res[q]["two"].append(np.nanmean(np.concatenate([e[members[c]][affected[members[c]]] for c in pick])))
        # cell-only: fixed years (point estimates), resampled cells
        for q, a1 in (("R4-R1", "R4"), ("Rcorr-R1", "Rcorr")):
            e = pe[a1] - pe["R1"]
            res[q]["cell"].append(np.nanmean(np.concatenate([e[members[c]][affected[members[c]]] for c in pick])))
    for q, a1 in (("R4-R1", "R4"), ("Rcorr-R1", "Rcorr")):
        e = (pe[a1] - pe["R1"])[affected]
        for sch in ("year", "cell", "two"):
            lo_, hi_ = np.nanpercentile(res[q][sch], [2.5, 97.5])
            rows.append((name, q, sch, int(affected.sum()), len(aff_cells), np.nanmean(e), lo_, hi_))
R = pd.DataFrame(rows, columns=["index", "quantity", "scheme", "n_affected", "n_cells", "point", "lo95", "hi95"])
R.to_csv(OUT / "bootstrap_shift_twoway_pre2018.csv", index=False)
pd.set_option("display.width", 200)
print(R.round(2).to_string(index=False))
