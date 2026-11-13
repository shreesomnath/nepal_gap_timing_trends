# Code Developer: Somnath Luitel (somnath.luitel416@topper.wku.edu)
"""Step 35: summarise RHtests_dlyPrcp results (step 34) and compare station trends with and without detected changepoints.

Reads <work>/out_*/st_<id>_1Cs.txt (changepoint list written by FindU.dlyPrcp: a header line with the number of changepoints,
then one line per changepoint: type, 'Yes' (significant at the stated level), date YYYYMMDD, ...). A station is 'flagged' when at least
one changepoint is listed as 'Yes'. Also compares with the relative SNHT flags of step 32.
Outputs: outputs/rhtests_pre2018.csv, outputs/rhtests_trends_pre2018.csv
Usage: python code/35_rhtests_summary.py <work_dir>
"""
import re
import sys
from pathlib import Path
import numpy as np
import pandas as pd

work = Path(sys.argv[1])
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
rows = []
for f in sorted(work.glob("out_*/st_*_1Cs.txt")):
    sid = re.match(r"st_(.+)_1Cs\.txt", f.name).group(1)
    lines = f.read_text(errors="ignore").splitlines()
    n = int(re.match(r"\s*(\d+)\s+changepoints", lines[0]).group(1)) if lines else -1
    cps = []
    for ln in lines[1:]:
        m = re.match(r"\s*(\d)\s+(Yes|\?|No)\s+(\d{8})", ln)
        if m:
            cps.append((m.group(2), m.group(3)))
    yes = [d for t, d in cps if t == "Yes"]
    rows.append((sid, n, len(yes), ";".join(yes), ";".join(d for t, d in cps if t != "Yes")))
R = pd.DataFrame(rows, columns=["station_id", "n_listed", "n_significant", "significant_dates", "other_dates"])
R["rhtests_flag"] = R.n_significant > 0
R.to_csv(OUT / "rhtests_pre2018.csv", index=False)
print("stations processed:", len(R), "| flagged:", int(R.rhtests_flag.sum()), "| none:", int((~R.rhtests_flag).sum()))
yrs = R.significant_dates.str.split(";").explode().dropna()
yrs = yrs[yrs != ""].str[:4].astype(int)
print("significant changepoint years (counts by decade):", yrs.groupby(yrs // 10 * 10).size().to_dict(), "| 2016-2017:", int(((yrs >= 2016)).sum()))

H = pd.read_csv(OUT / "homogeneity_pre2018.csv", dtype={"station_id": str})[["station_id", "result"]].rename(columns={"result": "snht"})
T = pd.read_csv(OUT / "station_trends_pre2018.csv", dtype={"station_id": str}).merge(R[["station_id", "rhtests_flag", "n_significant"]], on="station_id", how="inner").merge(H, on="station_id", how="left")
print(pd.crosstab(T.drop_duplicates("station_id").rhtests_flag, T.drop_duplicates("station_id").snht))
rng = np.random.default_rng(35)
rows = []
for n in ["Rx1day", "Rx5day", "PRCPTOT", "R95pTOT"]:
    for flag, lab in [(False, "no changepoint"), (True, "changepoint flagged")]:
        s = T[(T["index"] == n) & (T.rhtests_flag == flag)]
        v = s.R1.dropna().values
        if len(v) < 3:
            continue
        bs = [np.median(rng.choice(v, len(v))) for _ in range(2000)]
        aff = s[(s.R4 - s.R1).abs() > 1e-9]          # stations with at least one partly observed year
        du, dc = (aff.R4 - aff.R1).dropna().values, (aff.Rcorr - aff.R1).dropna().values
        bu = [rng.choice(du, len(du)).mean() for _ in range(2000)] if len(du) > 2 else [np.nan, np.nan]
        bc = [rng.choice(dc, len(dc)).mean() for _ in range(2000)] if len(dc) > 2 else [np.nan, np.nan]
        rows.append((n, lab, len(v), np.median(v), *np.percentile(bs, [2.5, 97.5]), float((v < 0).mean()),
                     len(du), du.mean() if len(du) else np.nan, *np.percentile(bu, [2.5, 97.5]),
                     dc.mean() if len(dc) else np.nan, *np.percentile(bc, [2.5, 97.5])))
S = pd.DataFrame(rows, columns=["index", "group", "n", "median_strict", "lo95", "hi95", "share_declining", "n_affected",
                                "mean_shift_uncorrected", "su_lo", "su_hi", "mean_shift_corrected", "sc_lo", "sc_hi"])
S.to_csv(OUT / "rhtests_trends_pre2018.csv", index=False)
pd.set_option("display.width", 200)
print(S.round(2).to_string(index=False))
