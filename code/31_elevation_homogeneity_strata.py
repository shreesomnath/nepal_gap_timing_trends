# Code Developer: Somnath Luitel (somnath.luitel416@topper.wku.edu)
"""Step 31: station trends (1981-2017) by elevation band and by homogeneity-screen result.

Reads outputs/station_trends_pre2018.csv (step 17) and outputs/homogeneity_pre2018.csv (step 32).
Elevation bands: below 500 m (Terai and Siwaliks), 500-1500 m, 1500-2500 m, 2500 m and above.
For each band (or homogeneity class) and index: number of stations, median strict trend with a 95% station-bootstrap interval,
share declining, and the median paired shift of the uncorrected and corrected series from the strict series.
Outputs: outputs/elevation_trends_pre2018.csv, outputs/homogeneity_trends_pre2018.csv
"""
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
rng = np.random.default_rng(31)
T = pd.read_csv(OUT / "station_trends_pre2018.csv", dtype={"station_id": str})
H = pd.read_csv(OUT / "homogeneity_pre2018.csv", dtype={"station_id": str})[["station_id", "result", "break_year_start", "shift_pct"]]
T = T.merge(H, on="station_id", how="left")
T["result"] = T.result.fillna("not_tested")
T["band"] = pd.cut(T.elev, [-1, 500, 1500, 2500, 9000], labels=["<500 m", "500-1500 m", "1500-2500 m", ">=2500 m"])


def summarise(g, key):
    rows = []
    for n_ in ["Rx1day", "Rx5day", "PRCPTOT", "R95pTOT"]:
        for lab, sub in g:
            s = sub[sub["index"] == n_]
            v = s.R1.dropna().values
            if len(v) < 3:
                continue
            bs = [np.median(rng.choice(v, len(v))) for _ in range(2000)]
            aff = s[(s.R4 - s.R1).abs() > 1e-9]            # stations with at least one partly observed year
            du, dc = (aff.R4 - aff.R1).dropna().values, (aff.Rcorr - aff.R1).dropna().values
            bu = [rng.choice(du, len(du)).mean() for _ in range(2000)] if len(du) > 2 else [np.nan, np.nan]
            bc = [rng.choice(dc, len(dc)).mean() for _ in range(2000)] if len(dc) > 2 else [np.nan, np.nan]
            rows.append((key, str(lab), n_, len(v), np.median(v), *np.percentile(bs, [2.5, 97.5]), float((v < 0).mean()),
                         len(du), du.mean() if len(du) else np.nan, *np.percentile(bu, [2.5, 97.5]),
                         dc.mean() if len(dc) else np.nan, *np.percentile(bc, [2.5, 97.5])))
    return pd.DataFrame(rows, columns=[key, "group", "index", "n", "median_strict", "lo95", "hi95", "share_declining", "n_affected",
                                       "mean_shift_uncorrected", "su_lo", "su_hi", "mean_shift_corrected", "sc_lo", "sc_hi"])


E = summarise(T.groupby("band", observed=True), "elevation_band")
E.to_csv(OUT / "elevation_trends_pre2018.csv", index=False)
Hh = summarise(T.groupby("result"), "homogeneity")
Hh.to_csv(OUT / "homogeneity_trends_pre2018.csv", index=False)
pd.set_option("display.width", 200)
print(E.round(2).to_string(index=False))
print(Hh.round(2).to_string(index=False))
