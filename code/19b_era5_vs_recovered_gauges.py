# Code Developer: Somnath Luitel (somnath.luitel416@topper.wku.edu)
"""Step 19b: does ERA5 describe the recovered 2018-2020 gauge days as well as the observed ones?

The two station files supplied later (Precipitation_1980 to 2024 / Precipitation_2020-2024) hold
daily gauge values, including 2018-2020, for stations that are blank in melted_ppt_from_1980.csv
in those years. Where a file station matched an archive station exactly (outputs/external_station_match.csv,
correlation > 0.99), compare ERA5 at the station cell with the gauge on
  recovered : 2018-2020 (archive blank)
  reference : 2021-2023 and, where available, 1981-2017
Statistics per station and period: daily correlation, wet-day agreement (>= 1 mm; hit rate and
false-alarm ratio), ratio of ERA5 to gauge mean, ratio of annual totals and of Rx1day.
This checks that ERA5 is a usable diagnostic for DHM days and that skill is not different in the
recovered years. Gauge data are never modified.
Usage: python code/19b_era5_vs_recovered_gauges.py [--era5-npz ...]
Output: outputs/era5_vs_recovered_gauges.csv
"""
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--era5-npz", default=str(ROOT / "outputs" / "era5_station_daily.npz"))
ap.add_argument("--out", default=str(ROOT / "outputs" / "era5_vs_recovered_gauges.csv"))
a = ap.parse_args()
e = np.load(a.era5_npz, allow_pickle=True)
sid = e["station_id"].astype(str)
dates = pd.DatetimeIndex(e["dates"])
ER = pd.DataFrame(e["era5_centre_mm"].astype(float).T, index=dates, columns=sid)
m = pd.read_csv(ROOT / "outputs" / "external_station_match.csv", dtype={"dhm": str})
m = m[m["corr"] > 0.99]
files = {"1980-2024": ROOT / "Precipitation_1980 to 2024.xlsx - 24h accumulated Precipitation.csv",
         "2020-2024": ROOT / "Precipitation_2020-2024.xlsx - 24h accumulated Precipitation.csv"}
periods = {"recovered_2018-2020": ("2018-01-01", "2020-12-31"), "reference_2021-2023": ("2021-01-01", "2023-12-31"),
           "reference_1981-2017": ("1981-01-01", "2017-12-31")}
rows = []
for _, r in m.iterrows():
    f = pd.read_csv(files[r["file"]])
    f["Time"] = pd.to_datetime(f["Time"], format="%m/%d/%Y %H:%M:%S")
    s = f.set_index(f.Time.dt.normalize())[r["name"]].astype(float)
    for pname, (a0, a1) in periods.items():
        j = pd.concat([s.rename("g"), ER[r["dhm"]].rename("e")], axis=1).loc[a0:a1].dropna()
        if len(j) < 300:
            continue
        wg, we = j.g >= 1, j.e >= 1
        ann = j.groupby(j.index.year).sum()
        cnt = j.groupby(j.index.year).size()
        ann = ann[cnt >= 360]
        rx = j.groupby(j.index.year).max()[cnt >= 360]
        rows.append((r["name"], r["dhm"], pname, len(j), j.g.corr(j.e), j.g.corr(j.e, method="spearman"),
                     (wg & we).sum() / max(wg.sum(), 1), (we & ~wg).sum() / max(we.sum(), 1),
                     j.e.mean() / j.g.mean(), (ann.e / ann.g).median() if len(ann) else np.nan,
                     (rx.e / rx.g).median() if len(rx) else np.nan))
R = pd.DataFrame(rows, columns=["station", "dhm_id", "period", "n_days", "corr", "spearman", "wet_hit_rate",
                                "wet_false_alarm_ratio", "mean_ratio_era5_gauge", "annual_total_ratio_median", "rx1day_ratio_median"])
R.to_csv(a.out, index=False)
pd.set_option("display.width", 220)
print(R.round(3).to_string(index=False))
print("\nmedian by period")
print(R.groupby("period")[["corr", "spearman", "wet_hit_rate", "wet_false_alarm_ratio", "mean_ratio_era5_gauge",
                           "annual_total_ratio_median", "rx1day_ratio_median"]].median().round(3).to_string())
