# Code Developer: Somnath Luitel (somnath.luitel416@topper.wku.edu)
"""Step 34: write daily series for RHtests_dlyPrcp (1981-2017, the 219 trend stations).

Format required by RHtests_dlyPrcp: four whitespace-separated columns (year month day value), missing value code -999.99.
Suspect days are written as missing. Input files go to <work>/input/st_<id>.txt and a list of stations to <work>/stations.txt.
Usage: python code/34_rhtests_prepare.py <work_dir>
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

work = Path(sys.argv[1]); (work / "input").mkdir(parents=True, exist_ok=True)
ROOT = Path(__file__).resolve().parents[1]
d = np.load(ROOT / "outputs" / "daily_qc.npz", allow_pickle=True)
X, status, elig, years, sid = d["X"], d["status"], d["eligible"], d["years"], d["station_id"].astype(str)
dates = pd.DatetimeIndex(pd.to_datetime(d["dates"]))
m = [list(years).index(y) for y in range(1981, 2018)]
sel = np.where(elig[:, m].sum(1) >= 26)[0]
keep = (dates >= "1981-01-01") & (dates <= "2017-12-31")
dd = dates[keep]
ids = []
for i in sel:
    v = np.where(status[i, keep] == 0, X[i, keep], -999.99)
    pd.DataFrame({"y": dd.year, "m": dd.month, "d": dd.day, "v": np.round(v, 2)}).to_csv(
        work / "input" / f"st_{sid[i]}.txt", sep=" ", header=False, index=False)
    ids.append(sid[i])
(work / "stations.txt").write_text("\n".join(ids))
print(len(ids), "station files written to", work / "input")
