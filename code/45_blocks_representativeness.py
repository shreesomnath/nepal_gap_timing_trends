# Code Developer: Somnath Luitel (somnath.luitel416@topper.wku.edu)
"""Step 45: (a) sensitivity of the strict-series field-significance test to the permutation block length; (b) representativeness of the 219 trend stations.

(a) Strict series (eligible years only), local Mann-Kendall decline counts (p < 0.05) against a null from permuting whole years in blocks of 1, 3 (main), 5 and 7 years,
    2 000 permutations each. Output: outputs/block_length_sensitivity_pre2018.csv
(b) Stations with any observation in 1981-2017 split into the 219 trend stations and the others: number, elevation bands, longitude zones, mean share of operating years
    that are partly observed. Output: outputs/representativeness_pre2018.csv
"""
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
src = (ROOT / "code" / "17_pre2018_trends.py").read_text(encoding="utf-8")
head = src.split("t0 = yrs.astype(float)")[0].replace("__file__", repr(str(ROOT / "code" / "17_pre2018_trends.py")))
ns = {"__name__": "prep"}
exec(compile(head, "prep17", "exec"), ns)
RAW, COMP, T, NAMES, used, yrs, mk_local, OUT, d = ns["RAW"], ns["COMP"], ns["T"], ns["NAMES"], ns["used"], ns["yrs"], ns["mk_local"], ns["OUT"], ns["d"]
rng = np.random.default_rng(45)
t0 = yrs.astype(float)
rows = []
for k, name in enumerate(NAMES):
    M = np.where(COMP, RAW[k], np.nan)
    p, s_ = mk_local(M, t0)
    obs = int(((p < 0.05) & (s_ < 0)).sum())
    for bl in (1, 3, 5, 7):
        blocks = [np.arange(s, min(s + bl, T)) for s in range(0, T, bl)]
        nn = []
        for _ in range(2000):
            order = np.concatenate([blocks[j] for j in rng.permutation(len(blocks))])
            pp, ss = mk_local(M[:, order], t0)
            nn.append(((pp < 0.05) & (ss < 0)).sum())
        nn = np.array(nn)
        rows.append((name, bl, obs, float(np.percentile(nn, 95)), float((nn >= obs).mean())))
B = pd.DataFrame(rows, columns=["index", "block_years", "observed_local_declines", "null_95th_percentile", "p_field"])
B.to_csv(OUT / "block_length_sensitivity_pre2018.csv", index=False)
print(B.round(3).to_string(index=False))

status, elig, years, sid = d["status"], d["eligible"], d["years"], d["station_id"].astype(str)
lon = d["lon"].astype(float); pv = pd.read_csv(OUT / "station_province.csv", dtype={"station_id": str}).set_index("station_id")
m_idx = [list(years).index(y) for y in range(1981, 2018)]
gy = pd.to_datetime(d["dates"]).year.to_numpy()
anyobs = np.array([(status[i, (gy >= 1981) & (gy <= 2017)] == 0).any() for i in range(len(sid))])
trend = np.isin(sid, used)
rows = []
for lab, m in (("trend stations", trend), ("other stations with data in 1981-2017", anyobs & ~trend)):
    ids = sid[m]
    el = pv.reindex(ids).elev.to_numpy().astype(float)
    zn = np.where(lon[m] < 83, "West", np.where(lon[m] < 86, "Central", "East"))
    part = []
    for i in np.where(m)[0]:
        ope = [(status[i, gy == y] != 2).any() for y in range(1981, 2018)]
        pr = [(status[i, gy == y] == 0).any() and not elig[i, years.tolist().index(y)] for y in range(1981, 2018)]
        part.append(np.sum(pr) / max(1, np.sum(ope)))
    rows.append((lab, int(m.sum()), int((el < 500).sum()), int(((el >= 500) & (el < 1500)).sum()), int(((el >= 1500) & (el < 2500)).sum()), int((el >= 2500).sum()),
                 int((zn == "West").sum()), int((zn == "Central").sum()), int((zn == "East").sum()), float(np.mean(part))))
Rp = pd.DataFrame(rows, columns=["group", "n", "elev_lt500", "elev_500_1500", "elev_1500_2500", "elev_ge2500", "West", "Central", "East", "mean_share_partly_observed_years"])
Rp.to_csv(OUT / "representativeness_pre2018.csv", index=False)
print(Rp.round(3).to_string(index=False))
