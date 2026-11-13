# Code Developer: Somnath Luitel (somnath.luitel416@topper.wku.edu)
"""Step 18b: compare station means in this archive with Poudel et al. (2022, J. Hydrometeorol.
23(4), 573-596; doi:10.1175/JHM-D-21-0122.1), Table 4 (mean annual PRCPTOT, Rx1day, Rx5day,
30 of the 34 DHM stations listed in their Table 1; stations 809, 815, 1007 and 420 are not in Table 4; 1986-2015). Our means use eligible years only in 1986-2015 (>= 15 years).
Poudel et al. filled gaps shorter than 5 days by the normal-ratio method; ours are not filled,
so small differences are expected. The table below is transcribed from the paper.
Output: outputs/poudel_station_means_compare.csv
"""
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
T = """604 404.6 43.5 67.6;303 818.5 46.6 90.1;104 1377.9 103.3 195.3;218 1119.8 93.4 177.8;406 1586.4 111.8 203.1;
804 4002.4 191.7 405.8;811 3867.1 190.9 408.5;814 5496.2 203.3 473.9;1029 1202.2 69.1 144.1;1030 1558.9 85.0 170.0;
1103 2478.9 86.4 227.8;1206 1806.6 97.8 207.0;1304 1527.0 96.4 194.2;1307 1014.6 87.8 164.1;1405 1994.3 85.0 173.9;
1407 1798.1 125.4 244.1;1410 2345.8 140.2 268.5;515 1624.4 117.1 215.1;706 2458.3 171.5 329.8;902 1999.3 146.6 273.7;
209 1880.1 160.8 300.6;409 1325.9 120.2 217.6;419 1473.7 123.6 252.8;705 1654.7 156.0 291.5;707 1788.7 162.0 314.5;
909 1849.5 165.2 313.4;911 1564.4 147.2 275.0;1319 1865.7 134.4 272.8;1320 1986.8 147.0 289.4;1421 2610.5 174.3 389.9"""
rows = [r.split() for r in T.replace("\n", "").split(";") if r.strip()]
P = pd.DataFrame(rows, columns=["station_id", "p_PRCPTOT", "p_Rx1day", "p_Rx5day"]).astype({"p_PRCPTOT": float, "p_Rx1day": float, "p_Rx5day": float})
d = np.load(OUT / "daily_qc.npz", allow_pickle=True)
elig = pd.DataFrame(d["eligible"], index=d["station_id"].astype(str), columns=d["years"])
ai = pd.read_csv(OUT / "annual_indices.csv", dtype={"station_id": str})
out = []
for s in P.station_id:
    if s not in elig.index:
        out.append((s, np.nan, np.nan, np.nan, 0)); continue
    ys = [int(y) for y in elig.columns if 1986 <= y <= 2015 and elig.loc[s, y]]
    sub = ai[(ai.station_id == s) & (ai.year.isin(ys))]
    out.append((s, sub.PRCPTOT.mean(), sub.Rx1day.mean(), sub.Rx5day.mean(), len(sub)))
O = pd.DataFrame(out, columns=["station_id", "o_PRCPTOT", "o_Rx1day", "o_Rx5day", "n_years"])
M = P.merge(O, on="station_id")
for k in ["PRCPTOT", "Rx1day", "Rx5day"]:
    M["ratio_" + k] = M["o_" + k] / M["p_" + k]
M.to_csv(OUT / "poudel_station_means_compare.csv", index=False)
pd.set_option("display.width", 220)
print(M.round(2).to_string(index=False))
ok = M[M.n_years >= 15]
print("\nstations compared (>= 15 eligible years):", len(ok), "of", len(M), "; not in archive or too short:", sorted(set(M.station_id) - set(ok.station_id)))
for k in ["PRCPTOT", "Rx1day", "Rx5day"]:
    r = ok["ratio_" + k]
    print(k, "median ratio ours/Poudel %.3f, IQR %.3f-%.3f, corr %.3f, within +-5%%: %d of %d" %
          (r.median(), r.quantile(.25), r.quantile(.75), np.corrcoef(ok["o_" + k], ok["p_" + k])[0, 1],
           ((r > .95) & (r < 1.05)).sum(), len(r)))
