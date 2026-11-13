# Code Developer: Somnath Luitel (somnath.luitel416@topper.wku.edu)
"""Step 18: compare with Karki et al. (2017, Climate 5(1), 4; doi:10.3390/cli5010004).

(1) Station ID check: the 76 stations of Karki et al. Table 1 (ID, region, lat, lon) against
    the same IDs in this archive (coordinates, whether data exist).
(2) Regional replication: median Sen slope (% of station median per decade) of Rx1day, Rx5day,
    R95pTOT and PRCPTOT on the Karki station set and Karki sub-regions, using our strict series
    (eligible years only), for 1981-2012 (Karki: 1970-2012; our archive starts 1980) and 1981-2017.
    Karki sub-regions: West/Central/East at 83 and 86 deg E; WL, WM, CL, CM, CH, EL, EM.
The table below is transcribed from the paper's Table 1.
Outputs: outputs/karki_station_check.csv, outputs/karki_regional_compare.csv
"""
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
K = """WL 106 28.683 80.35;WL 209 28.8 80.55;WL 416 28.052 81.523;WL 510 27.7 82.533;
WM 101 29.65 80.5;WM 103 29.467 80.533;WM 104 29.3 80.583;WM 201 29.617 80.867;WM 202 29.55 81.217;WM 203 29.267 80.983;WM 206 28.953 81.442;WM 303 29.275 82.18;WM 308 29.2 81.9;WM 402 28.85 81.717;WM 404 28.7 82.2;WM 406 28.587 81.635;WM 504 28.3 82.633;WM 511 28.383 82.167;
CL 703 27.694 83.466;CL 902 27.654 84.351;CL 903 27.591 84.522;CL 907 27.281 84.992;CL 909 27.164 84.98;CL 910 27.183 85.167;CL 911 27.079 84.933;CL 912 27.017 85.383;CL 1109 27.1 85.66;CL 1110 27.013 85.921;CL 1111 26.711 85.924;
CM 701 27.95 83.433;CM 722 28.167 83.267;CM 802 28.283 84.367;CM 804 28.2 83.979;CM 807 28.133 84.35;CM 808 27.942 84.406;CM 810 27.883 83.817;CM 814 28.297 83.818;CM 904 27.56 85.139;CM 1008 27.813 85.625;CM 1015 27.688 85.221;CM 1022 27.593 85.379;CM 1023 27.639 85.705;CM 1029 27.652 85.326;CM 1030 27.704 85.373;CM 1115 27.42 85.849;
CH 601 28.78 83.72;CH 604 28.739 83.681;CH 607 28.633 83.609;
EL 1112 26.93 86.145;EL 1213 26.933 86.517;EL 1216 26.656 86.212;EL 1311 26.792 87.285;EL 1316 26.82 87.159;EL 1319 26.481 87.264;EL 1320 26.699 87.279;EL 1408 26.671 87.703;EL 1409 26.625 87.989;
EM 1102 27.667 86.05;EM 1103 27.633 86.233;EM 1202 27.7 86.717;EM 1203 27.443 86.569;EM 1204 27.36 86.749;EM 1206 27.308 86.504;EM 1207 27.215 86.444;EM 1210 27.136 86.43;EM 1211 27.029 86.843;EM 1303 27.292 87.317;EM 1305 27.154 87.289;EM 1306 27.05 87.244;EM 1307 26.983 87.346;EM 1308 26.932 87.32;EM 1309 26.914 87.16;EM 1322 26.938 87.155;EM 1325 27.353 87.146;EM 1403 27.55 87.783;EM 1410 26.887 88.027"""
rows = [r.split() for r in K.replace("\n", "").split(";") if r.strip()]
kk = pd.DataFrame(rows, columns=["kreg", "station_id", "klat", "klon"]).astype({"klat": float, "klon": float})
assert len(kk) == 76, len(kk)

inv = pd.read_csv(OUT / "station_inventory.csv", dtype={"Station_ID": str}).rename(columns={"Station_ID": "station_id"})
chk = kk.merge(inv[["station_id", "Latitude", "Longitude", "Elevation", "first_obs", "last_obs", "eligible_years"]], on="station_id", how="left")
chk["dist_km"] = np.hypot(chk.Latitude - chk.klat, (chk.Longitude - chk.klon) * np.cos(np.radians(chk.klat))) * 111.2
chk.to_csv(OUT / "karki_station_check.csv", index=False)
print("Karki stations found in archive:", chk.Latitude.notna().sum(), "of 76")
print("distance (km) between Karki and archive coordinates: median %.2f, max %.2f, > 3 km: %d" %
      (chk.dist_km.median(), chk.dist_km.max(), (chk.dist_km > 3).sum()))
print(chk[chk.dist_km > 3][["station_id", "kreg", "klat", "klon", "Latitude", "Longitude", "dist_km"]].round(3).to_string(index=False))
print("archive stations with no data after 2017 among Karki set:",
      int((pd.to_datetime(chk.last_obs) < "2018-01-01").sum()))

d = np.load(OUT / "daily_qc.npz", allow_pickle=True)
elig = pd.DataFrame(d["eligible"], index=d["station_id"].astype(str), columns=d["years"])
ai = pd.read_csv(OUT / "annual_indices.csv", dtype={"station_id": str})
ai = ai.set_index(["station_id", "year"])


def sen_pct(s, idx, y0, y1, min_years=20):
    ys = [int(y) for y in elig.columns if y0 <= y <= y1 and s in elig.index and elig.loc[s, y]]
    if len(ys) < min_years:
        return np.nan
    v = np.array([ai.loc[(s, y), idx] if (s, y) in ai.index else np.nan for y in ys], float)
    ok = np.isfinite(v)
    if ok.sum() < min_years or np.nanmedian(v[ok]) <= 0:
        return np.nan
    return 100 * 10 * stats.theilslopes(v[ok], np.array(ys)[ok])[0] / np.nanmedian(v[ok])


def broad(r):
    return {"W": "West", "C": "Central", "E": "East"}[r[0]]


res = []
for (y0, y1) in [(1981, 2012), (1981, 2017)]:
    for idx in ["Rx1day", "Rx5day", "R95pTOT", "PRCPTOT"]:
        kk["t"] = [sen_pct(s, idx, y0, y1) for s in kk.station_id]
        for label, grp in [("All Karki stations", kk)] + [(r, g) for r, g in kk.groupby("kreg")] + \
                          [(broad(r) + " (3 zones)", g) for r, g in kk.groupby(kk.kreg.map(broad))]:
            v = grp.t.dropna()
            res.append((f"{y0}-{y1}", idx, label, len(grp), len(v), v.median() if len(v) else np.nan,
                        (v < 0).mean() if len(v) else np.nan))
r = pd.DataFrame(res, columns=["period", "index", "region", "n_karki_stations", "n_with_trend", "median_pct_dec", "share_declining"])
r.to_csv(OUT / "karki_regional_compare.csv", index=False)
pd.set_option("display.width", 200)
print(r[r["index"] == "Rx1day"].round(2).to_string(index=False))
print(r[(r["index"] == "R95pTOT") & r.region.str.contains("3 zones|All")].round(2).to_string(index=False))
