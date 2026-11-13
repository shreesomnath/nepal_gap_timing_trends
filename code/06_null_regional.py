# Code Developer: Somnath Luitel (somnath.luitel416@topper.wku.edu)
"""Step 6: (a) can the archive's gap pattern alone create significant trends?
            (b) does the east-west contrast survive completeness rules and a fixed network?

(a) Null experiment. For each station used in step 3 (R1 rule), the complete years are
resampled at random into every year 1981-2023 in which the station operated, which
removes any real trend. Years that were partly observed in the archive then receive
that station's actual missingness pattern for that year, so the timing and shape of
gaps are real while the climate signal is absent. Trends are estimated under R1
(complete years only) and R4 (every year with data). Repeated 100 times.

(b) Province is assigned by point-in-polygon on the official local-unit boundary after
transforming station coordinates (assumed WGS84) to the boundary's Everest 1830 datum.

Outputs: outputs/null_trends.csv, outputs/regional_trends.csv, gis/stations_wgs84.shp
"""
import numpy as np
import pandas as pd
import shapefile
from scipy import stats
from shapely.geometry import shape, Point
from shapely.strtree import STRtree
from pyproj import CRS, Transformer
from pathlib import Path

rng = np.random.default_rng(11)
ROOT = Path(__file__).resolve().parents[1]
OUT, GIS = ROOT / "outputs", ROOT / "gis"
d = np.load(OUT / "daily_qc.npz", allow_pickle=True)
X, status, elig, years = d["X"], d["status"], d["eligible"], d["years"]
sid = d["station_id"]
dates = pd.to_datetime(d["dates"])
yr = dates.year.to_numpy()
V = np.where(status == 0, X, np.nan)
p95 = pd.read_csv(OUT / "p95_baseline.csv", dtype={"station_id": str}).p95_1981_2010.to_numpy()
tr = pd.read_csv(OUT / "trend_by_rule.csv", dtype={"station_id": str})
used = set(tr[(tr.rule == "R1_strict") & (tr["index"] == "Rx1day")].station_id)
PER = (years >= 1981) & (years <= 2023)


def yv(i, y):
    v = V[i, yr == y]
    return np.delete(v, 59) if v.size == 366 else v


def idx(v, p):
    wet = v[v >= 1]
    return np.nanmax(v), wet.sum(), wet[wet > p].sum()


def mk_p(y):
    n = len(y)
    s = np.sum(np.sign(y[None, :] - y[:, None])[np.triu_indices(n, 1)])
    var = n * (n - 1) * (2 * n + 5) / 18
    z = (s - np.sign(s)) / np.sqrt(var) if s else 0.0
    return 2 * (1 - stats.norm.cdf(abs(z)))


def fdr(p, q=0.10):
    p = np.asarray(p); o = np.argsort(p)
    ok = p[o] <= q * np.arange(1, len(p) + 1) / len(p)
    k = np.where(ok)[0].max() + 1 if ok.any() else 0
    s = np.zeros(len(p), bool); s[o[:k]] = True
    return s


# ---------------- (a) null experiment ----------------
st_idx = [i for i in range(len(sid)) if sid[i] in used and np.isfinite(p95[i])]
info = []
for i in st_idx:
    comp, part = [], {}
    for j, y in enumerate(years):
        if not PER[j] or (status[i, yr == y] == 2).all():
            continue
        v = yv(i, y)
        if elig[i, j]:
            comp.append(y)
        elif np.isfinite(v).any():
            part[y] = ~np.isfinite(v)
    pool = np.array([yv(i, y) for y in comp])
    info.append((i, comp, part, pool))

NAMES = ["Rx1day", "PRCPTOT", "R95pTOT"]
rows = []
for rep in range(100):
    res = {(r, n): [] for r in ("R1", "R4") for n in NAMES}
    for i, comp, part, pool in info:
        allyears = sorted(comp + list(part))
        draw = pool[rng.integers(len(pool), size=len(allyears))]
        vals, isc = [], []
        for y, v in zip(allyears, draw):
            v = v.copy()
            if y in part:
                v[part[y]] = np.nan
            vals.append(idx(v, p95[i])); isc.append(y not in part)
        vals, isc, t = np.array(vals), np.array(isc), np.array(allyears, float)
        for k, n in enumerate(NAMES):
            for r, m in (("R1", isc), ("R4", np.ones_like(isc))):
                if m.sum() >= 30:
                    yy = vals[m, k]
                    res[(r, n)].append((stats.theilslopes(yy, t[m])[0] * 10 / np.median(yy) * 100,
                                        mk_p(yy)))
    for (r, n), lst in res.items():
        a = np.array(lst)
        sig = fdr(a[:, 1])
        rows.append((rep, r, n, len(a), np.median(a[:, 0]), (a[:, 0] < 0).mean(),
                     int(((a[:, 1] < 0.05) & (a[:, 0] < 0)).sum()),
                     int(((a[:, 1] < 0.05) & (a[:, 0] > 0)).sum()),
                     int((sig & (a[:, 0] < 0)).sum())))
nul = pd.DataFrame(rows, columns=["rep", "rule", "index", "n", "median_pct_dec", "share_neg",
                                  "sig_neg_local", "sig_pos_local", "sig_neg_fdr"])
nul.to_csv(OUT / "null_trends.csv", index=False)
print("NULL experiment (no real trend), mean over 100 replicates:")
print(nul.groupby(["index", "rule"])[["n", "median_pct_dec", "share_neg", "sig_neg_local",
                                      "sig_pos_local", "sig_neg_fdr"]].mean().round(2).to_string())
q = nul.groupby(["index", "rule"]).sig_neg_fdr.quantile(0.95)
print("95th percentile of FDR-significant negative trends under null:\n", q.to_string())

# ---------------- (b) provinces and regional trends ----------------
r = shapefile.Reader(str(GIS / "Local Unit" / "local_unit"))
geoms = [shape(s.__geo_interface__) for s in r.shapes()]
recs = r.records()
tree = STRtree(geoms)
ev = CRS.from_proj4("+proj=longlat +a=6377276.345 +rf=300.8017 +towgs84=293.17,726.18,245.36 +no_defs")
to_ev = Transformer.from_crs(CRS.from_epsg(4326), ev, always_xy=True)
PNAME = {"1": "Koshi", "2": "Madhesh", "5": "Lumbini"}
prov, dist_ = [], []
for lo, la in zip(d["lon"], d["lat"]):
    p = Point(*to_ev.transform(lo, la))
    h = [k for k in tree.query(p) if geoms[k].contains(p)]
    rec = recs[h[0]] if h else None
    prov.append(PNAME.get(rec["Province"], rec["Province"]) if rec else "outside")
    dist_.append(rec["DISTRICT"] if rec else "")
stmeta = pd.DataFrame({"station_id": sid, "province": prov, "district": dist_,
                       "lon": d["lon"], "lat": d["lat"], "elev": d["elev"]})
stmeta.to_csv(OUT / "station_province.csv", index=False)
WEST = {"Sudur Pashchim", "Karnali", "Lumbini"}
stmeta["region"] = np.where(stmeta.province.isin(WEST), "West", np.where(stmeta.province == "Gandaki", "Central", "East"))

t2 = tr.merge(stmeta, on="station_id")
fixed = set.intersection(*[set(tr[(tr.rule == k) & (tr["index"] == "Rx1day")].station_id)
                           for k in tr.rule.unique()])
t2["network"] = np.where(t2.station_id.isin(fixed), "fixed", "added")
reg = t2.groupby(["index", "rule", "region"]).agg(n=("station_id", "count"),
                                                  median_pct_dec=("slope_pct_dec", "median"),
                                                  share_pos=("slope_dec", lambda s: (s > 0).mean()),
                                                  sig_fdr=("sig_fdr", "sum"))
reg.to_csv(OUT / "regional_trends.csv")
print("\nRegional trends by rule (all stations passing each rule):")
print(reg.round(2).unstack("region").to_string())

# 2018-2020 dropout geography by region
drop = []
for i in range(len(sid)):
    a17 = (status[i, yr == 2017] == 0).mean() > 0.5
    a18 = (status[i, (yr >= 2018) & (yr <= 2019)] == 0).mean() > 0.5
    drop.append("dropped" if a17 and not a18 else ("kept" if a17 and a18 else "other"))
stmeta["dropout_2018_19"] = drop
print("\n2018-19 dropout by region:\n", pd.crosstab(stmeta.region, stmeta.dropout_2018_19).to_string())

# ---------------- GIS export (WGS84) ----------------
inv = pd.read_csv(OUT / "station_inventory.csv", dtype={"Station_ID": str})
w = shapefile.Writer(str(GIS / "stations_wgs84"), shapeType=shapefile.POINT)
w.field("STN_ID", "C", 10); w.field("PROVINCE", "C", 20); w.field("DISTRICT", "C", 30)
w.field("ELEV_M", "N", 8, 0); w.field("ELIG_YRS", "N", 4, 0); w.field("COMPLETE", "N", 6, 3)
w.field("DROP1819", "C", 8)
for k in ("Rx1day", "PRCPTOT"):
    for rr in ("R1", "R4"):
        w.field(f"{k[:4]}_{rr}", "N", 8, 2)   # trend, % per decade
piv = tr.assign(r=tr.rule.str[:2]).pivot_table(index="station_id", columns=["index", "r"],
                                                values="slope_pct_dec")
for i, s in enumerate(sid):
    m = stmeta.iloc[i]; iv = inv.iloc[i]
    w.point(float(m.lon), float(m.lat))
    vals = []
    for k in ("Rx1day", "PRCPTOT"):
        for rr in ("R1", "R4"):
            v = piv[(k, rr)].get(s, np.nan) if (k, rr) in piv.columns else np.nan
            vals.append(None if pd.isna(v) else round(float(v), 2))
    w.record(s, m.province, m.district, int(m.elev), int(iv.eligible_years),
             float(iv.active_completeness), m.dropout_2018_19, *vals)
w.close()
(GIS / "stations_wgs84.prj").write_text(CRS.from_epsg(4326).to_wkt(version="WKT1_ESRI"))
print("\nGIS layer written:", GIS / "stations_wgs84.shp")
