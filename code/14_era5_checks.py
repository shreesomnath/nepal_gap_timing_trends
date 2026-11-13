# Code Developer: Somnath Luitel (somnath.luitel416@topper.wku.edu)
"""Step 14: ERA5 diagnostics for (A) informative missingness and (B) unusual 2018-2020.


IMPORTANT. ERA5 is a model reanalysis. It is used here only to answer two diagnostic
questions about the gauge network and never as a reference for gauge accuracy. Per the
ECMWF ERA5 data documentation (Observations, Tables 14-16) its only precipitation inputs
are satellite microwave radiances and ground-based radar precipitation composites; surface
SYNOP/METAR reports enter for pressure, temperature and humidity, not rainfall. DHM gauge
totals are not listed as an input. That is the basis for treating ERA5 as independent of
the gauges (see outputs/era5_download_spec.md for the sources and limits of that check).

Inputs (read only):
  era5/*.nc                    hourly ERA5 total_precipitation (metres), see 14a_era5_download.py
  outputs/daily_qc.npz         gauge values, status codes (0 observed, 1 missing,
                               2 outside operating period, 3 suspect), dates, coordinates
  outputs/station_province.csv province per station (regions as in 06_null_regional.py)
  gis/Local Unit/local_unit.*  used only for the Nepal cell mask of the area mean

Day boundary. DHM reads its daily gauges at 08:45 Nepal time, which is 03:00 UTC, so the
default ERA5 "day D" is the 24 hourly accumulations ending 03 UTC on date D (stamps
D-1 04:00 UTC to D 03:00 UTC). ERA5 hourly total_precipitation stamped t is the accumulation
over the hour ending at t. Whether the archive labels the reading with the date of the
reading (assumed) or the previous date is not documented; --label-shift -1 tests the other
reading and --day-end-utc 0 --label-shift -1 gives the plain UTC calendar day.

Test A (informative missingness). For every station-day with status 1 (missing inside the
operating period) we compare ERA5 on that day with ERA5 on days the same station was
observed (status 0), in a +/-15 day window of day of year:
  same_year    : reference days from the same calendar year (year-matched). Reference needs
                 at least --min-ref observed days; otherwise the day is not used.
  other_years  : reference days from other years, excluding 2018-2020, for gaps where the
                 same year offers no observed days (the 2018-2019 outage).
Gap types: short (run <= 30 days), long (31-365 days), outage (status 1 days in 2018-2019 at
stations observed in 2017 but not in 2018-2019). Gaps longer than 365 days outside the
outage are not analysed. Statistics: ratio of mean ERA5 rain (missing / reference), difference
in wet-day probability (ERA5 >= --wet-mm), difference in the probability that ERA5 exceeds
its own station-cell 95th percentile of wet days in 1981-2017. Confidence intervals are
circular block bootstrap over years (blocks of --block years) for gap types analysed
within years, and bootstrap over stations for the outage, which spans only two years.
Because the outage cannot be compared within a year, it is reported relative to a control:
the same statistics for observed 2018-2019 days at stations that kept reporting
('outage_DiD' = outage ratio / control ratio, and difference of differences).

Test B (were 2018-2020 unusual). Nepal-wide area-mean (cos-latitude weighted, cells whose
centre lies inside Nepal) ERA5 annual total, annual Rx1day and June-September total, with
rank (1 = largest) and standardised anomaly against 1981-2017; the same indices averaged over
stations that dropped out in 2018-2019 versus stations that kept reporting.

Gauge data are never modified. Outputs:
  outputs/era5_station_daily.npz        daily ERA5 mm at each station cell and 3x3 mean
  outputs/era5_testA_missingness.csv
  outputs/era5_testB_nepal_annual.csv, era5_testB_station_groups.csv, era5_testB_summary.csv
  figures/fig7_era5_missingness.pdf and .png
"""
import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

EAST = {"Koshi", "Madhesh", "Bagmati"}
CENTRAL = {"Gandaki"}
WEST = {"Lumbini", "Karnali", "Sudur Pashchim", "Sudurpashchim"}
REGIONS = ["All", "East", "Central", "West"]
CATS = ["short", "long", "outage", "control_kept"]  # index 0..3 = category code 1..4
MODES = ["same_year", "other_years"]
Y0, Y1 = 1980, 2023
NY = Y1 - Y0 + 1
BASE = (1981, 2017)
FIRST_YEAR = 1981

BLUE, ORANGE, GREEN, PURPLE, VERM = "#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00"


# ----------------------------------------------------------------------------- ERA5 reading
def read_tp(path):
    """Return times, lat (ascending), lon, hourly precipitation in mm from one ERA5 NetCDF."""
    import xarray as xr
    ds = xr.open_dataset(path)
    name = "tp" if "tp" in ds else list(ds.data_vars)[0]
    da = ds[name]
    units = str(da.attrs.get("units", "m"))
    if units not in ("m", "m of water equivalent"):
        raise ValueError(f"{path}: unexpected units '{units}' for {name} (expected metres)")
    tdim = "valid_time" if "valid_time" in da.dims else "time"
    if "expver" in da.dims:  # files mixing final ERA5 (0001) and ERA5T (0005)
        da = da.isel(expver=0).combine_first(da.isel(expver=1))
    da = da.reset_coords(drop=True)
    latn = "latitude" if "latitude" in da.dims else "lat"
    lonn = "longitude" if "longitude" in da.dims else "lon"
    da = da.transpose(tdim, latn, lonn).sortby(latn)
    vals = np.clip(da.values.astype("float64") * 1000.0, 0, None)  # tiny negatives are rounding
    times = pd.DatetimeIndex(da[tdim].values)
    out = times, da[latn].values, da[lonn].values, vals
    ds.close()
    return out


def aggregate_hourly(times, vals, end_hour, shift):
    """Sum hourly stamps into days ending at end_hour UTC. Returns labels, sums, counts."""
    u = times - pd.Timedelta(hours=end_hour)
    label = (u - pd.Timedelta(seconds=1)).floor("D") + pd.Timedelta(days=1 + shift)
    codes, uniq = pd.factorize(label)
    sums = np.zeros((len(uniq),) + vals.shape[1:])
    for k in range(len(uniq)):
        sums[k] = vals[codes == k].sum(0)
    counts = np.bincount(codes, minlength=len(uniq))
    return uniq, sums, counts


def build_daily(era5_dir, dates, end_hour, shift):
    """Daily ERA5 (T, ny, nx) in mm on the station dates, NaN where fewer than 24 hours exist."""
    files = sorted(Path(era5_dir).glob("*.nc"))
    if not files:
        sys.exit(f"No .nc files in {era5_dir}. Download them first (outputs/era5_download_spec.md).")
    acc, lat, lon = {}, None, None
    for f in files:
        t, la, lo, v = read_tp(f)
        if lat is None:
            lat, lon = la, lo
        elif la.shape != lat.shape or lo.shape != lon.shape or not (np.allclose(la, lat) and np.allclose(lo, lon)):
            sys.exit(f"{f.name}: grid differs from the first file; use one area box for all requests")
        lab, s, c = aggregate_hourly(t, v, end_hour, shift)
        for k, d in enumerate(lab):
            if d in acc:
                acc[d][0] += s[k]
                acc[d][1] += c[k]
            else:
                acc[d] = [s[k].copy(), int(c[k])]
        print(f"  read {f.name}: {len(t)} hours")
    dates = pd.DatetimeIndex(dates)
    D = np.full((len(dates),) + lat.shape + lon.shape, np.nan, dtype="float32")
    pos = {d: i for i, d in enumerate(dates)}
    n_ok = 0
    for d, (s, c) in acc.items():
        if c == 24 and d in pos:
            D[pos[d]] = s
            n_ok += 1
    cov = np.isfinite(D[:, 0, 0])[dates.year >= FIRST_YEAR].mean()
    print(f"  daily ERA5 coverage 1981-2023: {cov:.3f} of days")
    if cov < 0.9:
        warnings.warn("ERA5 covers under 90% of 1981-2023 days; check the downloaded years")
    return D, lat, lon


def station_series(D, lat, lon, slat, slon):
    """Centre-cell and 3x3 area-mean daily series for each station, plus cell indices."""
    ns = len(slat)
    centre = np.full((ns, D.shape[0]), np.nan, dtype="float32")
    nb3 = np.full_like(centre, np.nan)
    ij = np.zeros((ns, 2), int)
    dlat = np.median(np.diff(lat)) if len(lat) > 1 else 0.25
    dlon = np.median(np.diff(lon)) if len(lon) > 1 else 0.25
    for s in range(ns):
        i = int(np.argmin(np.abs(lat - slat[s])))
        j = int(np.argmin(np.abs(lon - slon[s])))
        if abs(lat[i] - slat[s]) > dlat / 2 + 1e-6 or abs(lon[j] - slon[s]) > dlon / 2 + 1e-6:
            sys.exit(f"Station {s} ({slat[s]}, {slon[s]}) lies outside the downloaded area box")
        ij[s] = i, j
        centre[s] = D[:, i, j]
        nb3[s] = D[:, max(i - 1, 0):i + 2, max(j - 1, 0):j + 2].mean((1, 2))
    return centre, nb3, ij


def nepal_cell_mask(lat, lon, how):
    LA, LO = np.meshgrid(lat, lon, indexing="ij")
    if how == "box":
        return (LA >= 26.35) & (LA <= 30.45) & (LO >= 80.06) & (LO <= 88.2)
    import shapefile
    from shapely.geometry import shape
    from shapely.ops import unary_union
    from shapely import contains_xy
    r = shapefile.Reader(str(ROOT / "gis" / "Local Unit" / "local_unit"))
    geom = unary_union([shape(s.__geo_interface__).buffer(0).simplify(0.005) for s in r.shapes()])
    # boundary is on the Everest datum (about 230 m from WGS84), negligible against 0.25 degree cells
    return contains_xy(geom, LO, LA)


# ----------------------------------------------------------------------------- Test A
def run_lengths(miss):
    """Length of the run of True values containing each True element (0 elsewhere)."""
    out = np.zeros(miss.shape, int)
    d = np.diff(np.concatenate([[0], miss.astype(int), [0]]))
    starts, ends = np.where(d == 1)[0], np.where(d == -1)[0]
    for s, e in zip(starts, ends):
        out[s:e] = e - s
    return out


def station_sums(e, st, yi, di, yrs, p95, dropped, kept, wet_mm, win, min_ref):
    """Per-year sums [n, e_target, e_ref, wet_diff, exc_diff] for one station.

    Returns array (2 modes, 4 categories, NY, 5).
    """
    T = len(e)
    valid = np.isfinite(e)
    ee = np.where(valid, e, 0.0)
    wet, exc = (ee >= wet_mm) & valid, (ee > p95) & valid
    obs = (st == 0) & valid

    def grid(a):
        G = np.zeros((NY, 366))
        G[yi, di] = a
        return G

    def wsum(G):
        C = np.concatenate([np.zeros((NY, 1)), np.cumsum(G, 1)], 1)
        d = np.arange(366)
        return C[:, np.minimum(d + win, 365) + 1] - C[:, np.maximum(d - win, 0)]

    W = [wsum(grid(a)) for a in (obs, np.where(obs, ee, 0.0), obs & wet, obs & exc)]
    pool = ((yrs >= FIRST_YEAR) & (yrs < 2018)) | (yrs > 2020)
    refs = {"same_year": W}
    tot = [(w * pool[:, None]).sum(0)[None, :] - w * pool[:, None] for w in W]
    refs["other_years"] = tot

    miss = st == 1
    rl = run_lengths(miss)
    yr = yrs[yi]
    cat = np.zeros(T, int)
    cat[miss & (rl <= 30)] = 1
    cat[miss & (rl > 30) & (rl <= 365)] = 2
    in_out = (yr == 2018) | (yr == 2019)
    if dropped:
        cat[miss & in_out] = 3
    if kept:
        cat[obs & in_out] = 4
    cat[~valid | (yr < FIRST_YEAR)] = 0
    Cg, Eg = np.zeros((NY, 366), int), grid(ee)
    Cg[yi, di] = cat
    Wg, Xg = grid(wet), grid(exc)

    out = np.zeros((2, 4, NY, 5))
    for m, mode in enumerate(MODES):
        n, se, sw, sx = refs[mode]
        ok = n >= min_ref
        with np.errstate(invalid="ignore", divide="ignore"):
            me, mw, mx = se / n, sw / n, sx / n
        for c in range(1, 5):
            if mode == "same_year" and c == 4:
                continue
            sel = (Cg == c) & ok
            out[m, c - 1, :, 0] = sel.sum(1)
            out[m, c - 1, :, 1] = np.where(sel, Eg, 0).sum(1)
            out[m, c - 1, :, 2] = np.where(sel, me, 0).sum(1)
            out[m, c - 1, :, 3] = np.where(sel, Wg - mw, 0).sum(1)
            out[m, c - 1, :, 4] = np.where(sel, Xg - mx, 0).sum(1)
    return out


def three_stats(v):
    n, em, er, wd, xd = v[..., 0], v[..., 1], v[..., 2], v[..., 3], v[..., 4]
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where((n > 0) & (er > 0), em / er, np.nan)
        dw = np.where(n > 0, wd / n, np.nan)
        dx = np.where(n > 0, xd / n, np.nan)
    return ratio, dw, dx


def block_years(rng, nyears, block, B):
    nb = int(np.ceil(nyears / block))
    starts = rng.integers(0, nyears, size=(B, nb))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]) % nyears
    return idx.reshape(B, -1)[:, :nyears]


def ci(x):
    x = x[np.isfinite(x)]
    return (np.nan, np.nan) if len(x) < 20 else tuple(np.percentile(x, [2.5, 97.5]))


def summarise_testA(S, region_of, nboot, block, seed):
    """S: (2, 4, ns, NY, 5). Returns the results table."""
    rng = np.random.default_rng(seed)
    yr_all = np.arange(Y0, Y1 + 1)
    use = yr_all >= FIRST_YEAR
    ny = use.sum()
    yidx = block_years(rng, ny, block, nboot)
    rows = []

    def est_and_ci(Ysel, Tsel, unit):
        """Point estimate from totals; CI by years (Ysel: ny x 5) or stations (Tsel: k x 5)."""
        pt = three_stats(Ysel.sum(0))
        if unit == "year":
            bs = three_stats(Ysel[yidx].sum(1))
        else:
            k = len(Tsel)
            draws = rng.integers(0, k, size=(nboot, k))
            bs = three_stats(Tsel[draws].sum(1))
        return pt, bs

    for reg in REGIONS:
        sel = np.ones(S.shape[2], bool) if reg == "All" else region_of == reg
        for m, mode in enumerate(MODES):
            for c, cat in enumerate(CATS):
                if mode == "same_year" and cat == "control_kept":
                    continue
                Sm = S[m, c][sel]
                Ysel = Sm.sum(0)[use]
                Tsel = Sm[:, use].sum(1)
                unit = "station" if cat in ("outage", "control_kept") else "year"
                pt, bs = est_and_ci(Ysel, Tsel, unit)
                row = dict(region=reg, gap_type=cat, reference=mode, bootstrap_unit=unit,
                           n_days=int(Ysel[:, 0].sum()), n_stations=int((Sm[:, use, 0].sum(1) > 0).sum()))
                for nm, p, b in zip(("ratio_mean_era5", "d_wet_prob", "d_exceed_p95_prob"), pt, bs):
                    row[nm] = p
                    row[nm + "_lo"], row[nm + "_hi"] = ci(b)
                rows.append(row)
        # outage relative to kept stations (difference in differences), other_years reference
        m = 1
        Sout, Sctl = S[m, 2][sel], S[m, 3][sel]
        To, Tc = Sout[:, use].sum(1), Sctl[:, use].sum(1)
        row = dict(region=reg, gap_type="outage_DiD", reference="other_years", bootstrap_unit="station",
                   n_days=int(To[:, 0].sum()), n_stations=int((To[:, 0] > 0).sum()))
        if To[:, 0].sum() > 0 and Tc[:, 0].sum() > 0:
            ro, dwo, dxo = three_stats(To.sum(0))
            rc, dwc, dxc = three_stats(Tc.sum(0))
            io = rng.integers(0, len(To), size=(nboot, len(To)))
            ic = rng.integers(0, len(Tc), size=(nboot, len(Tc)))
            bo, bc = three_stats(To[io].sum(1)), three_stats(Tc[ic].sum(1))
            vals = [(ro / rc, bo[0] / bc[0]), (dwo - dwc, bo[1] - bc[1]), (dxo - dxc, bo[2] - bc[2])]
        else:
            vals = [(np.nan, np.array([np.nan]))] * 3
        for nm, (p, b) in zip(("ratio_mean_era5", "d_wet_prob", "d_exceed_p95_prob"), vals):
            row[nm] = p
            row[nm + "_lo"], row[nm + "_hi"] = ci(b)
        rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- Test B
def annual_indices(arr, dates):
    """arr (k, T) daily mm. Returns dict of (k, nyears) arrays for 1981-2023, NaN for incomplete years."""
    years = np.arange(FIRST_YEAR, Y1 + 1)
    out = {n: np.full((arr.shape[0], len(years)), np.nan) for n in ("annual_total", "Rx1day", "JJAS_total")}
    yr, mo = dates.year.values, dates.month.values
    for j, y in enumerate(years):
        m = yr == y
        a = arr[:, m]
        full = np.isfinite(a).all(1)
        out["annual_total"][full, j] = a[full].sum(1)
        out["Rx1day"][full, j] = a[full].max(1)
        jj = np.isin(mo[m], [6, 7, 8, 9])
        out["JJAS_total"][full, j] = a[full][:, jj].sum(1)
    return years, out


def score_series(years, x):
    """Return z (against 1981-2017), rank (1 = largest), and the 3-year window statistics."""
    base = (years >= BASE[0]) & (years <= BASE[1]) & np.isfinite(x)
    if base.sum() < 10:
        return np.full(len(x), np.nan), np.full(len(x), np.nan)
    z = (x - x[base].mean()) / x[base].std(ddof=1)
    rank = np.full(len(x), np.nan)
    ok = np.isfinite(x)
    rank[ok] = (-x[ok]).argsort().argsort() + 1
    return z, rank


def testB_tables(dates, nep_series, grp_series):
    from scipy import stats
    rows_n, rows_g, rows_s = [], [], []

    def one(name, arr, collector, avg_over_stations):
        years, ind = annual_indices(arr, dates)
        for idx, v in ind.items():
            x = np.nanmean(v, 0) if avg_over_stations else v[0]
            x = np.where(np.isfinite(v).any(0), x, np.nan)
            z, rank = score_series(years, x)
            for j, y in enumerate(years):
                collector.append(dict(series=name, index=idx, year=int(y), value=x[j], z_1981_2017=z[j],
                                      rank_of_n=rank[j], n_years=int(np.isfinite(x).sum()),
                                      in_2018_2020=bool(2018 <= y <= 2020)))
            sel = (years >= 2018) & (years <= 2020)
            zz = z[sel]
            win = np.array([np.nanmean(z[k:k + 3]) for k in range(len(z) - 2)])
            m3 = np.nanmean(zz) if np.isfinite(zz).all() and sel.sum() == 3 else np.nan
            others = z[~sel & np.isfinite(z)]
            mw = stats.mannwhitneyu(zz, others).pvalue if np.isfinite(zz).all() and len(others) > 3 else np.nan
            rows_s.append(dict(series=name, index=idx,
                               z_2018=z[years == 2018][0], z_2019=z[years == 2019][0], z_2020=z[years == 2020][0],
                               rank_2018=rank[years == 2018][0], rank_2019=rank[years == 2019][0],
                               rank_2020=rank[years == 2020][0], n_years=int(np.isfinite(x).sum()),
                               mean_z_2018_2020=m3, mean_z_other_years=np.mean(others) if len(others) else np.nan,
                               share_3yr_windows_as_low=np.nanmean(win <= m3) if np.isfinite(m3) else np.nan,
                               share_3yr_windows_as_high=np.nanmean(win >= m3) if np.isfinite(m3) else np.nan,
                               mannwhitney_p_2018_2020_vs_other=mw))

    one("Nepal_area_mean", nep_series[None, :], rows_n, False)
    for g, arr in grp_series.items():
        one(g, arr, rows_g, True)
    return pd.DataFrame(rows_n), pd.DataFrame(rows_g), pd.DataFrame(rows_s)


# ----------------------------------------------------------------------------- Figure
def make_figure(A, Bn, Bg, path_stem):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    mm = 1 / 25.4
    INK, LIGHT = "#222222", "#D9D9D9"
    plt.rcParams.update({"font.family": ["Arial", "DejaVu Sans"], "font.size": 8, "axes.labelsize": 8,
                         "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
                         "axes.edgecolor": INK, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
                         "ytick.major.width": 0.6, "text.color": INK, "axes.labelcolor": INK,
                         "xtick.color": INK, "ytick.color": INK, "pdf.fonttype": 42})
    fig = plt.figure(figsize=(180 * mm, 128 * mm))
    gs = fig.add_gridspec(2, 6, left=0.09, right=0.985, top=0.92, bottom=0.085, hspace=1.0, wspace=1.9)
    colours = dict(zip(REGIONS, [BLUE, ORANGE, GREEN, PURPLE]))
    gaps = [("short", "same_year", "Short\n(≤30 d)"), ("long", "same_year", "Long\n(31–365 d)"),
            ("outage_DiD", "other_years", "2018–19\noutage")]
    panels = [("ratio_mean_era5", "Ratio of mean ERA5 rain,\nmissing / reference", 1.0),
              ("d_wet_prob", "Difference in wet-day\nprobability", 0.0),
              ("d_exceed_p95_prob", "Difference in probability\nabove ERA5 95th percentile", 0.0)]
    for k, (col, ylab, ref) in enumerate(panels):
        ax = fig.add_subplot(gs[0, 2 * k:2 * k + 2])
        ax.axhline(ref, color=INK, lw=0.6, ls="--")
        for r, reg in enumerate(REGIONS):
            for g, (cat, mode, _) in enumerate(gaps):
                q = A[(A.region == reg) & (A.gap_type == cat) & (A.reference == mode)]
                if q.empty or not np.isfinite(q[col].iloc[0]):
                    continue
                x = g + (r - 1.5) * 0.17
                lo, hi = q[col + "_lo"].iloc[0], q[col + "_hi"].iloc[0]
                v = q[col].iloc[0]
                if np.isfinite(lo):
                    ax.plot([x, x], [lo, hi], color=colours[reg], lw=1.1)
                ax.plot(x, v, "o", color=colours[reg], ms=3.6, mec="white", mew=0.4)
        ax.set_xticks(range(3))
        ax.set_xticklabels([g[2] for g in gaps])
        ax.set_xlim(-0.55, 2.55)
        ax.set_ylabel(ylab)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.grid(axis="y", color=LIGHT, lw=0.5)
        ax.set_axisbelow(True)
        ax.text(-0.30 if k == 0 else -0.33, 1.07, "abc"[k], transform=ax.transAxes, fontsize=10, fontweight="bold")
    fig.legend([Line2D([], [], marker="o", color=colours[r], lw=1.1, ms=4) for r in REGIONS], REGIONS,
               loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.0), handletextpad=0.4,
               columnspacing=1.6)

    def shade(ax):
        ax.axvspan(2017.5, 2020.5, color=LIGHT, alpha=0.7, lw=0)
        ax.axhline(0, color=INK, lw=0.6, ls="--")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    axd = fig.add_subplot(gs[1, 0:3])
    shade(axd)
    for idx, c, lab in (("annual_total", BLUE, "Annual total"), ("Rx1day", ORANGE, "Rx1day"),
                        ("JJAS_total", GREEN, "Jun–Sep total")):
        q = Bn[Bn["index"] == idx].sort_values("year")
        axd.plot(q.year, q.z_1981_2017, "-o", color=c, lw=1.0, ms=2.4, label=lab)
    axd.set_ylabel("Anomaly vs 1981–2017\n(standard deviations)")
    axd.set_xlabel("Year")
    axd.legend(frameon=False, ncol=3, loc="lower left", bbox_to_anchor=(0.0, 1.0), handlelength=1.4,
               columnspacing=0.9, handletextpad=0.4, borderaxespad=0.2)
    axd.text(-0.19, 1.2, "d", transform=axd.transAxes, fontsize=10, fontweight="bold")

    axe = fig.add_subplot(gs[1, 3:6])
    shade(axe)
    for grp, c in (("dropped_2018_19", VERM), ("kept_2018_19", BLUE)):
        for idx, ls in (("annual_total", "-"), ("Rx1day", ":")):
            q = Bg[(Bg.series == grp) & (Bg["index"] == idx)].sort_values("year")
            axe.plot(q.year, q.z_1981_2017, ls, color=c, lw=1.1)
    axe.set_ylabel("Anomaly vs 1981–2017\n(standard deviations)")
    axe.set_xlabel("Year")
    axe.legend([Line2D([], [], color=VERM, lw=1.2), Line2D([], [], color=BLUE, lw=1.2),
                Line2D([], [], color=INK, lw=1.1, ls="-"), Line2D([], [], color=INK, lw=1.1, ls=":")],
               ["Silent 2018–19", "Reporting 2018–19", "Annual total", "Rx1day"],
               frameon=False, ncol=2, loc="lower left", bbox_to_anchor=(0.0, 1.0), handlelength=1.6,
               columnspacing=1.0, handletextpad=0.4, borderaxespad=0.2)
    axe.text(-0.19, 1.2, "e", transform=axe.transAxes, fontsize=10, fontweight="bold")
    fig.savefig(str(path_stem) + ".pdf")
    fig.savefig(str(path_stem) + ".png", dpi=600)
    plt.close(fig)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--era5-dir", default=str(ROOT / "era5"))
    ap.add_argument("--out-dir", default=str(ROOT / "outputs"))
    ap.add_argument("--fig-dir", default=str(ROOT / "figures"))
    ap.add_argument("--day-end-utc", type=int, default=3, help="hour (UTC) at which the ERA5 day ends; DHM 08:45 NPT = 03 UTC")
    ap.add_argument("--label-shift", type=int, default=0, help="days added to the date label of the day")
    ap.add_argument("--cell", choices=["centre", "3x3"], default="centre", help="ERA5 series used in the tests")
    ap.add_argument("--wet-mm", type=float, default=1.0)
    ap.add_argument("--window", type=int, default=15, help="+/- days of day of year")
    ap.add_argument("--min-ref", type=int, default=5, help="minimum observed reference days in the window")
    ap.add_argument("--nboot", type=int, default=2000)
    ap.add_argument("--block", type=int, default=3, help="block length in years for the year bootstrap")
    ap.add_argument("--mask", choices=["polygon", "box"], default="polygon", help="Nepal cell mask for the area mean")
    ap.add_argument("--seed", type=int, default=20240614)
    a = ap.parse_args()
    out, figd = Path(a.out_dir), Path(a.fig_dir)
    out.mkdir(parents=True, exist_ok=True)
    figd.mkdir(parents=True, exist_ok=True)

    d = np.load(ROOT / "outputs" / "daily_qc.npz", allow_pickle=True)
    status, dates = d["status"], pd.DatetimeIndex(d["dates"])
    sid = d["station_id"].astype(str)
    slat, slon = d["lat"].astype(float), d["lon"].astype(float)  # WGS84 assumed
    prov = pd.read_csv(ROOT / "outputs" / "station_province.csv", dtype={"station_id": str}).set_index("station_id")
    province = prov.loc[sid, "province"].to_numpy()
    region_of = np.array(["East" if p in EAST else "Central" if p in CENTRAL else "West" if p in WEST else "?"
                          for p in province])
    if (region_of == "?").any():
        sys.exit("Unknown province names: " + str(set(province[region_of == "?"])))

    print("Building daily ERA5 (day ends %02d UTC, label shift %d)" % (a.day_end_utc, a.label_shift))
    D, lat, lon = build_daily(a.era5_dir, dates, a.day_end_utc, a.label_shift)
    centre, nb3, ij = station_series(D, lat, lon, slat, slon)
    mask = nepal_cell_mask(lat, lon, a.mask)
    if mask.sum() < 20:
        warnings.warn(f"Only {mask.sum()} grid cells inside Nepal for the area mean")
    w = np.cos(np.deg2rad(lat))[:, None] * mask
    with np.errstate(invalid="ignore"):
        nep = (np.nan_to_num(D) * w[None]).sum((1, 2)) / w.sum()
        nep[~np.isfinite(D[:, 0, 0])] = np.nan
    np.savez_compressed(out / "era5_station_daily.npz", dates=dates.to_numpy(), station_id=sid,
                        era5_centre_mm=centre, era5_3x3_mm=nb3, nepal_area_mean_mm=nep.astype("float32"),
                        cell_lat=lat[ij[:, 0]], cell_lon=lon[ij[:, 1]], day_end_utc=a.day_end_utc,
                        label_shift=a.label_shift)

    E = centre if a.cell == "centre" else nb3
    yrs = np.arange(Y0, Y1 + 1)
    yi, di = dates.year.values - Y0, dates.dayofyear.values - 1
    base = (dates.year.values >= BASE[0]) & (dates.year.values <= BASE[1])
    dropped = np.zeros(len(sid), bool)
    kept = np.zeros(len(sid), bool)
    for s in range(len(sid)):  # same definition as 06_null_regional.py
        a17 = (status[s, dates.year == 2017] == 0).mean() > 0.5
        a18 = (status[s, (dates.year >= 2018) & (dates.year <= 2019)] == 0).mean() > 0.5
        dropped[s], kept[s] = a17 and not a18, a17 and a18
    print(f"stations: {len(sid)}, dropped 2018-19: {dropped.sum()}, kept: {kept.sum()}")

    S = np.zeros((2, 4, len(sid), NY, 5))
    for s in range(len(sid)):
        x = E[s, base]
        x = x[np.isfinite(x) & (x >= a.wet_mm)]
        p95 = np.percentile(x, 95) if len(x) > 100 else np.inf
        S[:, :, s] = station_sums(E[s].astype(float), status[s], yi, di, yrs, p95, dropped[s], kept[s],
                                  a.wet_mm, a.window, a.min_ref)
    A = summarise_testA(S, region_of, a.nboot, a.block, a.seed)
    A.to_csv(out / "era5_testA_missingness.csv", index=False)
    print(A[A.region == "All"].round(3).to_string())

    grp = {"dropped_2018_19": E[dropped].astype(float), "kept_2018_19": E[kept].astype(float)}
    Bn, Bg, Bs = testB_tables(dates, nep.astype(float), grp)
    Bn.to_csv(out / "era5_testB_nepal_annual.csv", index=False)
    Bg.to_csv(out / "era5_testB_station_groups.csv", index=False)
    Bs.to_csv(out / "era5_testB_summary.csv", index=False)
    print(Bs.round(3).to_string())

    make_figure(A, Bn, Bg, figd / "fig7_era5_missingness")
    print("done; ERA5 used only as a diagnostic, not as truth for gauge accuracy")


if __name__ == "__main__":
    main()
