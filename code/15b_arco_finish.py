# Code Developer: Somnath Luitel (somnath.luitel416@topper.wku.edu)
"""Finish steps for the ARCO-ERA5 extraction (code/15_arco_extract.py). Read-only on era5/ and era5_arco/*.nc.

    python -W ignore code/15b_arco_finish.py verify     # all years: hours, NaN, magnitudes, units
    python -W ignore code/15b_arco_finish.py zarr       # era5_arco/nepal_tp.zarr from the yearly files
    python -W ignore code/15b_arco_finish.py compare    # era5/era5_tp_1980.nc (CDS) vs ARCO, if the CDS file exists
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
ARCO = ROOT / "era5_arco"
CDS = ROOT / "era5"
CELLS = {"Kathmandu": (27.75, 85.25), "Pokhara": (28.25, 84.0), "Biratnagar": (26.5, 87.25),
         "Nepalgunj": (28.0, 81.5), "Jumla": (29.25, 82.25), "Lumbini": (27.5, 83.25)}


def years():
    return sorted(int(p.stem.split("_")[-1]) for p in ARCO.glob("era5_tp_*.nc") if "tmp" not in p.name)


def verify():
    bad = 0
    for y in years():
        ds = xr.open_dataset(ARCO / f"era5_tp_{y}.nc")
        da = ds["total_precipitation"]
        t = pd.DatetimeIndex(ds["valid_time"].values)
        extra_start = pd.Timedelta(hours=1) if y == 1980 else pd.Timedelta(0)
        extra_end = pd.Timedelta(hours=4) if y == 2023 else pd.Timedelta(0)
        exp = pd.date_range(pd.Timestamp(f"{y}-01-01") - extra_start, pd.Timestamp(f"{y}-12-31 23:00") + extra_end, freq="h")
        v = da.values
        problems = []
        if not t.equals(exp):
            problems.append("time axis")
        if da.attrs.get("units") != "m":
            problems.append("units")
        if np.isnan(v).any():
            problems.append("NaN")
        if v.max() >= 0.2:
            problems.append("max>=0.2")
        if v.min() < -1e-6:
            problems.append(f"min {v.min():.2e}")
        print(f"{y}: {len(t)} h, max {v.max():.4f}, domain-mean annual {float(v.sum(0).mean()):.2f} m "
              f"{'OK' if not problems else 'PROBLEM ' + ','.join(problems)}")
        bad += bool(problems)
        ds.close()
    ys = years()
    missing = sorted(set(range(1980, 2024)) - set(ys))
    print(f"{len(ys)} files; missing years: {missing or 'none'}; problems: {bad}")


def zarr_copy():
    ds = xr.open_mfdataset([ARCO / f"era5_tp_{y}.nc" for y in years()], combine="by_coords")
    ds = ds.sel(valid_time=~ds.get_index("valid_time").duplicated())
    out = ARCO / "nepal_tp.zarr"
    ds.chunk({"valid_time": 8784, "latitude": 21, "longitude": 37}).to_zarr(out, mode="w")
    print("wrote", out, dict(ds.sizes))


def read_cds(path):
    ds = xr.open_dataset(path)
    name = "tp" if "tp" in ds else list(ds.data_vars)[0]
    da = ds[name]
    if "expver" in da.dims:
        da = da.isel(expver=0).combine_first(da.isel(expver=1))
    tdim = "valid_time" if "valid_time" in da.dims else "time"
    return da.reset_coords(drop=True).rename({tdim: "valid_time"}), da.attrs


def compare():
    f = CDS / "era5_tp_1980.nc"
    if not f.exists():
        sys.exit(f"{f} not there yet; nothing to compare")
    a = xr.open_dataset(ARCO / "era5_tp_1980.nc")["total_precipitation"]
    c, attrs = read_cds(f)
    print("CDS units", attrs.get("units"), "dims", c.dims, "grid", c.sizes["latitude"], "x", c.sizes["longitude"])
    print("ARCO lat", float(a.latitude[0]), float(a.latitude[-1]), "lon", float(a.longitude[0]), float(a.longitude[-1]))
    print("CDS  lat", float(c.latitude[0]), float(c.latitude[-1]), "lon", float(c.longitude[0]), float(c.longitude[-1]))
    t = a.get_index("valid_time").intersection(c.get_index("valid_time"))
    print("overlapping hours:", len(t), "of ARCO", a.sizes["valid_time"], "CDS", c.sizes["valid_time"])
    rows = []
    for k, (la, lo) in CELLS.items():
        x = a.sel(valid_time=t).sel(latitude=la, longitude=lo, method="nearest")
        y = c.sel(valid_time=t).sel(latitude=la, longitude=lo, method="nearest")
        xv, yv = x.values.astype(float), y.values.astype(float)
        # hourly, plus +-1 h shifts to detect a timestamp offset
        r = {s: np.corrcoef(xv[max(s, 0):len(xv) + min(s, 0)], yv[max(-s, 0):len(yv) + min(-s, 0)])[0, 1] for s in (-1, 0, 1)}
        rows.append(dict(cell=k, lat=float(x.latitude), lon=float(x.longitude), sum_arco_m=xv.sum(), sum_cds_m=yv.sum(),
                         ratio=xv.sum() / yv.sum() if yv.sum() else np.nan, r_hourly=r[0], r_lag_m1=r[-1], r_lag_p1=r[1],
                         max_abs_diff=np.abs(xv - yv).max()))
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    print("A grid difference shows as differing lat/lon above or nonzero max_abs_diff; "
          "a timestamp offset shows as r_lag_m1 or r_lag_p1 exceeding r_hourly.")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    {"verify": verify, "zarr": zarr_copy, "compare": compare}.get(cmd, lambda: sys.exit(__doc__))()
    sys.stdout.flush()
    os._exit(0)
