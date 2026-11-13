# Code Developer: Somnath Luitel (somnath.luitel416@topper.wku.edu)
"""Extract hourly ERA5 total precipitation for Nepal from Google's public ARCO-ERA5 Zarr store.

Parallel route to the Copernicus CDS download (code/14a_era5_download.py). Anonymous access.
Store: gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3, variable total_precipitation
(metres of water, accumulation over the hour ending at the stamp; verified against
mean_total_precipitation_rate * 3600 at the same stamp, ratio 1.0000).

Writes era5_arco/era5_tp_YYYY.nc (float32, zlib, dims valid_time/latitude/longitude, units 'm'),
readable by code/14_era5_checks.py --era5-dir era5_arco. Resumable: finished years are skipped.
The 1980 file also holds 1979-12-31 23:00; the last year also holds the first four hours of the next year.

    python -W ignore code/15_arco_extract.py [--years 1980 1981] [--threads 16]
"""
import argparse
import os
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
STORE = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
BOX = dict(latitude=slice(31, 26), longitude=slice(79.5, 88.5))  # lat is descending, lon is 0-360
Y0, Y1 = 1980, 2023
KATHMANDU = (27.75, 85.25)


def log(fh, msg):
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    fh.write(line + "\n")
    fh.flush()


def fetch(sub, i, tries=6):
    for k in range(tries):
        try:
            return sub.isel(time=i).values
        except Exception:
            if k == tries - 1:
                raise
            time.sleep(2 ** k)


def verify(path, times):
    ds = xr.open_dataset(path)
    da = ds["total_precipitation"]
    assert da.attrs.get("units") == "m", "units attribute"
    assert da.dims == ("valid_time", "latitude", "longitude"), da.dims
    assert da.dtype == np.float32
    assert len(ds["valid_time"]) == len(times), (len(ds["valid_time"]), len(times))
    v = da.values
    assert not np.isnan(v).any(), "NaN present"
    assert v.max() < 0.2, f"hourly max {v.max()}"
    cell = da.sel(latitude=KATHMANDU[0], longitude=KATHMANDU[1], method="nearest")
    ann = float(cell.where(cell["valid_time"].dt.year == times[len(times) // 2].year, drop=True).sum())
    ds.close()
    return float(v.max()), ann


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, nargs="*", default=list(range(Y0, Y1 + 1)))
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--out", default=str(ROOT / "era5_arco"))
    ap.add_argument("--log", default=None)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    fh = open(a.log or (out / "extract.log"), "a")
    ds = xr.open_zarr(STORE, chunks=None, storage_options=dict(token="anon"))
    tidx = pd.DatetimeIndex(ds["time"].values)
    sub = ds["total_precipitation"].sel(**BOX)
    log(fh, f"opened store; box {sub.sizes['latitude']}x{sub.sizes['longitude']} cells; "
            f"lat {float(sub.latitude[0])}..{float(sub.latitude[-1])} lon {float(sub.longitude[0])}..{float(sub.longitude[-1])}")
    t_all = time.time()
    for y in a.years:
        path = out / f"era5_tp_{y}.nc"
        start = pd.Timestamp(f"{y}-01-01 00:00") - (pd.Timedelta(hours=1) if y == Y0 else pd.Timedelta(0))
        end = pd.Timestamp(f"{y}-12-31 23:00") + (pd.Timedelta(hours=4) if y == Y1 else pd.Timedelta(0))
        times = pd.date_range(start, end, freq="h")
        if path.exists():
            try:
                verify(path, times)
                log(fh, f"{y}: exists and verified, skipping")
                continue
            except Exception as e:
                log(fh, f"{y}: existing file failed verification ({e}); redoing")
        i0 = tidx.get_loc(start)
        idx = list(range(i0, i0 + len(times)))
        assert tidx[idx[-1]] == end
        t0 = time.time()
        with ThreadPoolExecutor(a.threads) as ex:
            arrs = list(ex.map(lambda i: fetch(sub, i), idx))
        data = np.stack(arrs).astype("float32")
        da = xr.DataArray(
            data, dims=("valid_time", "latitude", "longitude"),
            coords=dict(valid_time=times, latitude=sub["latitude"].values, longitude=sub["longitude"].values),
            name="total_precipitation",
            attrs=dict(units="m", long_name="Total precipitation",
                       comment="ERA5 hourly accumulation over the hour ending at valid_time (UTC); "
                               "source ARCO-ERA5 ar/full_37-1h-0p25deg-chunk-1.zarr-v3"))
        tmp = path.with_suffix(".tmp.nc")
        da.to_dataset().to_netcdf(tmp, encoding={"total_precipitation": dict(zlib=True, complevel=4, dtype="float32")})
        os.replace(tmp, path)
        mx, ann = verify(path, times)
        dt = time.time() - t0
        log(fh, f"{y}: {len(times)} h in {dt:.0f}s ({dt / len(times):.3f} s/h); max hourly {mx:.4f} m; "
                f"Kathmandu-cell annual {ann:.2f} m; {path.stat().st_size / 1e6:.1f} MB")
    log(fh, f"done in {(time.time() - t_all) / 3600:.2f} h")
    fh.close()
    sys.stdout.flush()
    os._exit(0)  # avoid the gcsfs/aiohttp shutdown hang on Windows


if __name__ == "__main__":
    main()
