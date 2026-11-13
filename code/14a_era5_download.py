# Code Developer: Somnath Luitel (somnath.luitel416@topper.wku.edu)
"""Download ERA5 hourly total precipitation for Nepal (run once before step 14).

Needs a free Copernicus account, the dataset licence accepted once on the dataset page
(https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels, download tab,
bottom of the form), and `pip install cdsapi`. The client reads your own credentials from
your own ~/.cdsapirc (on Windows C:\\Users\\<you>\\.cdsapirc). This script contains and asks
for no key.

One file per year, 1980-2023, saved as era5/era5_tp_YYYY.nc.
1980 is needed only because the day ending 03 UTC on 1 January 1981 starts on 31 December 1980.
Existing files are skipped, so the script can be restarted after an interruption.

Usage:  python code/14a_era5_download.py            (all years)
        python code/14a_era5_download.py 1990 1995   (first and last year)

ERA5 is a model product used here only for two diagnostic questions (see 14_era5_checks.py).
"""
import sys
import time
import zipfile
from pathlib import Path

import cdsapi

TARGET = Path(__file__).resolve().parents[1] / "era5"
DATASET = "reanalysis-era5-single-levels"
AREA = [31.0, 79.5, 26.0, 88.5]  # North, West, South, East (degrees); Nepal is 26.35-30.45 N, 80.06-88.20 E
FIRST, LAST = 1980, 2023
MAX_TRIES = 12


def request_for(year):
    return {
        "product_type": "reanalysis",
        "variable": ["total_precipitation"],
        "year": [str(year)],
        "month": [f"{m:02d}" for m in range(1, 13)],
        "day": [f"{d:02d}" for d in range(1, 32)],
        "time": [f"{h:02d}:00" for h in range(24)],
        "area": AREA,
        "data_format": "netcdf",
        "download_format": "unarchived",
    }


def fetch(client, year):
    out = TARGET / f"era5_tp_{year}.nc"
    if out.exists() and out.stat().st_size > 1_000_000:
        print(f"{year}: already present, skipped")
        return
    tmp = TARGET / f"era5_tp_{year}.part"
    for attempt in range(1, MAX_TRIES + 1):
        try:
            client.retrieve(DATASET, request_for(year), str(tmp))
            break
        except Exception as exc:  # network or queue failure: wait and try again
            print(f"{year}: attempt {attempt} failed: {exc}")
            if attempt == MAX_TRIES:
                raise
            # the CDS limits queued requests per dataset; wait longer instead of resubmitting
            time.sleep(300 if "queued requests" in str(exc) else 60)
    if zipfile.is_zipfile(tmp):  # CDS sometimes returns a zip; keep the NetCDF inside
        with zipfile.ZipFile(tmp) as z:
            members = [n for n in z.namelist() if n.endswith(".nc")]
            if len(members) != 1:
                raise RuntimeError(f"{year}: zip holds {z.namelist()}, expected one .nc file")
            out.write_bytes(z.read(members[0]))
        tmp.unlink()
    else:
        tmp.replace(out)
    print(f"{year}: saved {out.name} ({out.stat().st_size / 1e6:.1f} MB)")


def main():
    first, last = FIRST, LAST
    if len(sys.argv) == 3:
        first, last = int(sys.argv[1]), int(sys.argv[2])
    TARGET.mkdir(parents=True, exist_ok=True)
    client = cdsapi.Client()  # credentials come from the user's own ~/.cdsapirc
    for year in range(first, last + 1):
        fetch(client, year)


if __name__ == "__main__":
    main()
