"""
One-time fetch script: pulls real ERA5 wind + CMEMS ocean current data for
the pipeline's operational region/time window and caches them locally as
NetCDF. Run this ONCE (locally, with your own API credentials) before a
demo — the rest of the pipeline reads from the cached files, never hits
the network live.

SETUP (do this once):
  1. ERA5 (wind):
     - Register at https://cds.climate.copernicus.eu
     - pip install cdsapi
     - Create ~/.cdsapirc with your UID:API key (instructions on the CDS
       site under your profile page)

  2. CMEMS (currents):
     - Register at https://marine.copernicus.eu
     - pip install copernicusmarine
     - Run `copernicusmarine login` once (stores credentials locally)

Then just run:  python fetch_real_env_data.py

Output:
  data/real_env/era5_wind.nc     (variables: u10, v10)
  data/real_env/cmems_currents.nc (variables: uo, vo)
"""

from __future__ import annotations
import os
from datetime import datetime, timedelta, timezone

# ── MUST MATCH environment_real.py's ANCHOR_UTC exactly ─────────────────
ANCHOR_UTC = datetime(2026, 8, 25, 0, 0, tzinfo=timezone.utc)

# Same region as pipeline.py's REGION_BOUNDS
REGION_BOUNDS = {"lat_min": 14.9, "lat_max": 15.7, "lon_min": 72.9, "lon_max": 73.9}

# Covers pipeline.py's T_START(70) through forecast end (~T_END+forecast=122h),
# with a few hours of buffer on each side.
FETCH_START_H = 60.0
FETCH_END_H = 130.0

OUT_DIR = os.path.join(os.path.dirname(__file__), "data", "real_env")
os.makedirs(OUT_DIR, exist_ok=True)


def _hour_range():
    start = ANCHOR_UTC + timedelta(hours=FETCH_START_H)
    end = ANCHOR_UTC + timedelta(hours=FETCH_END_H)
    return start, end


def fetch_era5_wind():
    import cdsapi

    start, end = _hour_range()
    print(f"Fetching ERA5 wind: {start} -> {end} UTC")

    # CDS 'area' format is [North, West, South, East]
    area = [REGION_BOUNDS["lat_max"], REGION_BOUNDS["lon_min"],
            REGION_BOUNDS["lat_min"], REGION_BOUNDS["lon_max"]]

    # ERA5 is published in whole UTC days; request full days spanning the window.
    days_needed = sorted({(start + timedelta(hours=h)).strftime("%Y-%m-%d")
                           for h in range(0, int((end - start).total_seconds() // 3600) + 1)})
    years = sorted({d[:4] for d in days_needed})
    months = sorted({d[5:7] for d in days_needed})
    day_nums = sorted({d[8:10] for d in days_needed})

    c = cdsapi.Client()
    out_path = os.path.join(OUT_DIR, "era5_wind.nc")
    c.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": "reanalysis",
            "variable": ["10m_u_component_of_wind", "10m_v_component_of_wind"],
            "year": years,
            "month": months,
            "day": day_nums,
            "time": [f"{h:02d}:00" for h in range(24)],
            "area": area,
            "format": "netcdf",
        },
        out_path,
    )
    print(f"Saved -> {out_path}")


def fetch_cmems_currents():
    import copernicusmarine

    start, end = _hour_range()
    print(f"Fetching CMEMS currents: {start} -> {end} UTC")

    out_path = os.path.join(OUT_DIR, "cmems_currents.nc")
    copernicusmarine.subset(
        dataset_id="cmems_mod_glo_phy_anfc_0.083deg_PT1H-m",
        variables=["uo", "vo"],
        minimum_longitude=REGION_BOUNDS["lon_min"],
        maximum_longitude=REGION_BOUNDS["lon_max"],
        minimum_latitude=REGION_BOUNDS["lat_min"],
        maximum_latitude=REGION_BOUNDS["lat_max"],
        minimum_depth=0,
        maximum_depth=1,  # surface only
        start_datetime=start.strftime("%Y-%m-%dT%H:%M:%S"),
        end_datetime=end.strftime("%Y-%m-%dT%H:%M:%S"),
        output_filename=os.path.basename(out_path),
        output_directory=os.path.dirname(out_path),
    )
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    fetch_era5_wind()
    fetch_cmems_currents()
    print("\nDone. Point drift/environment_real.py at these files (already configured "
          "if paths weren't changed). Set USE_REAL_ENV=1 to enable in the pipeline.")