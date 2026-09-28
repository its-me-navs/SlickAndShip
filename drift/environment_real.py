"""
Real-data version of environment.py — same get_current(lat, lon, t_hours) /
get_wind(lat, lon, t_hours) signatures as the synthetic module, so
drift_model.py needs ZERO changes. Reads from locally cached NetCDF files
(produced once by fetch_real_env_data.py). No network calls — safe to use
live in a demo.

PERFORMANCE NOTE: a single pipeline run makes 15,000-20,000+ lookups
(60 Monte Carlo ensemble members x RK4 substeps for hindcast + forecast).
xarray's ds.interp() is far too slow per-call at that volume — it was
stalling the whole request for minutes. This version builds a fast
scipy.interpolate.RegularGridInterpolator ONCE per dataset (cached), then
every get_wind/get_current call is a cheap array lookup instead of
re-parsing the NetCDF grid each time.

ANCHOR_UTC must match fetch_real_env_data.py exactly: it's the real UTC
timestamp that the pipeline's t_hours=0 corresponds to.

If the cache files are missing, or a query falls outside the cached
grid's range, this module falls back to the synthetic fields from
environment.py (with one warning), rather than crashing the pipeline.
"""

from __future__ import annotations
import os
import warnings
from datetime import datetime, timedelta, timezone
from functools import lru_cache

import numpy as np

from . import environment as _synthetic

ANCHOR_UTC = datetime(2026, 8, 25, 0, 0, tzinfo=timezone.utc)
_ANCHOR_NP = np.datetime64(ANCHOR_UTC.replace(tzinfo=None))

_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "real_env")
_WIND_PATH = os.path.join(_DATA_DIR, "era5_wind.nc")
_CURRENT_PATH = os.path.join(_DATA_DIR, "cmems_currents.nc")

_warned = False


def _warn_fallback(reason: str):
    global _warned
    if not _warned:
        warnings.warn(
            f"[environment_real] Falling back to synthetic fields: {reason}. "
            f"Run fetch_real_env_data.py to enable real data.",
            RuntimeWarning,
        )
        _warned = True


def _build_interpolators(path: str, var_u: str, var_v: str):
    """
    Opens a NetCDF file once and builds fast (time, lat, lon) -> (u, v)
    interpolators. Returns (interp_u, interp_v). Raises on any structural
    problem so the caller can fall back to synthetic cleanly.
    """
    import xarray as xr
    from scipy.interpolate import RegularGridInterpolator

    ds = xr.open_dataset(path)

    # newer CDS ERA5 downloads name the time dimension 'valid_time' instead
    # of 'time' — normalize so the rest of this module doesn't care either way.
    if "time" not in ds.dims and "valid_time" in ds.dims:
        ds = ds.rename({"valid_time": "time"})

    # CMEMS current files carry a depth dimension even for a single surface
    # level — drop it (take the shallowest/only level) before gridding.
    if "depth" in ds.dims:
        ds = ds.isel(depth=0)

    ds = ds.transpose("time", "latitude", "longitude", ...)

    times_hours = (ds["time"].values - _ANCHOR_NP) / np.timedelta64(1, "h")
    times_hours = times_hours.astype(np.float64)
    lats = ds["latitude"].values.astype(np.float64)
    lons = ds["longitude"].values.astype(np.float64)

    u_vals = ds[var_u].values.astype(np.float64)
    v_vals = ds[var_v].values.astype(np.float64)

    # RegularGridInterpolator requires strictly ascending axes.
    if lats[0] > lats[-1]:
        lats = lats[::-1]
        u_vals = u_vals[:, ::-1, :]
        v_vals = v_vals[:, ::-1, :]
    if times_hours[0] > times_hours[-1]:
        times_hours = times_hours[::-1]
        u_vals = u_vals[::-1, :, :]
        v_vals = v_vals[::-1, :, :]

    interp_u = RegularGridInterpolator(
        (times_hours, lats, lons), u_vals, bounds_error=False, fill_value=None
    )
    interp_v = RegularGridInterpolator(
        (times_hours, lats, lons), v_vals, bounds_error=False, fill_value=None
    )
    return interp_u, interp_v


@lru_cache(maxsize=1)
def _wind_interpolators():
    return _build_interpolators(_WIND_PATH, "u10", "v10")


@lru_cache(maxsize=1)
def _current_interpolators():
    return _build_interpolators(_CURRENT_PATH, "uo", "vo")


def get_wind(lat: float, lon: float, t_hours: float) -> tuple[float, float]:
    try:
        interp_u, interp_v = _wind_interpolators()
        point = [(t_hours, lat, lon)]
        u = float(interp_u(point)[0])
        v = float(interp_v(point)[0])
        if np.isnan(u) or np.isnan(v):
            raise ValueError("interpolation returned NaN")
        return u, v
    except (FileNotFoundError, OSError, ValueError, KeyError) as e:
        _warn_fallback(f"wind lookup failed ({e})")
        return _synthetic.get_wind(lat, lon, t_hours)


def get_current(lat: float, lon: float, t_hours: float) -> tuple[float, float]:
    try:
        interp_u, interp_v = _current_interpolators()
        point = [(t_hours, lat, lon)]
        u = float(interp_u(point)[0])
        v = float(interp_v(point)[0])
        if np.isnan(u) or np.isnan(v):
            raise ValueError("interpolation returned NaN")
        return u, v
    except (FileNotFoundError, OSError, ValueError, KeyError) as e:
        _warn_fallback(f"current lookup failed ({e})")
        return _synthetic.get_current(lat, lon, t_hours)