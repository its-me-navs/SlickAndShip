"""
Physics-informed drift model.

Combines:
  - Surface current advection (from environment.get_current)
  - Wind-driven drift, standard practice is ~3% of wind speed, deflected
    ~15-25 deg from wind direction due to Ekman effect (we use 20 deg)
  - A small Stokes drift contribution (wave-driven, roughly aligned with wind)

This is the standard oil-spill-trajectory formulation used by operational
tools like GNOME (NOAA) and MOTHY, simplified for the demo. A learned
residual-correction model (e.g. a small MLP trained on real spill
backtrack cases) would slot into `residual_correction()` once you have
labeled data to train it on.
"""

from __future__ import annotations
import numpy as np
import os
if os.getenv("USE_REAL_ENV") == "1":
    from .environment_real import get_current as _default_current, get_wind as _default_wind
else:
    from .environment import get_current as _default_current, get_wind as _default_wind

WIND_DRIFT_FACTOR = 0.03       # fraction of wind speed contributing to drift
WIND_DEFLECTION_DEG = 20.0     # Ekman deflection angle
STOKES_FACTOR = 0.01           # small additional wave-driven term
KM_PER_DEG_LAT = 111.0


def _km_per_deg_lon(lat: float) -> float:
    return 111.0 * np.cos(np.radians(lat))


def residual_correction(lat: float, lon: float, t_hours: float) -> tuple[float, float]:
    """Placeholder for a learned correction term. Returns zero until trained."""
    return 0.0, 0.0


def total_drift_velocity(lat: float, lon: float, t_hours: float,
                          current_fn=_default_current, wind_fn=_default_wind) -> tuple[float, float]:
    """Combined current + wind-drift + Stokes drift velocity, in m/s."""
    cu, cv = current_fn(lat, lon, t_hours)
    wu, wv = wind_fn(lat, lon, t_hours)

    theta = np.radians(WIND_DEFLECTION_DEG)
    wind_drift_u = WIND_DRIFT_FACTOR * (wu * np.cos(theta) - wv * np.sin(theta))
    wind_drift_v = WIND_DRIFT_FACTOR * (wu * np.sin(theta) + wv * np.cos(theta))

    stokes_u = STOKES_FACTOR * wu
    stokes_v = STOKES_FACTOR * wv

    ru, rv = residual_correction(lat, lon, t_hours)

    return cu + wind_drift_u + stokes_u + ru, cv + wind_drift_v + stokes_v + rv


def advect(lat0: float, lon0: float, t0_hours: float, duration_hours: float,
           dt_hours: float = 0.5, direction: str = "forward",
           current_fn=_default_current, wind_fn=_default_wind) -> list[tuple[float, float, float]]:
    """
    RK4 integration of a position through the drift field.
    direction='backward' flips the velocity sign to hindcast the origin.
    current_fn/wind_fn can be swapped for perturbed versions (see
    monte_carlo_backtrack) or, later, real reanalysis-data lookups.
    """
    sign = 1.0 if direction == "forward" else -1.0
    n_steps = max(1, int(abs(duration_hours) / dt_hours))
    lat, lon, t = lat0, lon0, t0_hours
    path = [(lat, lon, t)]

    def vel(lat_, lon_, t_):
        u, v = total_drift_velocity(lat_, lon_, t_, current_fn, wind_fn)
        return sign * u, sign * v

    for _ in range(n_steps):
        dt_sec = dt_hours * 3600
        k1u, k1v = vel(lat, lon, t)

        lat2 = lat + k1v * dt_sec / 2 / (KM_PER_DEG_LAT * 1000)
        lon2 = lon + k1u * dt_sec / 2 / (_km_per_deg_lon(lat) * 1000)
        k2u, k2v = vel(lat2, lon2, t + dt_hours / 2)

        lat3 = lat + k2v * dt_sec / 2 / (KM_PER_DEG_LAT * 1000)
        lon3 = lon + k2u * dt_sec / 2 / (_km_per_deg_lon(lat) * 1000)
        k3u, k3v = vel(lat3, lon3, t + dt_hours / 2)

        lat4 = lat + k3v * dt_sec / (KM_PER_DEG_LAT * 1000)
        lon4 = lon + k3u * dt_sec / (_km_per_deg_lon(lat) * 1000)
        k4u, k4v = vel(lat4, lon4, t + dt_hours)

        u_avg = (k1u + 2 * k2u + 2 * k3u + k4u) / 6
        v_avg = (k1v + 2 * k2v + 2 * k3v + k4v) / 6

        lon += (u_avg * dt_sec) / (_km_per_deg_lon(lat) * 1000)
        lat += (v_avg * dt_sec) / (KM_PER_DEG_LAT * 1000)
        t += sign * dt_hours

        path.append((lat, lon, t))

    return path


def monte_carlo_backtrack(spill_lat: float, spill_lon: float, spill_time_hours: float,
                           hindcast_hours: float, n_ensemble: int = 60,
                           current_noise_std: float = 0.05, wind_noise_std: float = 1.0,
                           seed: int = 42) -> list[tuple[float, float, float]]:
    """
    Monte Carlo ensemble backtrack: perturbs the environmental fields per
    realization to produce a spread of plausible origin points/times rather
    than a single (overconfident) pin on the map.
    Returns a list of (lat, lon, t_hours) origin candidates.
    """
    rng = np.random.default_rng(seed)
    origins = []

    for _ in range(n_ensemble):
        cu_n, cv_n = rng.normal(0, current_noise_std, size=2)
        wu_n, wv_n = rng.normal(0, wind_noise_std, size=2)

        def noisy_current(lat, lon, t, _n=(cu_n, cv_n)):
            u, v = _default_current(lat, lon, t)
            return u + _n[0], v + _n[1]

        def noisy_wind(lat, lon, t, _n=(wu_n, wv_n)):
            u, v = _default_wind(lat, lon, t)
            return u + _n[0], v + _n[1]

        path = advect(spill_lat, spill_lon, spill_time_hours, hindcast_hours,
                       dt_hours=0.5, direction="backward",
                       current_fn=noisy_current, wind_fn=noisy_wind)
        origins.append(path[-1])

    return origins


def forward_forecast(lat: float, lon: float, t_hours: float, forecast_hours: float,
                      dt_hours: float = 0.5) -> list[tuple[float, float, float]]:
    """Forward drift path for the visualization / response-planning side."""
    return advect(lat, lon, t_hours, forecast_hours, dt_hours=dt_hours, direction="forward")
