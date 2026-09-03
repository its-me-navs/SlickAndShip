"""
Synthetic ocean current + wind field.

In production you'd swap these two functions for real reanalysis data
(e.g. OSCAR/HYCOM for currents, ERA5 for wind), queried by lat/lon/time.
For the hackathon demo we generate smooth, physically-plausible synthetic
fields so the drift model has something realistic to integrate over.
"""

import numpy as np


def get_current(lat: float, lon: float, t_hours: float) -> tuple[float, float]:
    """
    Synthetic surface current vector (u, v) in m/s at a given position/time.
    Modeled as a slowly-rotating mesoscale gyre plus a small tidal oscillation,
    which is a reasonable stand-in for the kind of coastal current structure
    you'd pull from OSCAR/HYCOM.
    """
    # Gyre center (arbitrary reference point for the demo region)
    lat0, lon0 = 15.0, 73.0
    dx = (lon - lon0) * 111.0 * np.cos(np.radians(lat0))  # km
    dy = (lat - lat0) * 111.0  # km
    r = np.sqrt(dx**2 + dy**2) + 1e-6

    # Rotational (gyre) component, decaying with distance from center
    gyre_strength = 0.25  # m/s at core
    omega = gyre_strength / (1 + (r / 40) ** 2)
    u_gyre = -omega * (dy / r)
    v_gyre = omega * (dx / r)

    # Semi-diurnal tidal oscillation (~12.4h period)
    tidal_amp = 0.08
    phase = 2 * np.pi * t_hours / 12.4
    u_tide = tidal_amp * np.cos(phase)
    v_tide = tidal_amp * np.sin(phase)

    return u_gyre + u_tide, v_gyre + v_tide


def get_wind(lat: float, lon: float, t_hours: float) -> tuple[float, float]:
    """
    Synthetic 10m wind vector (u, v) in m/s. Modeled as a steady monsoon-like
    prevailing wind with a slow diurnal wobble.
    """
    base_u, base_v = 4.0, -6.0  # steady prevailing wind, m/s
    wobble = 1.5
    phase = 2 * np.pi * t_hours / 24.0
    u = base_u + wobble * np.sin(phase)
    v = base_v + wobble * np.cos(phase)
    return u, v
