"""
Synthetic AIS traffic generator.

Produces a background of "innocent" vessel tracks moving through the region,
plus one "guilty" vessel whose path passes near the spill origin at the
estimated origin time and which goes AIS-dark for a window around that
time (mirroring real-world spoofing/shutoff behavior).

The AIS DataFrame schema mirrors real AIS message fields (MMSI, timestamp,
lat/lon, SOG, COG, vessel type) so this can be swapped for a real
MarineCadastre extract with no changes downstream.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

VESSEL_TYPES = ["Tanker", "Cargo", "Fishing", "Tug", "Passenger"]
KM_PER_DEG_LAT = 111.0


def _km_per_deg_lon(lat: float) -> float:
    return 111.0 * np.cos(np.radians(lat))


def _straight_line_track(mmsi: int, vessel_type: str, lat0: float, lon0: float,
                          course_deg: float, speed_knots: float,
                          t_start_hours: float, t_end_hours: float,
                          ping_interval_hours: float, rng: np.random.Generator) -> pd.DataFrame:
    """Generate a near-straight-line AIS track with minor speed/course jitter."""
    speed_kmh = speed_knots * 1.852
    rows = []
    lat, lon = lat0, lon0
    t = t_start_hours
    while t <= t_end_hours:
        jitter_course = course_deg + rng.normal(0, 3)
        jitter_speed = max(0.5, speed_knots + rng.normal(0, 0.5))
        rows.append({
            "mmsi": mmsi, "vessel_type": vessel_type, "timestamp_h": t,
            "lat": lat, "lon": lon, "sog_knots": jitter_speed, "cog_deg": jitter_course % 360,
        })
        dt = ping_interval_hours
        dist_km = speed_kmh * dt
        lat += (dist_km * np.cos(np.radians(course_deg))) / KM_PER_DEG_LAT
        lon += (dist_km * np.sin(np.radians(course_deg))) / _km_per_deg_lon(lat)
        t += dt
    return pd.DataFrame(rows)


def generate_background_traffic(region_bounds: dict, t_start_hours: float, t_end_hours: float,
                                  n_vessels: int = 15, seed: int = 7) -> pd.DataFrame:
    """
    region_bounds: {'lat_min','lat_max','lon_min','lon_max'}
    Generates n_vessels innocent tracks with random start points, courses,
    speeds and *start times staggered across the window* (real AIS traffic
    isn't all present from t_start — vessels transit in and out on their own
    schedules), each pinging AIS every 6-12 minutes (realistic AIS interval).
    Each vessel's track is centered on a random point in time so a
    reasonable population is actually present near any given moment in the
    window, not just clustered at the start.
    """
    rng = np.random.default_rng(seed)
    tracks = []
    window = t_end_hours - t_start_hours
    for i in range(n_vessels):
        mmsi = 200_000_000 + i
        vessel_type = rng.choice(VESSEL_TYPES)
        # anchor point + time: where/when this vessel is somewhere in the region
        anchor_lat = rng.uniform(region_bounds["lat_min"], region_bounds["lat_max"])
        anchor_lon = rng.uniform(region_bounds["lon_min"], region_bounds["lon_max"])
        anchor_t = rng.uniform(t_start_hours, t_end_hours)
        course = rng.uniform(0, 360)
        speed = rng.uniform(6, 18)  # knots
        speed_kmh = speed * 1.852
        ping_interval = rng.uniform(0.1, 0.2)  # ~6-12 min

        # backfill from t_start to anchor, forwardfill anchor to t_end,
        # both passing through (anchor_lat, anchor_lon) at anchor_t
        back_hours = anchor_t - t_start_hours
        lat0 = anchor_lat - (speed_kmh * back_hours * np.cos(np.radians(course))) / KM_PER_DEG_LAT
        lon0 = anchor_lon - (speed_kmh * back_hours * np.sin(np.radians(course))) / _km_per_deg_lon(anchor_lat)

        pre = _straight_line_track(mmsi, vessel_type, lat0, lon0, course, speed,
                                    t_start_hours, anchor_t, ping_interval, rng)
        post = _straight_line_track(mmsi, vessel_type, anchor_lat, anchor_lon, course, speed,
                                     anchor_t, t_end_hours, ping_interval, rng)
        track = pd.concat([pre, post], ignore_index=True).drop_duplicates(subset="timestamp_h")
        tracks.append(track)
    return pd.concat(tracks, ignore_index=True)


def inject_guilty_vessel(origin_lat: float, origin_lon: float, origin_time_hours: float,
                          t_start_hours: float, t_end_hours: float,
                          gap_before_hours: float = 1.5, gap_after_hours: float = 2.0,
                          mmsi: int = 999_999_999, vessel_type: str = "Tanker",
                          seed: int = 99) -> pd.DataFrame:
    """
    Builds a vessel track that passes directly through the spill origin at
    the origin time, then removes AIS pings for a window around that time
    to simulate a transponder shutoff/spoof. Ground-truth mmsi is returned
    so you can check the scoring engine actually recovers it, but nothing
    downstream is told which vessel this is.
    """
    rng = np.random.default_rng(seed)
    course = rng.uniform(0, 360)
    speed = rng.uniform(10, 14)

    # backfill track from t_start to origin_time, forwardfill origin_time to t_end,
    # both centered on passing through (origin_lat, origin_lon) at origin_time
    speed_kmh = speed * 1.852
    ping_interval = 0.15

    pre = _straight_line_track(mmsi, vessel_type,
                                origin_lat - (speed_kmh * (origin_time_hours - t_start_hours) *
                                              np.cos(np.radians(course))) / KM_PER_DEG_LAT,
                                origin_lon - (speed_kmh * (origin_time_hours - t_start_hours) *
                                              np.sin(np.radians(course))) / _km_per_deg_lon(origin_lat),
                                course, speed, t_start_hours, origin_time_hours, ping_interval, rng)
    post = _straight_line_track(mmsi, vessel_type, origin_lat, origin_lon, course, speed,
                                 origin_time_hours, t_end_hours, ping_interval, rng)
    full_track = pd.concat([pre, post], ignore_index=True).drop_duplicates(subset="timestamp_h")

    # remove pings inside the gap window to simulate AIS shutoff around the spill
    gap_start = origin_time_hours - gap_before_hours
    gap_end = origin_time_hours + gap_after_hours
    full_track = full_track[~full_track["timestamp_h"].between(gap_start, gap_end)]

    return full_track.sort_values("timestamp_h").reset_index(drop=True)
