"""
Runs the detection->drift->AIS->scoring pipeline once and caches the
results in JSON-friendly shapes for the FastAPI layer to serve.

Kept separate from main.py (which is the CLI/demo entry point) so the API
and the CLI script share the exact same pipeline logic without duplicating it.
"""

import sys
from pathlib import Path

# make the project root importable regardless of where uvicorn is launched from
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from drift.drift_model import monte_carlo_backtrack, forward_forecast
from ais.synthetic_ais import generate_background_traffic, inject_guilty_vessel
from ais.gap_detection import detect_gaps, gap_relevance
from scoring.scoring_engine import reconstruct_traffic, rank_suspects

SPILL_LAT, SPILL_LON = 15.30, 73.40
SPILL_DETECTION_TIME_H = 100.0
HINDCAST_WINDOW_H = 18.0
FORECAST_WINDOW_H = 12.0
REGION_BOUNDS = {"lat_min": 14.9, "lat_max": 15.7, "lon_min": 72.9, "lon_max": 73.9}
T_START, T_END = 70.0, 110.0
GUILTY_MMSI = 999_999_999

_cache = {}


def run_pipeline(force: bool = False) -> dict:
    """Runs the full pipeline once, caches, and returns a JSON-serializable dict."""
    if _cache and not force:
        return _cache

    origin_ensemble = monte_carlo_backtrack(
        SPILL_LAT, SPILL_LON, SPILL_DETECTION_TIME_H,
        hindcast_hours=HINDCAST_WINDOW_H, n_ensemble=60,
    )
    lats = np.array([o[0] for o in origin_ensemble])
    lons = np.array([o[1] for o in origin_ensemble])
    origin_lat, origin_lon = float(lats.mean()), float(lons.mean())
    origin_time = SPILL_DETECTION_TIME_H - HINDCAST_WINDOW_H
    spread_km = float(lats.std() * 111)

    background = generate_background_traffic(REGION_BOUNDS, T_START, T_END, n_vessels=15)
    guilty = inject_guilty_vessel(origin_lat, origin_lon, origin_time, T_START, T_END,
                                   gap_before_hours=1.5, gap_after_hours=2.0, mmsi=GUILTY_MMSI)
    ais_all = pd.concat([background, guilty], ignore_index=True)

    traffic = reconstruct_traffic(ais_all, REGION_BOUNDS, T_START, T_END)
    gaps = detect_gaps(ais_all, gap_threshold_hours=0.75)
    gaps_scored = gap_relevance(gaps, origin_time, origin_window_hours=4.0)
    ranked = rank_suspects(traffic, gaps_scored, origin_ensemble, origin_lat, origin_lon, origin_time)

    forward_path = forward_forecast(SPILL_LAT, SPILL_LON, SPILL_DETECTION_TIME_H, FORECAST_WINDOW_H)

    # vessel tracks as JSON-friendly polylines, keyed by mmsi
    tracks = {}
    for mmsi, grp in ais_all.groupby("mmsi"):
        grp = grp.sort_values("timestamp_h")
        tracks[str(int(mmsi))] = {
            "vessel_type": str(grp["vessel_type"].iloc[0]),
            "points": [
                {"lat": float(r.lat), "lon": float(r.lon), "t": float(r.timestamp_h),
                 "sog": float(r.sog_knots), "cog": float(r.cog_deg)}
                for r in grp.itertuples()
            ],
        }

    result = {
        "spill": {
            "lat": SPILL_LAT, "lon": SPILL_LON, "detection_time_h": SPILL_DETECTION_TIME_H,
        },
        "origin": {
            "lat": origin_lat, "lon": origin_lon, "time_h": origin_time,
            "spread_km": round(spread_km, 2),
            "ensemble": [{"lat": float(o[0]), "lon": float(o[1]), "t": float(o[2])} for o in origin_ensemble],
        },
        "forward_forecast": [{"lat": float(p[0]), "lon": float(p[1]), "t": float(p[2])} for p in forward_path],
        "region_bounds": REGION_BOUNDS,
        "tracks": tracks,
        "gaps": gaps_scored.to_dict(orient="records") if not gaps_scored.empty else [],
        "suspects": ranked.to_dict(orient="records"),
        "guilty_mmsi": str(GUILTY_MMSI),  # for demo validation banner only
    }
    _cache.clear()
    _cache.update(result)
    return result
