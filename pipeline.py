"""
Callable version of the end-to-end pipeline (see main.py for the CLI/demo
version this was extracted from). Returns plain Python structures matching
the frontend's expected JSON contract (dashboard/frontend/app.js) so the
API layer can pass this straight through.
"""

import time
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


def run_full_pipeline(seed: int | None = None, spill_override: dict | None = None) -> dict:
    """
    seed=None uses the fixed default seeds (reproducible demo run).
    Pass an int (e.g. derived from time.time()) to generate a fresh
    random scenario for the "New scenario" button.

    spill_override: optional {"lat": .., "lon": ..} from a real SAR
    detection (see detection/sar_detection.py + POST /api/detect). Only the
    spill location is real when this is used — detection time stays on the
    synthetic region timeline, since there's no real AIS traffic in this
    sandbox to correlate against a real timestamp.
    """
    spill_lat = spill_override["lat"] if spill_override else SPILL_LAT
    spill_lon = spill_override["lon"] if spill_override else SPILL_LON

    bt_seed = 42 if seed is None else seed
    bg_seed = 7 if seed is None else seed + 1
    guilty_seed = 99 if seed is None else seed + 2

    origin_ensemble = monte_carlo_backtrack(
        spill_lat, spill_lon, SPILL_DETECTION_TIME_H,
        hindcast_hours=HINDCAST_WINDOW_H, n_ensemble=60, seed=bt_seed,
    )
    lats = np.array([o[0] for o in origin_ensemble])
    lons = np.array([o[1] for o in origin_ensemble])
    origin_lat, origin_lon = float(lats.mean()), float(lons.mean())
    origin_time = SPILL_DETECTION_TIME_H - HINDCAST_WINDOW_H
    spread_km = float(lats.std() * 111)

    forward_path = forward_forecast(spill_lat, spill_lon, SPILL_DETECTION_TIME_H, FORECAST_WINDOW_H)

    background = generate_background_traffic(REGION_BOUNDS, T_START, T_END, n_vessels=15, seed=bg_seed)
    guilty = inject_guilty_vessel(origin_lat, origin_lon, origin_time, T_START, T_END,
                                   gap_before_hours=1.5, gap_after_hours=2.0,
                                   mmsi=GUILTY_MMSI, seed=guilty_seed)
    ais_all = pd.concat([background, guilty], ignore_index=True)

    traffic = reconstruct_traffic(ais_all, REGION_BOUNDS, T_START, T_END)

    gaps = detect_gaps(ais_all, gap_threshold_hours=0.75)
    gaps_scored = gap_relevance(gaps, origin_time, origin_window_hours=4.0)

    ranked = rank_suspects(traffic, gaps_scored, origin_ensemble, origin_lat, origin_lon, origin_time)
    ranked["mmsi"] = ranked["mmsi"].astype(int)

    # tracks keyed by string mmsi, points as {lat, lon, t_h, sog_knots, cog_deg}
    tracks = {}
    for mmsi, grp in ais_all.groupby("mmsi"):
        grp = grp.sort_values("timestamp_h")
        tracks[str(int(mmsi))] = {
            "vessel_type": grp["vessel_type"].iloc[0],
            "points": [
                {"lat": round(r.lat, 5), "lon": round(r.lon, 5), "t_h": round(r.timestamp_h, 2),
                 "sog_knots": round(r.sog_knots, 1), "cog_deg": round(r.cog_deg, 1)}
                for r in grp.itertuples()
            ],
        }

    gaps_records = gaps_scored.round(3).to_dict(orient="records") if not gaps_scored.empty else []

    return {
        "spill": {"lat": spill_lat, "lon": spill_lon, "detection_time_h": SPILL_DETECTION_TIME_H},
        "origin": {
            "lat": origin_lat, "lon": origin_lon, "time_h": origin_time,
            "spread_km": round(spread_km, 2),
            "ensemble": [{"lat": round(o[0], 5), "lon": round(o[1], 5)} for o in origin_ensemble],
        },
        "forward_forecast": [{"lat": round(p[0], 5), "lon": round(p[1], 5)} for p in forward_path],
        "region_bounds": REGION_BOUNDS,
        "tracks": tracks,
        "gaps": gaps_records,
        "suspects": ranked.to_dict(orient="records"),
        "guilty_mmsi": str(GUILTY_MMSI),  # demo ground truth; UI never labels it, just doesn't double-draw it
    }
