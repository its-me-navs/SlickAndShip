"""
Suspect scoring engine.

Combines four independent signals into one explainable, weighted confidence
score per vessel:
  - proximity_score   : how close the vessel's (interpolated) position gets
                         to the origin ensemble cloud, in the origin window
  - trajectory_score  : whether the vessel's heading is consistent with
                         having been at the origin (not just physically near it)
  - behavior_score    : anomalies relative to the vessel's own normal
                         speed/course (sudden slowdowns, erratic turns)
  - gap_score          : AIS-gap relevance from gap_detection.gap_relevance

Each component and the breakdown are kept in the output so the final rank
is defensible, not a black-box number.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

KM_PER_DEG_LAT = 111.0
DEFAULT_WEIGHTS = {"proximity": 0.35, "trajectory": 0.15, "behavior": 0.15, "gap": 0.35}


def _km_per_deg_lon(lat: float) -> float:
    return 111.0 * np.cos(np.radians(lat))


def _haversine_km(lat1, lon1, lat2, lon2) -> float:
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * 6371 * np.arcsin(np.sqrt(a))


def reconstruct_traffic(ais_df: pd.DataFrame, region_bounds: dict,
                         t_start_hours: float, t_end_hours: float) -> pd.DataFrame:
    """Filter AIS traffic to the region and time window around the incident."""
    mask = (
        ais_df["lat"].between(region_bounds["lat_min"], region_bounds["lat_max"])
        & ais_df["lon"].between(region_bounds["lon_min"], region_bounds["lon_max"])
        & ais_df["timestamp_h"].between(t_start_hours, t_end_hours)
    )
    return ais_df[mask].copy()


def _interpolate_position(track: pd.DataFrame, t: float):
    """Linear interpolation of a vessel's position at time t from its pings."""
    track = track.sort_values("timestamp_h")
    if t <= track["timestamp_h"].iloc[0]:
        row = track.iloc[0]
        return row["lat"], row["lon"]
    if t >= track["timestamp_h"].iloc[-1]:
        row = track.iloc[-1]
        return row["lat"], row["lon"]
    before = track[track["timestamp_h"] <= t].iloc[-1]
    after = track[track["timestamp_h"] >= t].iloc[0]
    if after["timestamp_h"] == before["timestamp_h"]:
        return before["lat"], before["lon"]
    frac = (t - before["timestamp_h"]) / (after["timestamp_h"] - before["timestamp_h"])
    lat = before["lat"] + frac * (after["lat"] - before["lat"])
    lon = before["lon"] + frac * (after["lon"] - before["lon"])
    return lat, lon


def proximity_score(track: pd.DataFrame, origin_ensemble: list[tuple[float, float, float]],
                     decay_km: float = 8.0) -> float:
    """
    Min distance from the vessel's interpolated position (at each origin
    candidate's time) to that origin candidate, converted to a 0-1 score
    via exponential decay. decay_km controls how forgiving the match is.
    """
    if track.empty:
        return 0.0
    best_dist = np.inf
    for (olat, olon, ot) in origin_ensemble:
        vlat, vlon = _interpolate_position(track, ot)
        d = _haversine_km(vlat, vlon, olat, olon)
        best_dist = min(best_dist, d)
    return float(np.exp(-best_dist / decay_km))


def trajectory_score(track: pd.DataFrame, origin_lat: float, origin_lon: float,
                      origin_time: float) -> float:
    """
    Checks whether the vessel's heading around the origin time actually
    points toward/through the origin, not just that it happened to be
    nearby while going somewhere unrelated.
    """
    if track.empty or len(track) < 2:
        return 0.0
    track = track.sort_values("timestamp_h")
    nearest_idx = (track["timestamp_h"] - origin_time).abs().idxmin()
    row = track.loc[nearest_idx]
    bearing_to_origin = np.degrees(np.arctan2(
        (origin_lon - row["lon"]) * _km_per_deg_lon(row["lat"]),
        (origin_lat - row["lat"]) * KM_PER_DEG_LAT,
    )) % 360
    heading_diff = abs((row["cog_deg"] - bearing_to_origin + 180) % 360 - 180)
    return float(max(0.0, 1 - heading_diff / 90))


def behavior_score(track: pd.DataFrame) -> float:
    """
    Flags anomalies relative to the vessel's own normal behavior: sudden
    speed drops (transferring cargo / tampering) or erratic course changes.
    """
    if len(track) < 3:
        return 0.0
    track = track.sort_values("timestamp_h")
    speed_std = track["sog_knots"].std()
    speed_mean = track["sog_knots"].mean()
    speed_cv = speed_std / speed_mean if speed_mean > 0 else 0

    course_diffs = track["cog_deg"].diff().abs().dropna()
    course_diffs = course_diffs.apply(lambda d: min(d, 360 - d))
    erratic_turns = (course_diffs > 30).mean() if len(course_diffs) else 0

    return float(min(1.0, 0.5 * min(speed_cv, 1.0) + 0.5 * erratic_turns))


def rank_suspects(ais_traffic: pd.DataFrame, gaps_with_relevance: pd.DataFrame,
                   origin_ensemble: list[tuple[float, float, float]],
                   origin_lat: float, origin_lon: float, origin_time: float,
                   weights: dict = None) -> pd.DataFrame:
    """
    Produces the final ranked suspect list. Vessels with a relevant AIS gap
    are included even if their visible track never gets close to the
    origin — that's the point of the dark-vessel channel.
    """
    weights = weights or DEFAULT_WEIGHTS
    rows = []
    all_mmsi = set(ais_traffic["mmsi"].unique())
    if not gaps_with_relevance.empty:
        all_mmsi |= set(gaps_with_relevance["mmsi"].unique())

    for mmsi in all_mmsi:
        track = ais_traffic[ais_traffic["mmsi"] == mmsi]
        vessel_type = track["vessel_type"].iloc[0] if not track.empty else (
            gaps_with_relevance[gaps_with_relevance["mmsi"] == mmsi]["vessel_type"].iloc[0]
        )

        prox = proximity_score(track, origin_ensemble)
        traj = trajectory_score(track, origin_lat, origin_lon, origin_time)
        beh = behavior_score(track)

        gap_rel = 0.0
        if not gaps_with_relevance.empty:
            vessel_gaps = gaps_with_relevance[gaps_with_relevance["mmsi"] == mmsi]
            if not vessel_gaps.empty:
                gap_rel = vessel_gaps["gap_relevance"].max()

        composite = (weights["proximity"] * prox + weights["trajectory"] * traj +
                     weights["behavior"] * beh + weights["gap"] * gap_rel)

        rows.append({
            "mmsi": mmsi, "vessel_type": vessel_type,
            "proximity_score": round(prox, 3), "trajectory_score": round(traj, 3),
            "behavior_score": round(beh, 3), "gap_score": round(gap_rel, 3),
            "confidence_pct": round(composite * 100, 1),
        })

    result = pd.DataFrame(rows).sort_values("confidence_pct", ascending=False).reset_index(drop=True)
    result.insert(0, "rank", np.arange(1, len(result) + 1))
    return result
