"""
AIS gap ("dark vessel") detection.

A vessel responsible for a spill has strong incentive to disable or spoof
its AIS transponder around the incident window. This module finds those
gaps per vessel and flags the ones that overlap a spill's estimated origin
window as high-suspicion, even when there's no continuous track linking
the vessel to the spill location.
"""

from __future__ import annotations
import pandas as pd


def detect_gaps(ais_df: pd.DataFrame, gap_threshold_hours: float = 0.75) -> pd.DataFrame:
    """
    For each vessel (mmsi), find consecutive-ping gaps larger than the
    threshold (normal AIS pings are every 6-12 min, so anything beyond
    ~45min-1h is already anomalous for an actively-transmitting vessel).
    Returns one row per gap with the vessel's position immediately before
    and after, so later stages can check if the gap plausibly covers a
    detour to the spill site.
    """
    gaps = []
    for mmsi, grp in ais_df.groupby("mmsi"):
        grp = grp.sort_values("timestamp_h").reset_index(drop=True)
        for i in range(len(grp) - 1):
            dt = grp.loc[i + 1, "timestamp_h"] - grp.loc[i, "timestamp_h"]
            if dt > gap_threshold_hours:
                gaps.append({
                    "mmsi": mmsi,
                    "vessel_type": grp.loc[i, "vessel_type"],
                    "gap_start_h": grp.loc[i, "timestamp_h"],
                    "gap_end_h": grp.loc[i + 1, "timestamp_h"],
                    "gap_duration_h": dt,
                    "lat_before": grp.loc[i, "lat"], "lon_before": grp.loc[i, "lon"],
                    "lat_after": grp.loc[i + 1, "lat"], "lon_after": grp.loc[i + 1, "lon"],
                })
    return pd.DataFrame(gaps)


def gap_relevance(gaps_df: pd.DataFrame, origin_time_hours: float,
                   origin_window_hours: float = 4.0) -> pd.DataFrame:
    """
    Scores each gap 0-1 by how well it overlaps the spill's likely origin
    window. A gap that fully contains the origin time is most suspicious;
    gaps far from it in time score near zero.
    """
    if gaps_df.empty:
        return gaps_df.assign(gap_relevance=[])

    def relevance(row):
        window_start = origin_time_hours - origin_window_hours / 2
        window_end = origin_time_hours + origin_window_hours / 2
        overlap_start = max(row["gap_start_h"], window_start)
        overlap_end = min(row["gap_end_h"], window_end)
        overlap = max(0.0, overlap_end - overlap_start)
        return min(1.0, overlap / (origin_window_hours / 2))

    gaps_df = gaps_df.copy()
    gaps_df["gap_relevance"] = gaps_df.apply(relevance, axis=1)
    return gaps_df
