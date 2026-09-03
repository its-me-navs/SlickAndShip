"""
End-to-end demo pipeline (Phase 1 core logic).

Flow:
  1. [STUB] Spill detection -> would come from the SAR segmentation model
     (trained separately, see notebooks/). Hardcoded here for the demo.
  2. Backward Monte Carlo drift hindcast -> origin ensemble (probability cone)
  3. Synthetic AIS traffic generation (background + one seeded guilty vessel
     with an AIS gap around the origin window)
  4. Traffic reconstruction (filter to region/time)
  5. AIS gap detection + relevance scoring
  6. Suspect scoring + ranking

Run: python3 main.py
"""

import numpy as np
import pandas as pd

from drift.drift_model import monte_carlo_backtrack, forward_forecast
from ais.synthetic_ais import generate_background_traffic, inject_guilty_vessel
from ais.gap_detection import detect_gaps, gap_relevance
from scoring.scoring_engine import reconstruct_traffic, rank_suspects

pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 10)

# ---- Step 1: [STUB] spill detection output (would come from SAR model) ----
SPILL_LAT, SPILL_LON = 15.30, 73.40
SPILL_DETECTION_TIME_H = 100.0   # hours since region-of-interest start
HINDCAST_WINDOW_H = 18.0          # how far back we search for the origin

print("=" * 70)
print("STEP 1: Spill detection (stub)")
print(f"  Detected slick at ({SPILL_LAT}, {SPILL_LON}) at t={SPILL_DETECTION_TIME_H}h")

# ---- Step 2: backward drift hindcast ----
print("\nSTEP 2: Backward drift hindcast (Monte Carlo ensemble)")
origin_ensemble = monte_carlo_backtrack(
    SPILL_LAT, SPILL_LON, SPILL_DETECTION_TIME_H,
    hindcast_hours=HINDCAST_WINDOW_H, n_ensemble=60,
)
lats = np.array([o[0] for o in origin_ensemble])
lons = np.array([o[1] for o in origin_ensemble])
origin_lat, origin_lon = lats.mean(), lons.mean()
origin_time = SPILL_DETECTION_TIME_H - HINDCAST_WINDOW_H
spread_km = lats.std() * 111
print(f"  Estimated origin: ({origin_lat:.4f}, {origin_lon:.4f}) at t={origin_time}h")
print(f"  Ensemble spread (1 std): ~{spread_km:.2f} km")

# ---- Step 3: synthetic AIS traffic ----
print("\nSTEP 3: Generating synthetic AIS traffic")
region_bounds = {"lat_min": 14.9, "lat_max": 15.7, "lon_min": 72.9, "lon_max": 73.9}
t_start, t_end = 70.0, 110.0

background = generate_background_traffic(region_bounds, t_start, t_end, n_vessels=15)
guilty = inject_guilty_vessel(origin_lat, origin_lon, origin_time, t_start, t_end,
                               gap_before_hours=1.5, gap_after_hours=2.0)
ais_all = pd.concat([background, guilty], ignore_index=True)
print(f"  {background['mmsi'].nunique()} background vessels + 1 seeded guilty vessel (MMSI hidden from scorer)")
print(f"  Total AIS pings: {len(ais_all)}")

# ---- Step 4: traffic reconstruction ----
print("\nSTEP 4: Reconstructing traffic in region/time window")
traffic = reconstruct_traffic(ais_all, region_bounds, t_start, t_end)
print(f"  {traffic['mmsi'].nunique()} vessels in reconstructed window")

# ---- Step 5: gap detection ----
print("\nSTEP 5: AIS gap detection")
gaps = detect_gaps(ais_all, gap_threshold_hours=0.75)
gaps_scored = gap_relevance(gaps, origin_time, origin_window_hours=4.0)
if not gaps_scored.empty:
    top_gaps = gaps_scored.sort_values("gap_relevance", ascending=False).head(5)
    print(top_gaps[["mmsi", "gap_duration_h", "gap_relevance"]].to_string(index=False))
else:
    print("  No gaps detected.")

# ---- Step 6: suspect scoring ----
print("\nSTEP 6: Suspect scoring & ranking")
ranked = rank_suspects(traffic, gaps_scored, origin_ensemble, origin_lat, origin_lon, origin_time)
print(ranked.head(10).to_string(index=False))

guilty_mmsi = 999_999_999
guilty_rank = ranked[ranked["mmsi"] == guilty_mmsi]
print("\n" + "=" * 70)
if not guilty_rank.empty:
    r = guilty_rank.iloc[0]
    print(f"VALIDATION: seeded guilty vessel (MMSI {guilty_mmsi}) recovered at rank "
          f"{int(r['rank'])} with {r['confidence_pct']}% confidence")
else:
    print("VALIDATION: guilty vessel not found in ranked output (check region/time filters)")

ranked.to_csv("data/ranked_suspects.csv", index=False)
gaps_scored.to_csv("data/ais_gaps.csv", index=False)
traffic.to_csv("data/reconstructed_traffic.csv", index=False)
print("\nSaved: data/ranked_suspects.csv, data/ais_gaps.csv, data/reconstructed_traffic.csv")
