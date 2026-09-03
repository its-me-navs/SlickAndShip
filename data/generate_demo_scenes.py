"""
Generates a handful of varied synthetic SAR-style scenes for live demo
upload — NOT real satellite imagery. Each is a georeferenced GeoTIFF
(Sigma0-style dB values + realistic speckle noise) with a slick region
placed inside the pipeline's operational area (drift_model/pipeline.py's
REGION_BOUNDS), so a detected spill here produces a sensible drift-back
origin and AIS correlation when traced through the rest of the pipeline.

Present these honestly in the demo as representative test scenes standing
in for real Sentinel-1 captures, not as real satellite data.
"""

import os
import numpy as np
import rasterio
from rasterio.transform import from_origin

OUT_DIR = os.path.join(os.path.dirname(__file__), "demo_scenes")
os.makedirs(OUT_DIR, exist_ok=True)

# same region the rest of the pipeline operates in (see pipeline.py REGION_BOUNDS)
REGION = {"lat_min": 14.9, "lat_max": 15.7, "lon_min": 72.9, "lon_max": 73.9}
H = 800
PIXEL_DEG = (REGION["lat_max"] - REGION["lat_min"]) / H  # square pixels, degrees
# region isn't square (0.8 deg lat x 1.0 deg lon) -- W must cover the full lon
# span at the same per-pixel degree size, not just reuse H, or scenes near the
# eastern edge silently fall outside the raster.
W = round((REGION["lon_max"] - REGION["lon_min"]) / PIXEL_DEG)


def make_scene(seed: int, center_lat: float, center_lon: float, size_px: float,
               elongation: float, angle_deg: float, contrast_db: float,
               fragments: int, noise_std: float = 1.5, sea_mean_db: float = -15.0):
    rng = np.random.default_rng(seed)
    sea = rng.normal(sea_mean_db, noise_std, size=(H, W)).astype(np.float32)

    top_lat = REGION["lat_max"]
    left_lon = REGION["lon_min"]
    row = int((top_lat - center_lat) / PIXEL_DEG)
    col = int((center_lon - left_lon) / PIXEL_DEG)

    yy, xx = np.mgrid[0:H, 0:W]
    theta = np.radians(angle_deg)
    yr = (yy - row) * np.cos(theta) - (xx - col) * np.sin(theta)
    xr = (yy - row) * np.sin(theta) + (xx - col) * np.cos(theta)
    blob = (yr**2 / size_px**2 + xr**2 / (size_px * elongation)**2) < 1

    mask = blob.copy()
    for i in range(fragments):
        fy = row + rng.integers(-int(size_px * 2), int(size_px * 2))
        fx = col + rng.integers(-int(size_px * 2), int(size_px * 2))
        fr = size_px * rng.uniform(0.15, 0.35)
        frag = ((yy - fy)**2 + (xx - fx)**2) < fr**2
        mask |= frag

    slick_vals = rng.normal(sea_mean_db - contrast_db, noise_std * 0.8, size=(H, W)).astype(np.float32)
    sar = np.where(mask, slick_vals, sea)
    sar2 = sar + rng.normal(0, noise_std * 0.6, size=(H, W)).astype(np.float32)  # 2nd polarization channel

    transform = from_origin(left_lon, top_lat, PIXEL_DEG, PIXEL_DEG)
    return np.stack([sar, sar2], axis=0), transform


SCENES = [
    dict(name="scene_A_compact_recent", seed=1, center_lat=15.32, center_lon=73.25,
         size_px=22, elongation=1.1, angle_deg=10, contrast_db=11, fragments=0,
         note="Compact, high-contrast, well-defined edges — reads as a recent spill."),
    dict(name="scene_B_large_offshore", seed=2, center_lat=15.55, center_lon=73.60,
         size_px=45, elongation=1.3, angle_deg=40, contrast_db=9, fragments=1,
         note="Larger slick, offshore position — good for showing area/geometry stats."),
    dict(name="scene_C_elongated_weathered", seed=3, center_lat=15.15, center_lon=73.15,
         size_px=22, elongation=4.5, angle_deg=70, contrast_db=7, fragments=3,
         note="Elongated + fragmented — reads as weathered/older in the age heuristic."),
    dict(name="scene_D_faint_lowconf", seed=4, center_lat=15.40, center_lon=73.75,
         size_px=20, elongation=1.4, angle_deg=25, contrast_db=5.0, fragments=0,
         note="Subtle/low-contrast — deliberately shows a lower confidence score, not every detection is a slam dunk."),
]

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from detection.sar_detection import detect_spill

    manifest_lines = ["# Demo scenes\n", "Synthetic (not real satellite imagery) — for live upload during the demo.\n"]
    for s in SCENES:
        arr, transform = make_scene(s["seed"], s["center_lat"], s["center_lon"], s["size_px"],
                                     s["elongation"], s["angle_deg"], s["contrast_db"], s["fragments"])
        path = os.path.join(OUT_DIR, s["name"] + ".tif")
        with rasterio.open(path, "w", driver="GTiff", height=H, width=W, count=2,
                            dtype="float32", crs="EPSG:4326", transform=transform) as dst:
            dst.write(arr)

        result = detect_spill(path, checkpoint_path=None, mode="classical")
        print(f"{s['name']:30s} conf={result['confidence']:.2f}  area={result.get('area_km2', 0):6.2f} km²  "
              f"age='{result['age_estimate']}'  lat/lon=({result.get('lat')}, {result.get('lon')})")

        loc_str = f"{result.get('lat'):.4f}, {result.get('lon'):.4f}" if result.get("has_detection") else "no detection"
        manifest_lines.append(
            f"### `{s['name']}.tif`\n{s['note']}\n\n"
            f"- Detected confidence: {result['confidence']:.2f}\n"
            f"- Area: {result.get('area_km2', 0):.2f} km²\n"
            f"- Age estimate: {result['age_estimate']}\n"
            f"- Detected location: {loc_str}\n\n"
        )

    with open(os.path.join(OUT_DIR, "MANIFEST.md"), "w") as f:
        f.writelines(manifest_lines)
    print(f"\nWrote {len(SCENES)} demo scenes + MANIFEST.md to {OUT_DIR}")
