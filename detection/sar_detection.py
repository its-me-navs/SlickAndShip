"""
Inference: given a SAR scene, produce a binary oil-slick mask, its geometry
(area, centroid), and — if the TIFF carries real georeferencing (CRS +
affine transform) — the centroid converted to lat/lon so it can feed
straight into the drift model as SPILL_LAT/SPILL_LON.

Two detection paths:
  - classical_detect(): a CFAR-style dark-patch threshold + largest-blob
    filter. No training required. Oil dampens radar backscatter, so slicks
    show up as anomalously dark regions on Sigma0 imagery — this is the
    same physical principle real operational detectors use as a first pass,
    just without the deep-learning refinement. Use this as your fallback if
    the trained model isn't ready by demo day.
  - unet_detect(): loads a checkpoint trained by
    notebooks/train_sar_segmentation.py and runs the real model.

IMPORTANT — georeferencing: the Zenodo dataset's TIFFs may or may not carry
a real CRS/affine transform (check with `rasterio.open(path).crs` — if it
prints None, they don't). If they don't, you cannot get a real lat/lon from
the image alone; you'll need to supply the scene's geographic bounding box
separately (from the SAR product's metadata / EMSA report) via
`scene_bounds` below.
"""

from __future__ import annotations
import numpy as np
from scipy import ndimage
import os
from pathlib import Path

try:
    import rasterio
    from rasterio.transform import Affine
except ImportError:
    rasterio = None


def _read_scene(path: str):
    """Returns (array [H,W] or [H,W,2], transform_or_None, crs_or_None)."""
    if rasterio is None:
        raise ImportError("pip install rasterio --break-system-packages")
    with rasterio.open(path) as src:
        arr = src.read()  # (bands, H, W)
        arr = np.moveaxis(arr, 0, -1)
        transform = src.transform if src.crs is not None else None
        crs = src.crs
    return arr, transform, crs


def classical_detect(sar_db: np.ndarray, dark_percentile: float = 12.0,
                      min_region_px: int = 150) -> tuple[np.ndarray, float]:
    """
    sar_db: single-channel Sigma0 dB array.
    Thresholds the darkest `dark_percentile`% of pixels, cleans up noise
    with binary opening, then keeps only the largest connected component
    (assumes one dominant slick per scene, reasonable for this dataset).
    Returns (binary_mask, confidence) where confidence is a rough proxy
    based on how much darker the region is than the surrounding sea.
    """
    thresh = np.percentile(sar_db, dark_percentile)
    raw_mask = (sar_db < thresh).astype(np.uint8)
    cleaned = ndimage.binary_opening(raw_mask, structure=np.ones((3, 3))).astype(np.uint8)

    labeled, n_regions = ndimage.label(cleaned)
    if n_regions == 0:
        return np.zeros_like(cleaned), 0.0

    sizes = ndimage.sum(cleaned, labeled, range(1, n_regions + 1))
    largest_label = int(np.argmax(sizes) + 1)
    if sizes[largest_label - 1] < min_region_px:
        return np.zeros_like(cleaned), 0.0

    mask = (labeled == largest_label).astype(np.uint8)

    sea_db = sar_db[cleaned == 0].mean() if (cleaned == 0).any() else sar_db.mean()
    slick_db = sar_db[mask == 1].mean()
    contrast_db = sea_db - slick_db  # oil is darker -> positive contrast
    confidence = float(np.clip(contrast_db / 8.0, 0.0, 1.0))  # ~8dB contrast -> high confidence

    return mask, confidence


_MODEL_CACHE: dict = {}


def _load_unet(checkpoint_path: str, device: str):
    key = (checkpoint_path, device)
    if key not in _MODEL_CACHE:
        import torch
        from .unet_model import UNet
        model = UNet(in_channels=2, out_channels=1, base=32)
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        _MODEL_CACHE[key] = model.eval().to(device)
    return _MODEL_CACHE[key]


def unet_detect(sar_2ch: np.ndarray, checkpoint_path: str, device: str = "cpu",
                tile: int = 128, stride: int = 128, batch_size: int = 16,
                threshold: float | None = None, min_region_px: int = 150) -> tuple[np.ndarray, float]:
    """Batched sliding-window U-Net inference over the full scene (the model
    was trained on 128x128 crops, so it must not see the whole image at once)."""
    import os
    import torch

    if sar_2ch.ndim != 3 or sar_2ch.shape[-1] != 2:
        raise ValueError(f"U-Net needs a 2-channel (VV+VH) scene, got shape {sar_2ch.shape}")
    if threshold is None:
        threshold = float(os.environ.get("UNET_THRESHOLD", "0.5"))

    model = _load_unet(checkpoint_path, device)

    img = np.nan_to_num(sar_2ch.astype(np.float32), nan=-35.0)
    img = (np.clip(img, -35, 5) + 35) / 40.0          # same normalization as training
    h, w = img.shape[:2]
    if h < tile or w < tile:
        raise ValueError(f"Scene {h}x{w} is smaller than the {tile}px tile")

    x = torch.from_numpy(np.moveaxis(img, -1, 0)).float()   # (2, H, W)
    tops = list(range(0, h - tile + 1, stride))
    lefts = list(range(0, w - tile + 1, stride))
    if tops[-1] != h - tile:
        tops.append(h - tile)
    if lefts[-1] != w - tile:
        lefts.append(w - tile)
    coords = [(t, l) for t in tops for l in lefts]

    prob_sum = np.zeros((h, w), dtype=np.float32)
    count = np.zeros((h, w), dtype=np.float32)
    with torch.inference_mode():
        for i in range(0, len(coords), batch_size):
            chunk = coords[i:i + batch_size]
            batch = torch.stack([x[:, t:t + tile, l:l + tile] for t, l in chunk]).to(device)
            out = torch.sigmoid(model(batch))[:, 0].cpu().numpy()
            for (t, l), p in zip(chunk, out):
                prob_sum[t:t + tile, l:l + tile] += p
                count[t:t + tile, l:l + tile] += 1
    probs = prob_sum / np.maximum(count, 1)

    mask = probs >= threshold
    labeled, n = ndimage.label(mask)                   # drop tiny speckle blobs
    if n:
        sizes = ndimage.sum(mask, labeled, range(1, n + 1))
        keep = [i + 1 for i, s in enumerate(sizes) if s >= min_region_px]
        mask = np.isin(labeled, keep)
    mask = mask.astype(np.uint8)

    confidence = float(probs[mask == 1].mean()) if mask.any() else 0.0
    return mask, confidence


def _pixel_to_lonlat(row: float, col: float, transform: "Affine") -> tuple[float, float]:
    lon, lat = transform * (col, row)
    return lat, lon


def compute_geometry(mask: np.ndarray, transform=None, pixel_size_m: float = 10.0) -> dict:
    """
    Area/perimeter/centroid from the mask. If a real geo transform is
    available, centroid is also converted to lat/lon. Area falls back to
    an assumed 10m/pixel Sentinel-1 ground resolution if no transform is
    present (documented, not silently wrong — flagged in the result).
    """
    ys, xs = np.where(mask == 1)
    if len(ys) == 0:
        return {"has_detection": False}

    centroid_row, centroid_col = float(ys.mean()), float(xs.mean())
    n_pixels = int(mask.sum())

    perimeter_mask = mask - ndimage.binary_erosion(mask).astype(mask.dtype)
    perimeter_px = int(perimeter_mask.sum())

    geo_available = transform is not None
    if geo_available:
        lat, lon = _pixel_to_lonlat(centroid_row, centroid_col, transform)
        # pixel size in meters, approximated from the transform's scale
        px_size_m = abs(transform.a) * 111_000 if abs(transform.a) < 1 else abs(transform.a)
    else:
        lat, lon = None, None
        px_size_m = pixel_size_m

    area_km2 = n_pixels * (px_size_m ** 2) / 1_000_000
    perimeter_km = perimeter_px * px_size_m / 1000

    return {
        "has_detection": True,
        "geo_available": geo_available,
        "lat": lat, "lon": lon,
        "centroid_pixel": {"row": centroid_row, "col": centroid_col},
        "area_km2": round(area_km2, 3),
        "perimeter_km": round(perimeter_km, 3),
        "n_pixels": n_pixels,
    }


def estimate_age_heuristic(sar_db: np.ndarray, mask: np.ndarray, geometry: dict) -> str:
    """
    Rough, unvalidated heuristic: fresh slicks tend to be more compact and
    sharply defined (high contrast, low compactness ratio); weathered/older
    slicks spread, fragment, and lose contrast as they emulsify. This is
    NOT a calibrated model — flag it as a heuristic in the UI, not a fact.
    """
    if not geometry.get("has_detection"):
        return "no detection"
    compactness = (geometry["perimeter_km"] ** 2) / (4 * np.pi * max(geometry["area_km2"], 1e-6))
    if compactness < 1.05:
        return "likely recent (compact, well-defined edges)"
    elif compactness < 1.5:
        return "moderate age (some spreading/fragmentation)"
    else:
        return "likely weathered (fragmented, diffuse — treat with caution, verify manually)"


def generate_overlay_png(sar_db: np.ndarray, mask: np.ndarray) -> bytes:
    """
    Grayscale SAR scene with the detected mask outlined in red, as a PNG
    (bytes) for the frontend to show without needing the raw array.
    """
    from PIL import Image, ImageDraw

    norm = np.clip((sar_db - sar_db.min()) / (sar_db.max() - sar_db.min() + 1e-9) * 255, 0, 255)
    base = Image.fromarray(norm.astype(np.uint8)).convert("RGB")

    boundary = (mask.astype(np.uint8) - ndimage.binary_erosion(mask).astype(np.uint8))
    overlay = np.array(base)
    overlay[boundary == 1] = [226, 87, 76]  # matches --accent-spill in the dashboard CSS
    fill_alpha = 0.18
    red_fill = np.array([226, 87, 76])
    mask_bool = mask.astype(bool)
    overlay[mask_bool] = (overlay[mask_bool] * (1 - fill_alpha) + red_fill * fill_alpha).astype(np.uint8)

    out = Image.fromarray(overlay)
    buf = __import__("io").BytesIO()
    out.save(buf, format="PNG")
    return buf.getvalue()


def detect_spill(image_path: str, checkpoint_path: str | None = None,
                  mode: str = "auto") -> dict:
    """
    mode: 'auto' (try U-Net if checkpoint_path given and torch available,
    else classical), 'classical', or 'unet'.
    """
    arr, transform, crs = _read_scene(image_path)
    single_channel = arr[..., 0] if arr.ndim == 3 else arr
    if checkpoint_path is None and os.environ.get("USE_UNET") == "1":
        checkpoint_path = str(Path(__file__).resolve().parents[1] / "best_unet.pt")

    use_unet = mode == "unet" or (mode == "auto" and checkpoint_path is not None)
    if use_unet:
        try:
            mask, confidence = unet_detect(arr, checkpoint_path)
            method = "unet"
        except Exception as e:
            mask, confidence = classical_detect(single_channel)
            method = f"classical (unet failed: {type(e).__name__}: {e})"
    else:
        mask, confidence = classical_detect(single_channel)
        method = "classical"

    geometry = compute_geometry(mask, transform)
    age = estimate_age_heuristic(single_channel, mask, geometry)
    overlay_png = generate_overlay_png(single_channel, mask)

    import base64
    return {
        "method": method,
        "confidence": round(confidence, 3),
        "age_estimate": age,
        "crs": str(crs) if crs else None,
        "overlay_png_base64": base64.b64encode(overlay_png).decode("ascii"),
        **geometry,
    }
