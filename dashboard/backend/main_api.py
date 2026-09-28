"""
FastAPI backend for the oil spill / vessel attribution dashboard.

Run from the project root:
    pip install fastapi uvicorn --break-system-packages
    python3 -m uvicorn dashboard.backend.main_api:app --reload --port 8000

Then open http://localhost:8000
"""

import os
import sys
import tempfile
import time
from pathlib import Path

# allow `from pipeline import run_full_pipeline` when running from anywhere
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from pipeline import run_full_pipeline
from detection.sar_detection import detect_spill

CHECKPOINT_PATH = str(Path(__file__).resolve().parents[2] / "best_unet.pt")

app = FastAPI(title="Oil Spill Attribution API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_cache = {}


@app.get("/api/pipeline")
def api_pipeline():
    """Returns the reproducible default demo scenario, cached after first run."""
    if "result" not in _cache:
        _cache["result"] = run_full_pipeline()
    return _cache["result"]


@app.get("/api/pipeline/refresh")
def api_pipeline_refresh():
    """Generates a fresh random synthetic scenario (for the 'New scenario' button)."""
    seed = int(time.time())
    _cache["result"] = run_full_pipeline(seed=seed)
    return _cache["result"]


@app.post("/api/detect")
async def api_detect(file: UploadFile = File(...)):
    """
    Runs spill detection on an uploaded SAR TIFF. Uses the trained U-Net
    checkpoint at detection/checkpoint.pt if it exists (train it with
    notebooks/train_sar_segmentation.py), otherwise falls back to the
    classical dark-patch detector automatically — the demo works either way.
    """
    suffix = Path(file.filename).suffix or ".tif"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        checkpoint = CHECKPOINT_PATH if os.path.exists(CHECKPOINT_PATH) else None
        result = detect_spill(tmp_path, checkpoint_path=checkpoint, mode="auto")
    finally:
        os.unlink(tmp_path)

    return result


@app.get("/api/pipeline/from-detection")
def api_pipeline_from_detection(lat: float, lon: float, seed: int | None = None):
    """
    Runs the full pipeline using a real detected spill location (from
    /api/detect) instead of the hardcoded demo stub. AIS/drift data around
    it is still synthetic — see run_full_pipeline's docstring.
    """
    run_seed = seed if seed is not None else int(time.time())
    _cache["result"] = run_full_pipeline(seed=run_seed, spill_override={"lat": lat, "lon": lon})
    return _cache["result"]


frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")
