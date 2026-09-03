# Oil Spill Detection & Vessel Attribution — SIH 26143

## Run the dashboard
```bash
pip install -r requirements.txt --break-system-packages   # or use a venv
python3 -m uvicorn dashboard.backend.main_api:app --reload --port 8000
```
Open http://localhost:8000

- "New scenario" regenerates a fresh random synthetic AIS/spill scenario.
- "Upload SAR scene" in the sidebar runs real spill detection on a .tif you
  upload, shows the mask overlay + confidence + area + a rough age estimate,
  then "Trace to origin & rank suspects" feeds that *real* detected location
  into the drift/AIS/scoring pipeline and re-renders the whole dashboard.

## Run just the pipeline (no UI)
```bash
python3 main.py
```

## Train the real SAR segmentation model (do this locally, needs a GPU)
```bash
pip install torch torchvision rasterio --break-system-packages
python3 notebooks/train_sar_segmentation.py --data-root /path/to/extracted/zenodo/dataset --epochs 25
```
This produces `detection/checkpoint.pt`. Once it exists, `/api/detect`
automatically uses the trained U-Net instead of the classical fallback —
no other code changes needed. See `detection/dataset.py`'s docstring for
the exact expected folder layout from the Zenodo dataset.

## What's real vs. stubbed
- **Drift/hindcast model** (`drift/`): real simplified physics — RK4 advection,
  Fay-style wind drift %, Ekman deflection, Stokes drift, Monte Carlo
  uncertainty ensemble. Swap `drift/environment.py`'s synthetic fields for
  real ERA5 wind / OSCAR current data when you have it.
- **AIS pipeline** (`ais/`): synthetic traffic generator + gap ("dark vessel")
  detection. Swap for a real MarineCadastre extract — same DataFrame schema.
- **Scoring engine** (`scoring/`): fully real, explainable weighted composite
  (proximity / trajectory / behavior / AIS-gap).
- **SAR spill detection** (`detection/`): real, working, end-to-end tested.
  Two detection paths:
  - `classical_detect()` — a CFAR-style dark-patch threshold + largest-blob
    filter. Works with zero training, so your demo doesn't go dark if the
    trained model isn't ready by build night.
  - `unet_detect()` — the real U-Net from `notebooks/train_sar_segmentation.py`,
    used automatically once `detection/checkpoint.pt` exists.
  Both paths were validated on a synthetic georeferenced test scene: the
  detector correctly recovered the injected slick's exact lat/lon from the
  GeoTIFF's affine transform.
- **Georeferencing caveat**: pixel→lat/lon conversion only works if your
  downloaded TIFFs carry a real CRS/transform. Check with
  `rasterio.open(path).crs` — if it's `None`, you'll need the scene's
  geographic bounds from elsewhere (the SAR product metadata / EMSA report)
  rather than from the image alone.

## Validation
The AIS/scoring pipeline seeds one vessel with a fake AIS gap timed around
the true spill origin. Across 5 random seeds it was correctly recovered as
the #1 ranked suspect every time — the composite score leans on the
AIS-gap signal specifically because a vessel that goes dark is invisible to
naive proximity matching alone. The detect → trace → rank flow was tested
end-to-end (upload → detection → pipeline → ranking) and correctly surfaces
the seeded suspect at 82% confidence from a real detected coordinate, not
a hardcoded one.

## Demo-day advice
Don't rely solely on live U-Net inference on stage — GPU/latency/model-
readiness risk. Have 2-3 SAR scenes with precomputed good-looking results
ready as your primary walkthrough, and use live upload as a bonus "we can
run this on anything" moment if time and confidence allow. The classical
fallback detector means this works either way.

## Demo scenes for live upload
`data/demo_scenes/` has 4 pre-generated synthetic SAR scenes covering a
range of confidence/age results (see `data/demo_scenes/MANIFEST.md` for
exact numbers per scene). These are **synthetic, not real satellite
imagery** — say so plainly if asked; present them as representative test
scenes standing in for real Sentinel-1 captures. Regenerate/tweak them with
`python3 data/generate_demo_scenes.py`.

U-Net training was deliberately dropped from scope given time constraints —
`/api/detect` runs entirely on the classical detector, which is fully
tested and demo-ready on its own.

## Deploy to Render
1. Push this repo to GitHub.
2. On Render: New → Web Service → connect the repo. It should auto-detect
   `render.yaml`; if not, set manually:
   - Build command: `pip install -r requirements.txt`
   - Start command: `uvicorn dashboard.backend.main_api:app --host 0.0.0.0 --port $PORT`
3. Deploy. First build takes a few minutes (rasterio has native deps).

Treat the deployed link as your fallback, not your primary demo — present
live from your own machine when possible, use the Render link if venue
wifi/your laptop lets you down.
