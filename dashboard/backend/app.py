"""
FastAPI backend for the oil spill attribution dashboard.

Run from the project root with:
    uvicorn dashboard.backend.app:app --reload --port 8000
Then open http://localhost:8000
"""

from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from .pipeline_service import run_pipeline

app = FastAPI(title="Oil Spill Attribution API")

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

FRONTEND_DIR = Path(__file__).resolve().parents[1] / "frontend"


@app.get("/api/pipeline")
def get_pipeline():
    """Full pipeline result: spill, origin cone, tracks, gaps, ranked suspects."""
    return run_pipeline()


@app.get("/api/pipeline/refresh")
def refresh_pipeline():
    """Re-runs the pipeline with fresh randomness (new synthetic scenario)."""
    return run_pipeline(force=True)


app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
