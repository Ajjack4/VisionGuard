"""
VisionGuard AI — REST API Server (FastAPI)
Runs in a daemon thread alongside the main processing loop.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import config

app = FastAPI(
    title="VisionGuard AI",
    description="Real-time violence detection system API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Shared runtime state (populated by main.py) ──────────────────────────────

_state: Dict[str, Any] = {
    "start_time": time.time(),
    "alert_engine": None,   # AlertEngine instance
    "classifier": None,     # ClipClassifier instance
    "stream_status": {
        "connected": False,
        "source": str(config.SOURCE),
        "fps": 0.0,
        "track_count": 0,
        "gate_open": False,
    },
}


def set_alert_engine(engine) -> None:
    _state["alert_engine"] = engine


def set_classifier(classifier) -> None:
    _state["classifier"] = classifier


def update_stream_status(
    connected: bool,
    fps: float,
    track_count: int,
    gate_open: bool,
) -> None:
    _state["stream_status"].update(
        connected=connected,
        fps=fps,
        track_count=track_count,
        gate_open=gate_open,
    )


# ── Request / Response models ─────────────────────────────────────────────────

class ThresholdUpdate(BaseModel):
    threshold: float


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health", summary="Health check")
def health():
    uptime = time.time() - _state["start_time"]
    return {"status": "ok", "uptime_seconds": round(uptime, 1)}


@app.get("/incidents", summary="Last 20 incidents")
def list_incidents():
    engine = _state["alert_engine"]
    if engine is None:
        return {"incidents": [], "message": "Alert engine not initialised yet."}
    return {"incidents": engine.get_recent_incidents(20)}


@app.get("/incidents/{incident_id}", summary="Single incident by ID")
def get_incident(incident_id: int):
    engine = _state["alert_engine"]
    if engine is None:
        raise HTTPException(status_code=503, detail="Alert engine not ready.")
    incident = engine.get_incident(incident_id)
    if incident is None:
        raise HTTPException(status_code=404, detail=f"Incident {incident_id} not found.")
    return incident


@app.patch("/incidents/{incident_id}/false-alarm", summary="Mark incident as false alarm")
def mark_false_alarm(incident_id: int):
    engine = _state["alert_engine"]
    if engine is None:
        raise HTTPException(status_code=503, detail="Alert engine not ready.")
    ok = engine.log_false_alarm(incident_id)
    if not ok:
        raise HTTPException(status_code=404, detail=f"Incident {incident_id} not found.")
    return {"message": f"Incident {incident_id} marked as false alarm."}


@app.get("/config", summary="Current configuration (read-only)")
def get_config():
    return {
        "source": str(config.SOURCE),
        "yolo_model": config.YOLO_MODEL,
        "yolo_conf": config.YOLO_CONF,
        "gate_alpha": config.GATE_ALPHA,
        "gate_min_people": config.GATE_MIN_PEOPLE,
        "buffer_size": config.BUFFER_SIZE,
        "clip_size": config.CLIP_SIZE,
        "classifier_threshold": config.CLASSIFIER_THRESHOLD,
        "persistence_count": config.PERSISTENCE_COUNT,
        "alert_cooldown": config.ALERT_COOLDOWN,
        "save_clips": config.SAVE_CLIPS,
        "save_snapshots": config.SAVE_SNAPSHOTS,
        "fps_limit": config.FPS_LIMIT,
        "api_port": config.API_PORT,
        "display": config.DISPLAY,
        "model_path": config.MODEL_PATH,
    }


@app.post("/config/threshold", summary="Update classifier threshold at runtime")
def update_threshold(body: ThresholdUpdate):
    if not 0.0 < body.threshold < 1.0:
        raise HTTPException(
            status_code=422,
            detail="Threshold must be between 0 and 1 (exclusive).",
        )
    config.CLASSIFIER_THRESHOLD = body.threshold
    clf = _state["classifier"]
    if clf is not None:
        clf.threshold = body.threshold
    return {"message": f"Threshold updated to {body.threshold}"}


@app.get("/stream/status", summary="Camera connection and pipeline status")
def stream_status():
    return _state["stream_status"]
