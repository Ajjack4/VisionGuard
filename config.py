"""
VisionGuard AI — Central Configuration
All tuneable parameters loaded from .env with sensible defaults.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from the project root (same directory as this file)
_ROOT = Path(__file__).parent
load_dotenv(_ROOT / ".env")


def _bool(val: str) -> bool:
    return val.strip().lower() in ("1", "true", "yes")


def _int_list(val: str):
    return [int(x.strip()) for x in val.split(",") if x.strip()]


# ── Paths ───────────────────────────────────────────────────────────────────
ROOT_DIR: Path = _ROOT
DATA_DIR: Path = _ROOT / "data"
ALERTS_DIR: Path = DATA_DIR / "alerts"
DB_PATH: Path = DATA_DIR / "visionguard.db"
ASSETS_DIR: Path = _ROOT / "assets"
ALERT_SOUND: Path = ASSETS_DIR / "alert.wav"

# Ensure directories exist at import time
ALERTS_DIR.mkdir(parents=True, exist_ok=True)
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

# ── Video Source ────────────────────────────────────────────────────────────
# Three named source params — set SOURCE_TYPE to activate one:
#   SOURCE_TYPE=webcam   → uses WEBCAM_INDEX (default: 0)
#   SOURCE_TYPE=file     → uses VIDEO_FILE path
#   SOURCE_TYPE=youtube  → uses YOUTUBE_URL (resolved via yt-dlp at runtime)
#
# Legacy single-value fallback: set SOURCE directly (still works).

SOURCE_TYPE: str = os.getenv("SOURCE_TYPE", "").strip().lower()  # webcam|file|youtube|""
WEBCAM_INDEX: int = int(os.getenv("WEBCAM_INDEX", "0"))
VIDEO_FILE: str = os.getenv("VIDEO_FILE", "").strip()
YOUTUBE_URL: str = os.getenv("YOUTUBE_URL", "").strip()
YOUTUBE_QUALITY: str = os.getenv("YOUTUBE_QUALITY", "best[height<=480]").strip()

# Resolve the effective SOURCE used by the rest of the pipeline
if SOURCE_TYPE == "webcam":
    SOURCE: int | str = WEBCAM_INDEX
elif SOURCE_TYPE == "file":
    if not VIDEO_FILE:
        raise ValueError("SOURCE_TYPE=file requires VIDEO_FILE to be set in .env")
    SOURCE = VIDEO_FILE
elif SOURCE_TYPE == "youtube":
    if not YOUTUBE_URL:
        raise ValueError("SOURCE_TYPE=youtube requires YOUTUBE_URL to be set in .env")
    SOURCE = YOUTUBE_URL          # StreamReader will resolve this to a direct URL
else:
    # Legacy / raw SOURCE override (backward-compatible)
    _raw_source = os.getenv("SOURCE", "0")
    SOURCE = int(_raw_source) if _raw_source.strip().isdigit() else _raw_source

# ── YOLO ────────────────────────────────────────────────────────────────────
YOLO_MODEL: str = os.getenv("YOLO_MODEL", "yolov8n.pt")
YOLO_CONF: float = float(os.getenv("YOLO_CONF", "0.45"))
YOLO_CLASSES: list = _int_list(os.getenv("YOLO_CLASSES", "0"))  # 0 = person

# ── Tracking ────────────────────────────────────────────────────────────────
TRACK_MAX_AGE: int = int(os.getenv("TRACK_MAX_AGE", "30"))

# ── Proximity Gate ──────────────────────────────────────────────────────────
GATE_ALPHA: float = float(os.getenv("GATE_ALPHA", "2.5"))
GATE_MIN_PEOPLE: int = int(os.getenv("GATE_MIN_PEOPLE", "2"))

# ── Temporal Buffer ─────────────────────────────────────────────────────────
BUFFER_SIZE: int = int(os.getenv("BUFFER_SIZE", "16"))
CLIP_SIZE: int = int(os.getenv("CLIP_SIZE", "112"))

# ── Classifier ──────────────────────────────────────────────────────────────
CLASSIFIER_THRESHOLD: float = float(os.getenv("CLASSIFIER_THRESHOLD", "0.65"))
PERSISTENCE_COUNT: int = int(os.getenv("PERSISTENCE_COUNT", "2"))
MODEL_PATH: str | None = os.getenv("MODEL_PATH", None) or None
# MODEL_TYPE controls which classifier backbone is used:
#   kinetics_heuristic  — zero-shot, works immediately (default for PoC)
#   r3d18               — binary head, requires fine-tuning to be accurate
#   x3d_xs              — faster Facebook X3D model via torch.hub (PoC)
#   slowfast_r50        — SlowFast R50 via torch.hub (PoC, more accurate)
#   slowfast_violence   — fine-tuned SlowFast R50 from train_slowfast.ipynb
#                         requires MODEL_PATH=models/slowfast_violence.pt
#                         and BUFFER_SIZE=32
MODEL_TYPE: str = os.getenv("MODEL_TYPE", "kinetics_heuristic").strip().lower()
# Score amplifier for kinetics_heuristic mode (higher = more sensitive)
KINETICS_SCORE_SCALE: float = float(os.getenv("KINETICS_SCORE_SCALE", "6.0"))

# ── Alerts ──────────────────────────────────────────────────────────────────
ALERT_COOLDOWN: int = int(os.getenv("ALERT_COOLDOWN", "60"))
SAVE_CLIPS: bool = _bool(os.getenv("SAVE_CLIPS", "true"))
SAVE_SNAPSHOTS: bool = _bool(os.getenv("SAVE_SNAPSHOTS", "true"))

# ── Display ─────────────────────────────────────────────────────────────────
DISPLAY: bool = _bool(os.getenv("DISPLAY", "true"))

# ── API ─────────────────────────────────────────────────────────────────────
API_PORT: int = int(os.getenv("API_PORT", "8000"))

# ── Device ──────────────────────────────────────────────────────────────────
# Set DEVICE=cuda to force GPU, DEVICE=cpu to force CPU.
# Leave blank for auto-detection: cuda > mps (Apple Silicon) > cpu.
def _auto_device() -> str:
    import torch
    _env = os.getenv("DEVICE", "").strip().lower()
    if _env:
        return _env
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        print(f"[Config] GPU detected: {name} — using CUDA")
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("[Config] Apple Silicon detected — using MPS")
        return "mps"
    print("[Config] No GPU found — running on CPU (expect lower FPS)")
    return "cpu"

DEVICE: str = _auto_device()

# ── Performance ─────────────────────────────────────────────────────────────
FPS_LIMIT: int = int(os.getenv("FPS_LIMIT", "15"))
# FRAME_SKIP: run detection+classification only on every Nth frame.
# Frames in between are displayed using the last known annotations.
# Set to 1 to process every frame, 2 to process every other, etc.
FRAME_SKIP: int = max(1, int(os.getenv("FRAME_SKIP", "2")))

# ── Debug ───────────────────────────────────────────────────────────────────
# Show a "Classifier View" window with the crop contact sheet the model sees.
DEBUG_CROPS: bool = _bool(os.getenv("DEBUG_CROPS", "false"))


def summary() -> str:
    """Return a human-readable config summary for startup logs."""
    _type_label = SOURCE_TYPE if SOURCE_TYPE else "auto"
    return (
        f"  Source         : {SOURCE}  [{_type_label}]\n"
        f"  YOLO model     : {YOLO_MODEL}  conf={YOLO_CONF}\n"
        f"  Buffer         : {BUFFER_SIZE} frames @ {CLIP_SIZE}x{CLIP_SIZE}px\n"
        f"  Gate           : alpha={GATE_ALPHA}  min_people={GATE_MIN_PEOPLE}\n"
        f"  Classifier     : threshold={CLASSIFIER_THRESHOLD}  "
        f"persistence={PERSISTENCE_COUNT}\n"
        f"  Alert cooldown : {ALERT_COOLDOWN}s\n"
        f"  FPS limit      : {FPS_LIMIT}\n"
        f"  API port       : {API_PORT}\n"
        f"  Display        : {DISPLAY}\n"
        f"  Device         : {DEVICE}\n"
        f"  Model type     : {MODEL_TYPE}\n"
        f"  Model path     : {MODEL_PATH or '(default backbone)'}\n"
        f"  Frame skip     : every {FRAME_SKIP} frame(s)\n"
    )
