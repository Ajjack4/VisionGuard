"""
VisionGuard AI — Streamlit Dashboard
Runs the full detection pipeline in a background thread and displays:
  • Top    — main camera feed with per-person bounding boxes + unique track IDs
  • Bottom — one panel per proximate pair showing the exact merged crop region
             fed to the classifier, the temporal contact sheet, and live confidence.

Run with:
    streamlit run streamlit_app.py
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import cv2
import numpy as np
import streamlit as st

import config

# ── ImageNet de-normalisation ─────────────────────────────────────────────────
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _denorm(crop_hwc: np.ndarray) -> np.ndarray:
    """Float32 ImageNet-normalised [H,W,C] → BGR uint8."""
    rgb = (crop_hwc * _STD + _MEAN).clip(0.0, 1.0)
    return cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)


def _make_contact(frames: list, buffer_size: int, cols: int = 8) -> np.ndarray:
    """Tile buffer frames into a contact sheet; empty slots shown as dark grey."""
    cell = config.CLIP_SIZE
    rows = (buffer_size + cols - 1) // cols
    sheet = np.zeros((rows * cell, cols * cell, 3), dtype=np.uint8)
    for idx in range(buffer_size):
        r, c = divmod(idx, cols)
        y0, x0 = r * cell, c * cell
        if idx < len(frames):
            sheet[y0:y0 + cell, x0:x0 + cell] = _denorm(frames[idx])
        else:
            sheet[y0:y0 + cell, x0:x0 + cell] = 28
    return sheet


def _bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ── Shared state (pipeline thread ↔ Streamlit UI) ────────────────────────────

@dataclass
class PairPanel:
    id1: int
    id2: int
    confidence: Optional[float]    # None until buffer is full
    n_frames: int
    buffer_size: int
    crop_rgb: Optional[np.ndarray]     # merged region crop — current frame (RGB)
    contact_rgb: Optional[np.ndarray]  # contact sheet of temporal buffer (RGB)


@dataclass
class AppState:
    lock: threading.Lock = field(default_factory=threading.Lock)
    main_frame_rgb: Optional[np.ndarray] = None
    panels: List[PairPanel] = field(default_factory=list)
    fps: float = 0.0
    track_count: int = 0
    gate_open: bool = False
    alert_active: bool = False
    status: str = "starting…"


# ── Draw the main annotated frame (no cv2.imshow) ─────────────────────────────

# Distinct box colours for up to 20 simultaneous tracks
_TRACK_COLOURS = [
    (0, 200, 0), (0, 165, 255), (255, 128, 0), (0, 255, 255),
    (200, 0, 200), (0, 200, 200), (128, 255, 0), (255, 0, 128),
    (0, 128, 255), (255, 200, 0), (64, 255, 64), (255, 64, 64),
    (64, 64, 255), (200, 255, 64), (64, 200, 255), (255, 64, 200),
    (200, 64, 255), (64, 255, 200), (128, 128, 255), (255, 128, 128),
]


def _track_colour(track_id, alert_active: bool, proximate_ids: set) -> tuple:
    if alert_active and track_id in proximate_ids:
        return (0, 0, 255)           # red — alert
    if track_id in proximate_ids:
        return (0, 128, 255)         # orange — being classified
    # abs(hash()) works for both int and string track IDs (DeepSORT can return either)
    return _TRACK_COLOURS[abs(hash(str(track_id))) % len(_TRACK_COLOURS)]


def _draw_main(
    frame: np.ndarray,
    tracks,
    pairs,
    merged_bboxes: list,
    alert_active: bool,
    fps: float,
) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]

    proximate_ids = {t.track_id for pair in pairs for t in pair}

    for track in tracks:
        if not track.is_confirmed():
            continue
        x1, y1, x2, y2 = track.bbox
        col = _track_colour(track.track_id, alert_active, proximate_ids)
        cv2.rectangle(out, (x1, y1), (x2, y2), col, 2)
        label = f"#{track.track_id}"
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(out, (x1, y1 - lh - 6), (x1 + lw + 4, y1), col, -1)
        cv2.putText(out, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    for mbbox in merged_bboxes:
        mx1, my1, mx2, my2 = mbbox
        cv2.rectangle(out, (mx1, my1), (mx2, my2), (255, 0, 255), 2)
        cv2.putText(out, "CLASSIFYING", (mx1 + 4, my1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 0, 255), 1)

    if alert_active:
        cv2.putText(out, "VIOLENCE DETECTED",
                    (w // 2 - 180, h // 2),
                    cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 255), 3)
        cv2.rectangle(out, (0, 0), (w - 1, h - 1), (0, 0, 255), 8)

    # HUD
    gate_str = "OPEN" if len(merged_bboxes) > 0 else "CLOSED"
    gate_col = (0, 165, 255) if len(merged_bboxes) > 0 else (0, 200, 0)
    for i, (txt, col) in enumerate([
        (f"FPS: {fps:.1f}", (200, 200, 200)),
        (f"Tracks: {len([t for t in tracks if t.is_confirmed()])}", (200, 200, 200)),
        (f"Gate: {gate_str}", gate_col),
    ]):
        cv2.putText(out, txt, (10, 28 + i * 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, col, 2)

    return out


# ── Pipeline background thread ────────────────────────────────────────────────

def _run_pipeline(state: AppState) -> None:
    from pipeline.stream_reader import StreamReader
    from pipeline.detector import HumanDetector
    from pipeline.tracker import PersonTracker
    from pipeline.gate import ProximityGate
    from pipeline.buffer import PairBuffer, TemporalBuffer
    from pipeline.classifier import ClipClassifier
    from pipeline.alert import AlertEngine

    state.status = "connecting…"

    reader = StreamReader(config.SOURCE, youtube_quality=config.YOUTUBE_QUALITY)
    if not reader.connect():
        state.status = "ERROR: could not connect to video source"
        return

    state.status = "loading models…"
    detector   = HumanDetector(model_path=config.YOLO_MODEL,
                               conf_threshold=config.YOLO_CONF,
                               device=config.DEVICE)
    tracker    = PersonTracker(max_age=config.TRACK_MAX_AGE)
    gate       = ProximityGate(alpha=config.GATE_ALPHA,
                               min_people=config.GATE_MIN_PEOPLE)
    pair_buf   = PairBuffer(buffer_size=config.BUFFER_SIZE,
                            clip_size=config.CLIP_SIZE)
    track_buf  = TemporalBuffer(buffer_size=config.BUFFER_SIZE,
                                clip_size=config.CLIP_SIZE)
    classifier = ClipClassifier(model_path_or_module=config.MODEL_PATH,
                                device=config.DEVICE,
                                threshold=config.CLASSIFIER_THRESHOLD)
    alerts     = AlertEngine(cooldown_seconds=config.ALERT_COOLDOWN,
                             persistence_count=config.PERSISTENCE_COUNT,
                             save_clips=config.SAVE_CLIPS,
                             save_snapshots=config.SAVE_SNAPSHOTS)

    fps_window: deque     = deque(maxlen=30)
    recent_bgr: deque     = deque(maxlen=5 * config.FPS_LIMIT)
    confidence_map: Dict  = {}
    frame_count           = 0
    last_frame_time       = time.time()
    alert_flash_until     = 0.0
    tracks                = []
    pairs                 = []
    state.status          = "running"

    while True:
        frame = reader.read_frame()
        if frame is None:
            if not reader.is_connected:
                state.status = "reconnecting…"
                time.sleep(2.0)
                reader.connect()
                state.status = "running"
            continue

        recent_bgr.append(frame.copy())
        now = time.time()
        fps_window.append(now - last_frame_time)
        fps = len(fps_window) / sum(fps_window) if fps_window else 0.0
        last_frame_time = now
        alert_active = now < alert_flash_until

        frame_count += 1
        if frame_count % config.FRAME_SKIP != 0:
            continue

        # ── Detect + Track + Gate ─────────────────────────────────────────────
        detections = detector.detect(frame)
        tracks     = tracker.update(detections, frame)
        pairs      = gate.evaluate(tracks)

        active_pair_keys = {
            (min(t1.track_id, t2.track_id), max(t1.track_id, t2.track_id))
            for t1, t2 in pairs
        }
        track_buf.cleanup_stale({t.track_id for t in tracks if t.is_confirmed()})
        pair_buf.cleanup_stale(active_pair_keys)
        for k in [k for k in confidence_map if k not in active_pair_keys]:
            del confidence_map[k]

        # ── Per-pair classification ───────────────────────────────────────────
        new_panels: List[PairPanel] = []
        merged_bboxes: List[list]   = []

        if gate.gate_open:
            for t1, t2 in pairs:
                merged_bbox = gate.get_merged_bbox(t1.bbox, t2.bbox)
                merged_bboxes.append(merged_bbox)

                crop1       = detector.extract_crop(frame, t1.bbox,
                                                    clip_size=config.CLIP_SIZE)
                crop2       = detector.extract_crop(frame, t2.bbox,
                                                    clip_size=config.CLIP_SIZE)
                merged_crop = detector.extract_crop(frame, merged_bbox,
                                                    padding=0.05,
                                                    clip_size=config.CLIP_SIZE)

                track_buf.update(t1.track_id, crop1)
                track_buf.update(t2.track_id, crop2)
                pair_key = (min(t1.track_id, t2.track_id),
                            max(t1.track_id, t2.track_id))
                pair_buf.update(pair_key, merged_crop)

                if pair_buf.is_ready(pair_key):
                    clip       = pair_buf.get_clip(pair_key)
                    confidence = classifier.predict(clip)
                    confidence_map[pair_key] = confidence

                    fired = alerts.update(
                        camera_id=reader.source_label,
                        track_pair=(t1.track_id, t2.track_id),
                        confidence=confidence,
                        frame=frame,
                        clip_frames=list(recent_bgr),
                    )
                    if fired:
                        alert_flash_until = time.time() + 2.0
                        alert_active = True

                buf_frames = list(pair_buf._buffers.get(pair_key, []))
                crop_bgr   = _denorm(merged_crop)
                contact    = _make_contact(buf_frames, config.BUFFER_SIZE)

                new_panels.append(PairPanel(
                    id1=t1.track_id,
                    id2=t2.track_id,
                    confidence=confidence_map.get(pair_key),
                    n_frames=len(buf_frames),
                    buffer_size=config.BUFFER_SIZE,
                    crop_rgb=_bgr_to_rgb(crop_bgr),
                    contact_rgb=_bgr_to_rgb(contact),
                ))

        # ── Push annotated frame to shared state ──────────────────────────────
        annotated = _draw_main(frame, tracks, pairs, merged_bboxes, alert_active, fps)

        with state.lock:
            state.main_frame_rgb = _bgr_to_rgb(annotated)
            state.panels         = new_panels
            state.fps            = fps
            state.track_count    = len([t for t in tracks if t.is_confirmed()])
            state.gate_open      = gate.gate_open
            state.alert_active   = alert_active

    reader.release()


# ── Streamlit render ──────────────────────────────────────────────────────────

def _render(state: AppState) -> None:
    with state.lock:
        frame     = state.main_frame_rgb
        panels    = list(state.panels)
        fps       = state.fps
        n_tracks  = state.track_count
        gate_open = state.gate_open
        alert     = state.alert_active
        status    = state.status

    # ── Status / metrics row ──────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("FPS", f"{fps:.1f}")
    c2.metric("People tracked", n_tracks)
    c3.metric("Proximity gate", "OPEN 🟠" if gate_open else "CLOSED 🟢")
    c4.metric("Alert", "⚠ ACTIVE 🔴" if alert else "Clear ✅")

    if status != "running":
        st.info(f"Pipeline: {status}")

    # ── Main detection view ───────────────────────────────────────────────────
    if frame is not None:
        st.image(frame, caption="Detection view — each track ID has a unique colour",
                 width='stretch')
    else:
        st.caption("Waiting for first frame…")

    # ── Per-pair classifier panels ────────────────────────────────────────────
    st.divider()
    if panels:
        st.subheader(f"Active classifiers — {len(panels)} pair(s)")
        cols = st.columns(len(panels))
        for col, p in zip(cols, panels):
            with col:
                conf = p.confidence
                is_alert = conf is not None and conf >= config.CLASSIFIER_THRESHOLD
                badge = "🔴" if is_alert else ("🟡" if conf is not None else "⏳")

                st.markdown(f"### {badge} Pair #{p.id1} + #{p.id2}")

                # Merged crop — what the model actually receives each frame
                if p.crop_rgb is not None:
                    st.image(p.crop_rgb,
                             caption="Merged crop fed to classifier (current frame)",
                             width='stretch')

                # Confidence / buffer-fill bar
                if conf is not None:
                    colour_label = "🔴 VIOLENT" if is_alert else "🟡 non-violent"
                    st.progress(float(min(1.0, conf)),
                                text=f"P(violent) = {conf:.1%}  →  {colour_label}")
                    st.caption(f"Threshold: {config.CLASSIFIER_THRESHOLD:.0%}")
                else:
                    fill = p.n_frames / p.buffer_size
                    st.progress(float(fill),
                                text=f"Buffering… {p.n_frames}/{p.buffer_size} frames")

                # Temporal contact sheet — all frames the next classification will use
                if p.contact_rgb is not None:
                    st.image(p.contact_rgb,
                             caption=(
                                 f"Temporal buffer — {p.n_frames}/{p.buffer_size} frames "
                                 f"(reads left→right, top→bottom)"
                             ),
                             width='stretch')
    else:
        st.caption(
            "No proximate pairs detected — classifier is idle. "
            "The gate opens when two tracked people come close together."
        )


# ── Pipeline singleton ────────────────────────────────────────────────────────
# @st.cache_resource runs exactly once per server process and shares the result
# across every session/rerun — the correct primitive for a background pipeline.

@st.cache_resource
def _get_pipeline_state() -> AppState:
    state = AppState()
    t = threading.Thread(target=_run_pipeline, args=(state,), daemon=True)
    t.start()
    return state


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(
        page_title="VisionGuard AI",
        page_icon="👁",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.title("👁 VisionGuard AI")

    state = _get_pipeline_state()

    # Live update loop — replaces the placeholder content at ~20 fps
    placeholder = st.empty()
    while True:
        with placeholder.container():
            _render(state)
        time.sleep(0.05)


if __name__ == "__main__":
    main()
