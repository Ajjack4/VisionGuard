"""
VisionGuard AI — Main Entry Point
Run with:  python main.py
           python main.py --source 0
           python main.py --source http://192.168.1.42:8080/video
           python main.py --source video.mp4
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

import config


# ── ANSI colours for console ─────────────────────────────────────────────────
_RED = "\033[91m"
_YELLOW = "\033[93m"
_GREEN = "\033[92m"
_CYAN = "\033[96m"
_BOLD = "\033[1m"
_RESET = "\033[0m"

# ── Alert flash state ─────────────────────────────────────────────────────────
_alert_flash_until: float = 0.0


# ── Annotation helpers ────────────────────────────────────────────────────────

def draw_annotations(
    frame: np.ndarray,
    tracks,
    pairs,
    gate_open: bool,
    last_confidence: float,
    fps: float,
    alert_active: bool,
) -> np.ndarray:
    """
    Draw all HUD elements, bounding boxes, and overlays onto *frame*.
    Returns an annotated copy (BGR).
    """
    out = frame.copy()
    h, w = out.shape[:2]
    now = time.time()

    # Collect track IDs that are in a proximate pair
    proximate_ids = set()
    for t1, t2 in pairs:
        proximate_ids.add(t1.track_id)
        proximate_ids.add(t2.track_id)

    # ── Bounding boxes ────────────────────────────────────────────────────────
    for track in tracks:
        if not track.is_confirmed():
            continue
        x1, y1, x2, y2 = track.bbox

        if alert_active:
            colour = (0, 0, 255)     # red
        elif track.track_id in proximate_ids:
            colour = (0, 165, 255)   # orange
        else:
            colour = (0, 200, 0)     # green

        cv2.rectangle(out, (x1, y1), (x2, y2), colour, 2)
        label = f"#{track.track_id}"
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(out, (x1, y1 - lh - 6), (x1 + lw + 4, y1), colour, -1)
        cv2.putText(
            out, label,
            (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
        )

    # ── VIOLENCE DETECTED overlay ─────────────────────────────────────────────
    if alert_active:
        cv2.putText(
            out, "⚠  VIOLENCE DETECTED",
            (w // 2 - 200, h // 2),
            cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 255), 3,
        )

    # ── Top-left HUD ─────────────────────────────────────────────────────────
    gate_str = "OPEN" if gate_open else "CLOSED"
    gate_col = (0, 165, 255) if gate_open else (0, 200, 0)
    hud_lines = [
        (f"FPS: {fps:.1f}", (200, 200, 200)),
        (f"Tracks: {len([t for t in tracks if t.is_confirmed()])}", (200, 200, 200)),
        (f"Gate: {gate_str}", gate_col),
    ]
    for i, (text, col) in enumerate(hud_lines):
        cv2.putText(
            out, text,
            (10, 28 + i * 26),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, col, 2,
        )

    # ── Top-right confidence bar ──────────────────────────────────────────────
    if last_confidence > 0:
        bar_w, bar_h = 160, 18
        bx = w - bar_w - 10
        by = 10
        # Background
        cv2.rectangle(out, (bx, by), (bx + bar_w, by + bar_h), (50, 50, 50), -1)
        # Fill
        fill = int(last_confidence * bar_w)
        bar_col = (0, 0, 255) if last_confidence >= config.CLASSIFIER_THRESHOLD else (0, 200, 255)
        cv2.rectangle(out, (bx, by), (bx + fill, by + bar_h), bar_col, -1)
        cv2.rectangle(out, (bx, by), (bx + bar_w, by + bar_h), (150, 150, 150), 1)
        cv2.putText(
            out, f"Conf: {last_confidence:.1%}",
            (bx, by + bar_h + 16),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1,
        )

    # ── Bottom bar ────────────────────────────────────────────────────────────
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    source_str = str(config.SOURCE)[:40]
    cv2.rectangle(out, (0, h - 28), (w, h), (20, 20, 20), -1)
    cv2.putText(
        out, f"{source_str}  |  {ts}",
        (8, h - 8),
        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 180, 180), 1,
    )

    # ── Red border flash on alert ─────────────────────────────────────────────
    global _alert_flash_until
    if now < _alert_flash_until:
        thickness = 8
        cv2.rectangle(out, (0, 0), (w - 1, h - 1), (0, 0, 255), thickness)

    return out


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run(source_override: Optional[str] = None) -> None:
    # -- apply CLI override ----------------------------------------------------
    if source_override is not None:
        val = source_override.strip()
        config.SOURCE = int(val) if val.isdigit() else val

    print(
        f"\n{_CYAN}{_BOLD}"
        "╔══════════════════════════════════════════╗\n"
        "║       VisionGuard AI  —  Starting        ║\n"
        "╚══════════════════════════════════════════╝"
        f"{_RESET}\n"
    )
    print(config.summary())
    print(
        f"{_YELLOW}{_BOLD}"
        "  ⚠  Classifier running with base Kinetics weights — fine-tune on\n"
        "     violence data for accurate predictions.\n"
        "     See README.md for fine-tuning instructions."
        f"{_RESET}\n"
    )

    # ── Initialise components ──────────────────────────────────────────────────
    from pipeline.stream_reader import StreamReader
    from pipeline.detector import HumanDetector
    from pipeline.tracker import PersonTracker
    from pipeline.gate import ProximityGate
    from pipeline.buffer import TemporalBuffer, PairBuffer
    from pipeline.classifier import ClipClassifier
    from pipeline.alert import AlertEngine
    from api.server import app as api_app, set_alert_engine, set_classifier, update_stream_status

    reader = StreamReader(config.SOURCE, youtube_quality=config.YOUTUBE_QUALITY)
    if not reader.connect():
        print(f"{_RED}Failed to connect to video source. Exiting.{_RESET}")
        sys.exit(1)

    detector = HumanDetector(
        model_path=config.YOLO_MODEL,
        conf_threshold=config.YOLO_CONF,
    )
    tracker = PersonTracker(max_age=config.TRACK_MAX_AGE)
    gate = ProximityGate(alpha=config.GATE_ALPHA, min_people=config.GATE_MIN_PEOPLE)
    pair_buffer = PairBuffer(
        buffer_size=config.BUFFER_SIZE,
        clip_size=config.CLIP_SIZE,
    )
    track_buffer = TemporalBuffer(
        buffer_size=config.BUFFER_SIZE,
        clip_size=config.CLIP_SIZE,
    )
    classifier = ClipClassifier(
        model_path_or_module=config.MODEL_PATH,
        threshold=config.CLASSIFIER_THRESHOLD,
    )
    alert_engine = AlertEngine(
        cooldown_seconds=config.ALERT_COOLDOWN,
        persistence_count=config.PERSISTENCE_COUNT,
        save_clips=config.SAVE_CLIPS,
        save_snapshots=config.SAVE_SNAPSHOTS,
    )

    # Share handles with API server
    set_alert_engine(alert_engine)
    set_classifier(classifier)

    # ── Start FastAPI in background thread ────────────────────────────────────
    import uvicorn

    def _run_api():
        uvicorn.run(
            api_app,
            host="0.0.0.0",
            port=config.API_PORT,
            log_level="warning",
        )

    api_thread = threading.Thread(target=_run_api, daemon=True)
    api_thread.start()
    print(
        f"{_GREEN}[API] Server running at http://localhost:{config.API_PORT}{_RESET}"
    )

    # ── FPS tracking ──────────────────────────────────────────────────────────
    fps_window: deque = deque(maxlen=30)
    last_frame_time = time.time()
    fps_limit_interval = 1.0 / config.FPS_LIMIT

    last_confidence: float = 0.0
    alert_active: bool = False
    recent_bgr_frames: deque = deque(maxlen=5 * config.FPS_LIMIT)  # ~5 s of frames

    global _alert_flash_until

    print(f"{_GREEN}[Main] Pipeline running. Press 'q' to quit.{_RESET}\n")

    try:
        while True:
            loop_start = time.time()

            # ── Read frame ────────────────────────────────────────────────────
            frame = reader.read_frame()
            if frame is None:
                if not reader.is_connected:
                    print("[Main] Stream disconnected. Attempting reconnect…")
                    time.sleep(2.0)
                    reader.connect()
                continue

            recent_bgr_frames.append(frame.copy())

            # ── Detect ───────────────────────────────────────────────────────
            detections = detector.detect(frame)

            # ── Track ────────────────────────────────────────────────────────
            tracks = tracker.update(detections, frame)

            # ── Gate ─────────────────────────────────────────────────────────
            pairs = gate.evaluate(tracks)
            active_pair_keys = {
                (min(t1.track_id, t2.track_id), max(t1.track_id, t2.track_id))
                for t1, t2 in pairs
            }

            # ── Cleanup stale buffers ─────────────────────────────────────────
            active_track_ids = {t.track_id for t in tracks if t.is_confirmed()}
            track_buffer.cleanup_stale(active_track_ids)
            pair_buffer.cleanup_stale(active_pair_keys)

            alert_fired_this_frame = False

            if gate.gate_open:
                for t1, t2 in pairs:
                    merged_bbox = gate.get_merged_bbox(t1.bbox, t2.bbox)

                    # Individual crops for per-track buffers
                    crop1 = detector.extract_crop(
                        frame, t1.bbox, clip_size=config.CLIP_SIZE
                    )
                    crop2 = detector.extract_crop(
                        frame, t2.bbox, clip_size=config.CLIP_SIZE
                    )
                    merged_crop = detector.extract_crop(
                        frame, merged_bbox, padding=0.05, clip_size=config.CLIP_SIZE
                    )

                    track_buffer.update(t1.track_id, crop1)
                    track_buffer.update(t2.track_id, crop2)
                    pair_key = (min(t1.track_id, t2.track_id),
                                max(t1.track_id, t2.track_id))
                    pair_buffer.update(pair_key, merged_crop)

                    if pair_buffer.is_ready(pair_key):
                        clip = pair_buffer.get_clip(pair_key)
                        confidence = classifier.predict(clip)
                        last_confidence = confidence

                        fired = alert_engine.update(
                            camera_id=reader.source_label,
                            track_pair=(t1.track_id, t2.track_id),
                            confidence=confidence,
                            frame=frame,
                            clip_frames=list(recent_bgr_frames),
                        )
                        if fired:
                            alert_fired_this_frame = True
                            _alert_flash_until = time.time() + 2.0

            alert_active = time.time() < _alert_flash_until

            # ── FPS calculation ───────────────────────────────────────────────
            now = time.time()
            elapsed = now - last_frame_time
            fps_window.append(elapsed)
            fps = len(fps_window) / sum(fps_window) if fps_window else 0.0
            last_frame_time = now

            # ── Update API state ──────────────────────────────────────────────
            update_stream_status(
                connected=reader.is_connected,
                fps=round(fps, 1),
                track_count=len([t for t in tracks if t.is_confirmed()]),
                gate_open=gate.gate_open,
            )

            # ── Display ───────────────────────────────────────────────────────
            if config.DISPLAY:
                annotated = draw_annotations(
                    frame=frame,
                    tracks=tracks,
                    pairs=pairs,
                    gate_open=gate.gate_open,
                    last_confidence=last_confidence,
                    fps=fps,
                    alert_active=alert_active,
                )
                cv2.imshow("VisionGuard AI", annotated)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    print("[Main] Quit key pressed.")
                    break

            # ── FPS limiter ───────────────────────────────────────────────────
            loop_elapsed = time.time() - loop_start
            sleep_for = fps_limit_interval - loop_elapsed
            if sleep_for > 0:
                time.sleep(sleep_for)

    except KeyboardInterrupt:
        print("\n[Main] Interrupted by user.")
    finally:
        reader.release()
        cv2.destroyAllWindows()
        print("[Main] Shutdown complete.")


# ── CLI entry point ───────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="VisionGuard AI — Real-time violence detection"
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help=(
            "Video source override. Examples:\n"
            "  0               webcam index\n"
            "  http://IP:8080/video  Android IP Webcam\n"
            "  video.mp4       local file"
        ),
    )
    parser.add_argument(
        "--mobile-setup",
        action="store_true",
        help="Print mobile camera setup instructions and exit.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.mobile_setup:
        from pipeline.stream_reader import get_mobile_camera_instructions
        print(get_mobile_camera_instructions())
        sys.exit(0)

    run(source_override=args.source)
