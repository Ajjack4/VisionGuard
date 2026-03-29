"""
VisionGuard AI — Streamlit Dashboard
A clean, easy-to-use surveillance dashboard.

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


# STYLING


def inject_css():
    st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    .stApp { background-color: #0f1117; }

    [data-testid="stSidebar"] {
        background-color: #1a1d27 !important;
        border-right: 1px solid #2d3047;
    }
    [data-testid="stSidebar"] * { color: #c8ccd8 !important; }

    [data-testid="metric-container"] {
        background: #1a1d27;
        border: 1px solid #2d3047;
        border-radius: 12px;
        padding: 18px 20px !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 12px !important;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: #7a7f9a !important;
    }
    [data-testid="stMetricValue"] {
        font-size: 30px !important;
        font-weight: 700 !important;
        color: #e8eaf2 !important;
    }

    .alert-box {
        background: #2a1a1a;
        border: 2px solid #ff4d4d;
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 20px;
    }
    .alert-title { font-size: 16px; font-weight: 700; color: #ff6060; margin: 0; }
    .alert-body  { font-size: 13px; color: #cc8888; margin: 4px 0 0; }

    .section-title {
        font-size: 13px;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #5a5f7a;
        margin: 24px 0 12px;
        padding-bottom: 8px;
        border-bottom: 1px solid #2d3047;
    }

    .info-box {
        background: #0d1a2a;
        border-left: 3px solid #60a0ff;
        border-radius: 0 8px 8px 0;
        padding: 12px 16px;
        font-size: 13px;
        color: #90b8e8;
        margin: 10px 0;
    }
    .warn-box {
        background: #2a1f0d;
        border-left: 3px solid #ffb340;
        border-radius: 0 8px 8px 0;
        padding: 12px 16px;
        font-size: 13px;
        color: #d4a050;
        margin: 10px 0;
    }

    .logo-text {
        font-size: 22px;
        font-weight: 800;
        color: #e8eaf2 !important;
        letter-spacing: -0.5px;
    }
    .logo-sub {
        font-size: 11px;
        color: #5a5f7a !important;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        margin-top: -2px;
    }

    .stProgress > div > div > div > div { background: #00e5a0; }

    .styled-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 13px;
        color: #c8ccd8;
    }
    .styled-table th {
        text-align: left;
        padding: 10px 14px;
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #5a5f7a;
        border-bottom: 1px solid #2d3047;
    }
    .styled-table td {
        padding: 12px 14px;
        border-bottom: 1px solid #1f2235;
        vertical-align: middle;
    }
    .styled-table tr:last-child td { border-bottom: none; }
    .styled-table tr:hover td { background: #1f2235; }

    hr { border-color: #2d3047 !important; }
    </style>
    """, unsafe_allow_html=True)



# IMAGE HELPERS


_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _denorm(crop_hwc: np.ndarray) -> np.ndarray:
    rgb = (crop_hwc * _STD + _MEAN).clip(0.0, 1.0)
    return cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)


def _make_contact(frames: list, buffer_size: int, cols: int = 8) -> np.ndarray:
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



# SHARED PIPELINE STATE


@dataclass
class ScenePanel:
    track_ids: list
    confidence: Optional[float]
    n_frames: int
    buffer_size: int
    crop_rgb: Optional[np.ndarray]
    contact_rgb: Optional[np.ndarray]


@dataclass
class AppState:
    lock: threading.Lock = field(default_factory=threading.Lock)
    main_frame_rgb: Optional[np.ndarray] = None
    panels: List[ScenePanel] = field(default_factory=list)
    fps: float = 0.0
    track_count: int = 0
    gate_open: bool = False
    alert_active: bool = False
    status: str = "starting..."
    recent_incidents: list = field(default_factory=list)



# FRAME DRAWING


_TRACK_COLOURS = [
    (0, 200, 0), (0, 165, 255), (255, 128, 0), (0, 255, 255),
    (200, 0, 200), (0, 200, 200), (128, 255, 0), (255, 0, 128),
    (0, 128, 255), (255, 200, 0), (64, 255, 64), (255, 64, 64),
    (64, 64, 255), (200, 255, 64), (64, 200, 255), (255, 64, 200),
    (200, 64, 255), (64, 255, 200), (128, 128, 255), (255, 128, 128),
]


def _track_colour(track_id, alert_active: bool, proximate_ids: set) -> tuple:
    if alert_active and track_id in proximate_ids:
        return (0, 0, 255)
    if track_id in proximate_ids:
        return (0, 128, 255)
    return _TRACK_COLOURS[abs(hash(str(track_id))) % len(_TRACK_COLOURS)]


def _draw_main(frame, tracks, groups, merged_bboxes, alert_active, fps):
    out = frame.copy()
    h, w = out.shape[:2]
    proximate_ids = {t.track_id for group in groups for t in group}

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

    gate_str = "GATE: OPEN" if merged_bboxes else "GATE: CLOSED"
    gate_col = (0, 165, 255) if merged_bboxes else (0, 200, 0)
    for i, (txt, col) in enumerate([
        (f"FPS: {fps:.1f}", (200, 200, 200)),
        (f"People: {len([t for t in tracks if t.is_confirmed()])}", (200, 200, 200)),
        (gate_str, gate_col),
    ]):
        cv2.putText(out, txt, (10, 28 + i * 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, col, 2)

    return out



# PIPELINE BACKGROUND THREAD


def _run_pipeline(state: AppState) -> None:
    from pipeline.stream_reader import StreamReader
    from pipeline.detector import HumanDetector
    from pipeline.tracker import PersonTracker
    from pipeline.gate import ProximityGate
    from pipeline.buffer import PairBuffer, TemporalBuffer
    from pipeline.classifier import ClipClassifier
    from pipeline.alert import AlertEngine

    state.status = "connecting..."
    reader = StreamReader(config.SOURCE, youtube_quality=config.YOUTUBE_QUALITY)
    if not reader.connect():
        state.status = "ERROR: could not connect to video source"
        return

    state.status = "loading models..."
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

    fps_window: deque    = deque(maxlen=30)
    recent_bgr: deque    = deque(maxlen=5 * config.FPS_LIMIT)
    confidence_map: Dict = {}
    frame_count          = 0
    last_frame_time      = time.time()
    alert_flash_until    = 0.0
    tracks               = []
    state.status         = "running"

    while True:
        frame = reader.read_frame()
        if frame is None:
            if not reader.is_connected:
                state.status = "reconnecting..."
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

        detections = detector.detect(frame)
        tracks     = tracker.update(detections, frame)

        groups = gate.evaluate_groups(tracks)
        active_group_keys = {
            tuple(sorted(t.track_id for t in g)) for g in groups
        }
        track_buf.cleanup_stale({t.track_id for t in tracks if t.is_confirmed()})
        pair_buf.cleanup_stale(active_group_keys)
        for k in [k for k in confidence_map if k not in active_group_keys]:
            del confidence_map[k]

        new_panels: List[ScenePanel] = []
        merged_bboxes: List[list]    = []

        if gate.gate_open:
            for group in groups:
                group_bbox = gate.get_group_bbox(group)
                merged_bboxes.append(group_bbox)

                for t in group:
                    crop = detector.extract_crop(frame, t.bbox,
                                                 clip_size=config.CLIP_SIZE)
                    track_buf.update(t.track_id, crop)

                group_crop = detector.extract_crop(frame, group_bbox,
                                                   padding=0.05,
                                                   clip_size=config.CLIP_SIZE)
                group_key = tuple(sorted(t.track_id for t in group))
                pair_buf.update(group_key, group_crop)

                if pair_buf.is_ready(group_key):
                    clip       = pair_buf.get_clip(group_key)
                    confidence = classifier.predict(clip)
                    confidence_map[group_key] = confidence

                    fired = alerts.update(
                        camera_id=reader.source_label,
                        track_pair=group_key,
                        confidence=confidence,
                        frame=frame,
                        clip_frames=list(recent_bgr),
                    )
                    if fired:
                        alert_flash_until = time.time() + 2.0
                        alert_active = True
                        with state.lock:
                            state.recent_incidents.insert(0, {
                                "time": time.strftime("%H:%M:%S"),
                                "tracks": " + ".join(f"#{i}" for i in group_key),
                                "confidence": confidence,
                            })
                            state.recent_incidents = state.recent_incidents[:10]

                buf_frames = list(pair_buf._buffers.get(group_key, []))
                crop_bgr   = _denorm(group_crop)
                contact    = _make_contact(buf_frames, config.BUFFER_SIZE)

                new_panels.append(ScenePanel(
                    track_ids=[t.track_id for t in group],
                    confidence=confidence_map.get(group_key),
                    n_frames=len(buf_frames),
                    buffer_size=config.BUFFER_SIZE,
                    crop_rgb=_bgr_to_rgb(crop_bgr),
                    contact_rgb=_bgr_to_rgb(contact),
                ))

        annotated = _draw_main(frame, tracks, groups, merged_bboxes, alert_active, fps)

        with state.lock:
            state.main_frame_rgb = _bgr_to_rgb(annotated)
            state.panels         = new_panels
            state.fps            = fps
            state.track_count    = len([t for t in tracks if t.is_confirmed()])
            state.gate_open      = gate.gate_open
            state.alert_active   = alert_active

    reader.release()


@st.cache_resource
def _get_pipeline_state() -> AppState:
    state = AppState()
    t = threading.Thread(target=_run_pipeline, args=(state,), daemon=True)
    t.start()
    return state



# PAGE: LIVE MONITOR


def page_live_monitor(state: AppState):
    with state.lock:
        frame     = state.main_frame_rgb
        panels    = list(state.panels)
        fps       = state.fps
        n_tracks  = state.track_count
        gate_open = state.gate_open
        alert     = state.alert_active
        status    = state.status

    # Alert banner
    if alert:
        st.markdown("""
        <div class="alert-box">
            <span style="font-size:28px">🚨</span>
            <p class="alert-title">Violence Detected!</p>
            <p class="alert-body">The system flagged a potential incident. Review the feed below.</p>
        </div>
        """, unsafe_allow_html=True)

    if status != "running":
        st.info(f"Pipeline: **{status}**")

    # Key metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Live FPS", f"{fps:.1f}")
    c2.metric("People Detected", n_tracks)
    c3.metric("AI Classifier", "Analysing..." if gate_open else "Idle")
    c4.metric("Alert Status", "ALERT" if alert else "All Clear")

    st.divider()

    # Live feed
    st.markdown('<div class="section-title">Camera Feed</div>', unsafe_allow_html=True)
    if frame is not None:
        st.image(frame, use_container_width=True,
                 caption="Each coloured box is a tracked person. Purple box = being analysed by AI.")
    else:
        st.markdown("""
        <div style="background:#1a1d27;border:1px dashed #2d3047;border-radius:12px;
                    padding:60px;text-align:center;color:#5a5f7a;">
            <div style="font-size:40px;margin-bottom:12px">📷</div>
            <div style="font-size:16px;font-weight:600">Waiting for camera...</div>
            <div style="font-size:13px;margin-top:6px">
                Go to <b>Settings</b> to configure your camera source
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Classifier panels
    if panels:
        st.divider()
        st.markdown('<div class="section-title">What the AI is Analysing Right Now</div>',
                    unsafe_allow_html=True)

        cols = st.columns(min(len(panels), 3))
        for col, p in zip(cols, panels):
            with col:
                conf = p.confidence
                is_alert = conf is not None and conf >= config.CLASSIFIER_THRESHOLD
                ids_str = " + ".join(f"#{i}" for i in p.track_ids)

                if is_alert:
                    color, icon = "#ff6060", "🔴"
                elif conf is not None:
                    color, icon = "#ffb340", "🟡"
                else:
                    color, icon = "#60a0ff", "⏳"

                st.markdown(f"""
                <div style="background:#1a1d27;border:1px solid #2d3047;border-radius:12px;
                            padding:14px 16px;margin-bottom:10px;">
                    <div style="font-size:14px;font-weight:700;color:{color}">
                        {icon} People {ids_str}
                    </div>
                    <div style="font-size:12px;color:#7a7f9a;margin-top:2px">
                        {len(p.track_ids)} people in close proximity
                    </div>
                </div>
                """, unsafe_allow_html=True)

                if p.crop_rgb is not None:
                    st.image(p.crop_rgb, use_container_width=True,
                             caption="Clip being analysed")

                if conf is not None:
                    label = "Violent" if is_alert else "Non-violent"
                    st.progress(float(min(1.0, conf)),
                                text=f"Confidence: {conf:.0%} — {label}")
                    st.caption(f"Alert fires above {config.CLASSIFIER_THRESHOLD:.0%}")
                else:
                    st.progress(p.n_frames / p.buffer_size,
                                text=f"Collecting frames: {p.n_frames}/{p.buffer_size}")
                    st.caption("Needs more frames before making a decision")

                if p.contact_rgb is not None:
                    with st.expander("Show all frames being analysed"):
                        st.image(p.contact_rgb, use_container_width=True,
                                 caption="Frames used by AI (left to right, top to bottom)")



# PAGE: INCIDENTS


def page_incidents(state: AppState):
    st.caption("All incidents are saved to your local database. Video clips are in `data/alerts/`.")

    api_url = f"http://localhost:{config.API_PORT}"

    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("Refresh", use_container_width=True):
            st.rerun()

    # Try live API first
    try:
        import requests
        resp = requests.get(f"{api_url}/incidents", timeout=2)
        db_incidents = resp.json().get("incidents", [])
    except Exception:
        db_incidents = []

    if db_incidents:
        rows_html = ""
        for inc in db_incidents:
            conf = inc.get("confidence", 0)
            conf_pct = f"{conf:.0%}" if isinstance(conf, float) else "—"
            conf_color = "#ff6060" if isinstance(conf, float) and conf >= config.CLASSIFIER_THRESHOLD else "#ffb340"
            fa = inc.get("false_alarm", False)
            badge = '<span style="background:#0d2a1a;color:#00e5a0;padding:2px 8px;border-radius:20px;font-size:11px">False alarm</span>' \
                    if fa else \
                    '<span style="background:#2a0d0d;color:#ff6060;padding:2px 8px;border-radius:20px;font-size:11px">Flagged</span>'
            rows_html += f"""<tr>
                <td style="font-family:monospace;color:#5a5f7a">#{inc.get('id','?')}</td>
                <td>{inc.get('timestamp','—')}</td>
                <td>{inc.get('camera_id','—')}</td>
                <td style="font-family:monospace">{inc.get('track_pair','—')}</td>
                <td><b style="color:{conf_color}">{conf_pct}</b></td>
                <td>{badge}</td>
            </tr>"""

        st.markdown(f"""
        <div style="background:#1a1d27;border:1px solid #2d3047;border-radius:12px;overflow:hidden;margin-bottom:20px">
        <table class="styled-table">
            <thead><tr><th>ID</th><th>Time</th><th>Camera</th><th>Tracks</th><th>Confidence</th><th>Status</th></tr></thead>
            <tbody>{rows_html}</tbody>
        </table>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-title">Mark a False Alarm</div>', unsafe_allow_html=True)
        inc_ids = [str(inc.get("id")) for inc in db_incidents if not inc.get("false_alarm")]
        if inc_ids:
            sel_id = st.selectbox("Which incident was a false alarm?", inc_ids)
            if st.button("Mark as False Alarm", type="primary"):
                try:
                    r = requests.patch(f"{api_url}/incidents/{sel_id}/false-alarm", timeout=2)
                    st.success(f"Incident #{sel_id} marked as false alarm.") if r.status_code == 200 \
                        else st.error(f"Error: {r.text}")
                except Exception as e:
                    st.error(f"Could not reach API: {e}")
        else:
            st.caption("All incidents have been reviewed.")

    else:
        # Fallback: in-memory incidents from this session
        with state.lock:
            incidents = list(state.recent_incidents)

        if incidents:
            for inc in incidents:
                conf = inc["confidence"]
                label = "Violent" if conf >= config.CLASSIFIER_THRESHOLD else "Low confidence"
                color = "#ff6060" if conf >= config.CLASSIFIER_THRESHOLD else "#ffb340"
                st.markdown(f"""
                <div style="background:#1a1d27;border:1px solid #2d3047;border-radius:10px;
                            padding:14px 18px;margin-bottom:8px;">
                    <div style="display:flex;align-items:center;gap:12px">
                        <span style="font-size:22px">🚨</span>
                        <div>
                            <div style="font-weight:600;color:{color}">{label}</div>
                            <div style="font-size:13px;color:#7a7f9a;margin-top:2px">
                                Tracks {inc["tracks"]} &middot; {conf:.0%} confidence &middot; {inc["time"]}
                            </div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background:#1a1d27;border:1px solid #2d3047;border-radius:12px;
                        padding:40px;text-align:center;color:#7a7f9a;">
                <div style="font-size:28px;margin-bottom:8px">🟢</div>
                <div style="font-size:14px;font-weight:600;color:#c8ccd8">No incidents yet</div>
                <div style="font-size:13px;margin-top:4px">
                    Incidents will appear here when the system detects something
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Saved clips
    st.divider()
    st.markdown('<div class="section-title">Saved Video Clips</div>', unsafe_allow_html=True)
    clips = sorted(config.ALERTS_DIR.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
    snaps = sorted(config.ALERTS_DIR.glob("*.jpg"), key=lambda p: p.stat().st_mtime, reverse=True)

    if clips:
        st.caption(f"{len(clips)} clip(s) saved in `{config.ALERTS_DIR}`")
        for clip in clips[:3]:
            st.video(str(clip))
    if snaps:
        snap_cols = st.columns(min(4, len(snaps)))
        for col, snap in zip(snap_cols, snaps[:4]):
            col.image(str(snap), caption=snap.name, use_container_width=True)
    if not clips and not snaps:
        st.caption(f"No clips saved yet — they appear here after an alert fires.")



# PAGE: SETTINGS


def page_settings():
    st.caption("Changes are saved to your `.env` file. Restart the app to apply them (except threshold, which updates live).")

    # --- Camera Source ---
    st.markdown('<div class="section-title">Camera Source</div>', unsafe_allow_html=True)

    source_map = {"Webcam": "webcam", "Video File": "file",
                  "YouTube / Live Stream": "youtube", "IP Camera (RTSP)": ""}
    reverse_map = {v: k for k, v in source_map.items()}
    current_source_label = reverse_map.get(config.SOURCE_TYPE, "Webcam")

    source_type = st.radio(
        "Where is your video coming from?",
        list(source_map.keys()),
        index=list(source_map.keys()).index(current_source_label),
        horizontal=True,
    )

    new_source_type = source_map[source_type]
    new_webcam_index = config.WEBCAM_INDEX
    new_video_file = config.VIDEO_FILE
    new_youtube_url = config.YOUTUBE_URL
    new_rtsp_url = "" if isinstance(config.SOURCE, int) else str(config.SOURCE)

    if source_type == "Webcam":
        new_webcam_index = st.number_input(
            "Camera number (0 = your default camera, 1 = second camera)",
            min_value=0, max_value=10, value=config.WEBCAM_INDEX)
        st.markdown('<div class="info-box">Most laptops have one built-in camera at index 0.</div>',
                    unsafe_allow_html=True)

    elif source_type == "Video File":
        new_video_file = st.text_input("Full path to video file",
            value=config.VIDEO_FILE or "",
            placeholder="C:/videos/clip.mp4  or  /home/user/clip.mp4")

    elif source_type == "YouTube / Live Stream":
        new_youtube_url = st.text_input("YouTube URL",
            value=config.YOUTUBE_URL or "",
            placeholder="https://www.youtube.com/watch?v=...")
        st.markdown('<div class="info-box">Lower quality streams have less lag and run faster.</div>',
                    unsafe_allow_html=True)

    elif source_type == "IP Camera (RTSP)":
        new_rtsp_url = st.text_input("Camera stream URL",
            value=new_rtsp_url,
            placeholder="rtsp://admin:password@192.168.1.100:554/stream")
        with st.expander("How to connect a phone as a camera"):
            st.markdown("""
**Android (IP Webcam app)**
1. Install *IP Webcam* from the Play Store
2. Open the app and tap **Start server**
3. Note the IP address shown (e.g. `192.168.1.42`)
4. Enter: `http://192.168.1.42:8080/video`

**iPhone (EpocCam)**
1. Install *EpocCam* on your phone and the PC driver from kinoni.com
2. Connect phone and PC to the same Wi-Fi
3. iPhone appears as camera index `1` or `2`
            """)

    st.divider()

    # --- AI Model ---
    st.markdown('<div class="section-title">AI Model</div>', unsafe_allow_html=True)

    model_options = {
        "kinetics_heuristic": "Fast — no training needed (good for testing)",
        "r3d18":              "R3D-18 — needs fine-tuning to work well",
        "slowfast_r50":       "SlowFast R50 — more accurate, needs a GPU",
        "slowfast_violence":  "SlowFast (your trained model) — best accuracy",
    }
    model_type = st.selectbox(
        "Which AI model?",
        options=list(model_options.keys()),
        format_func=lambda k: f"{k}  —  {model_options[k]}",
        index=list(model_options.keys()).index(config.MODEL_TYPE)
               if config.MODEL_TYPE in model_options else 0,
    )
    new_model_path = config.MODEL_PATH or ""
    if model_type in ("r3d18", "slowfast_violence"):
        new_model_path = st.text_input("Path to your trained model (.pt file)",
            value=new_model_path, placeholder="models/slowfast_violence.pt")
        st.markdown('<div class="warn-box">This model needs training first. '
                    'Open <code>train_slowfast.ipynb</code> to train it.</div>',
                    unsafe_allow_html=True)

    st.divider()

    # --- Detection Sensitivity ---
    st.markdown('<div class="section-title">Detection Sensitivity</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        threshold = st.slider("Violence confidence threshold",
            0.1, 0.99, float(config.CLASSIFIER_THRESHOLD), 0.01,
            help="How sure the AI must be before raising an alert. Higher = fewer false alarms.")
        sensitivity = "Very sensitive" if threshold < 0.5 else "Balanced" if threshold < 0.75 else "Conservative"
        st.caption(f"Current: {threshold:.0%} — {sensitivity}")

        persistence = st.slider("Consecutive detections before alert",
            1, 8, int(config.PERSISTENCE_COUNT), 1,
            help="How many clips in a row must show violence. Higher = more reliable, slower.")
        st.caption("Raise this to reduce false alarms")

    with col2:
        cooldown = st.slider("Alert cooldown (seconds)",
            5, 300, int(config.ALERT_COOLDOWN), 5,
            help="Minimum gap between repeated alerts for the same group.")
        st.caption(f"Won't re-alert same group for {cooldown}s")

        yolo_conf = st.slider("Person detection confidence",
            0.1, 0.95, float(config.YOLO_CONF), 0.01,
            help="How sure YOLO must be before marking something as a person.")
        st.caption("Lower to detect more people; raise to reduce ghost boxes")

    st.divider()

    # --- Performance ---
    st.markdown('<div class="section-title">Performance</div>', unsafe_allow_html=True)
    col3, col4 = st.columns(2)

    with col3:
        fps_limit = st.slider("Max FPS", 5, 60, int(config.FPS_LIMIT), 5,
            help="Processing speed cap. Lower = less CPU/GPU usage.")
        frame_skip = st.slider("Frame skip", 1, 8, int(config.FRAME_SKIP), 1,
            help="Only analyse every Nth frame. 2 = every other frame. Faster but less precise.")
        st.caption("Set frame skip to 1 for best accuracy; 3-4 on slow hardware")

    with col4:
        save_clips = st.toggle("Save alert video clips (.mp4)", value=config.SAVE_CLIPS)
        save_snaps = st.toggle("Save alert snapshots (.jpg)", value=config.SAVE_SNAPSHOTS)
        device_opts = ["auto", "cuda", "mps", "cpu"]
        device = st.selectbox("Compute device",
            device_opts,
            index=device_opts.index(config.DEVICE) if config.DEVICE in device_opts else 0,
            help="auto = uses GPU if available, otherwise CPU")

    st.divider()

    if st.button("Save All Settings", type="primary", use_container_width=True):
        _write_env({
            "SOURCE_TYPE": new_source_type,
            "WEBCAM_INDEX": str(new_webcam_index),
            "VIDEO_FILE": new_video_file,
            "YOUTUBE_URL": new_youtube_url,
            "SOURCE": new_rtsp_url if source_type == "IP Camera (RTSP)" else "",
            "CLASSIFIER_THRESHOLD": str(threshold),
            "PERSISTENCE_COUNT": str(persistence),
            "ALERT_COOLDOWN": str(cooldown),
            "YOLO_CONF": str(yolo_conf),
            "FPS_LIMIT": str(fps_limit),
            "FRAME_SKIP": str(frame_skip),
            "SAVE_CLIPS": "true" if save_clips else "false",
            "SAVE_SNAPSHOTS": "true" if save_snaps else "false",
            "MODEL_TYPE": model_type,
            "MODEL_PATH": new_model_path,
            "DEVICE": device,
        })
        st.success("Settings saved! Restart the app for changes to take effect.")
        st.markdown('<div class="info-box">Run <code>streamlit run streamlit_app.py</code> again to apply.</div>',
                    unsafe_allow_html=True)

    st.divider()

    # Live threshold update
    st.markdown('<div class="section-title">Update Threshold Right Now (no restart needed)</div>',
                unsafe_allow_html=True)
    live_thresh = st.slider("New threshold", 0.1, 0.99,
                            float(config.CLASSIFIER_THRESHOLD), 0.01, key="live_thresh")
    if st.button("Apply Now"):
        try:
            import requests
            r = requests.post(f"http://localhost:{config.API_PORT}/config/threshold",
                              json={"threshold": live_thresh}, timeout=2)
            st.success(f"Threshold updated to {live_thresh:.0%}!") if r.status_code == 200 \
                else st.error(r.text)
        except Exception:
            config.CLASSIFIER_THRESHOLD = live_thresh
            st.success(f"Threshold updated to {live_thresh:.0%} in memory.")


def _write_env(updates: dict):
    env_path = config.ROOT_DIR / ".env"
    existing = {}
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                k, _, v = line.partition("=")
                existing[k.strip()] = v.strip()
    existing.update({k: v for k, v in updates.items() if v})
    env_path.write_text("\n".join(f"{k}={v}" for k, v in existing.items()) + "\n")



# PAGE: HOW IT WORKS


def page_how_it_works():
    st.caption("A plain-English guide to how VisionGuard AI detects violence.")

    steps = [
        ("Step 1 — Camera Input",
         "Video frames are read from your camera (webcam, phone, IP camera, or file) in real time."),
        ("Step 2 — Person Detection (YOLOv8)",
         "Every frame is scanned by YOLOv8, which finds and boxes every person. "
         "Each person gets a unique ID that follows them across frames."),
        ("Step 3 — Proximity Gate",
         "The expensive AI only runs when 2+ people are close to each other. "
         "If everyone is far apart, violence can't happen — so we skip classification entirely."),
        ("Step 4 — Frame Buffer",
         "We collect a short clip of frames (default: 16). Violence happens over time — "
         "the AI needs a sequence, not just a single image."),
        ("Step 5 — Violence Classifier (3D-CNN)",
         "The clip is fed to a neural network trained to recognise violent motion. "
         "It outputs a score: 0 = calm, 1 = violent."),
        ("Step 6 — Alert",
         "If the score exceeds your threshold for several clips in a row, an alert fires: "
         "sound plays, a clip is saved, and the live view flashes red."),
    ]

    for icon, title, desc in steps:
        st.markdown(f"""
        <div style="background:#1a1d27;border:1px solid #2d3047;border-radius:12px;
                    padding:18px 20px;margin-bottom:10px;display:flex;gap:16px;align-items:flex-start">
            <span style="font-size:28px;flex-shrink:0">{icon}</span>
            <div>
                <div style="font-size:15px;font-weight:700;color:#e8eaf2;margin-bottom:4px">{title}</div>
                <div style="font-size:13.5px;color:#9096b0;line-height:1.6">{desc}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()
    st.markdown('<div class="section-title">Tuning Guide</div>', unsafe_allow_html=True)

    for name, default, tip in [
        ("Violence threshold", "0.65",
         "Raise to 0.80+ to only alert on obvious violence (fewer false alarms). "
         "Lower to 0.50 to catch more (more false alarms)."),
        ("Persistence count", "2",
         "Set to 1 for instant alerts. Set to 3-4 to require sustained violence before alerting."),
        ("Alert cooldown", "60s",
         "Prevents spam alerts from the same ongoing incident."),
        ("Person detection confidence", "0.45",
         "Lower to 0.3 in dark/crowded scenes. Raise to 0.6 in clean, well-lit environments."),
        ("Frame skip", "2",
         "1 = process every frame (most accurate). 3-4 = faster on slow hardware."),
    ]:
        with st.expander(f"  {name}  (default: {default})"):
            st.write(tip)



# SIDEBAR + MAIN


def render_sidebar(state: AppState) -> str:
    with st.sidebar:
        st.markdown("""
        <div style="padding:8px 0 20px">
            <div class="logo-text">👁 VisionGuard AI</div>
            <div class="logo-sub">Real-time Violence Detection</div>
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        pages = {
            "Incidents":     "incidents",
            "Live Monitor":  "live",
            "Settings":      "settings",
            "How It Works":  "howto",
        }
        selected = st.radio("Navigate", list(pages.keys()),
                            label_visibility="collapsed")

        st.divider()

        with state.lock:
            fps, n_tracks = state.fps, state.track_count
            gate_open, alert = state.gate_open, state.alert_active
            status = state.status

    return pages[selected]


def main():
    st.set_page_config(
        page_title="VisionGuard AI",
        page_icon="👁",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_css()

    state = _get_pipeline_state()
    page = render_sidebar(state)

    titles = {
        "live":      ("Live Monitor",  "Real-time detection feed"),
        "incidents": ("Incidents",     "View and manage flagged events"),
        "settings":  ("Settings",      "Configure camera and detection"),
        "howto":     ("How It Works",  "Understand how the AI works"),
    }
    title, subtitle = titles.get(page, ("VisionGuard AI", ""))
    st.markdown(f"""
    <div style="margin-bottom:20px">
        <h1 style="font-size:26px;font-weight:800;color:#e8eaf2;margin:0">{title}</h1>
        <p style="font-size:14px;color:#7a7f9a;margin:4px 0 0">{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)

    if page == "live":
        placeholder = st.empty()
        while True:
            with placeholder.container():
                page_live_monitor(state)
            time.sleep(0.05)
    elif page == "incidents":
        page_incidents(state)
    elif page == "settings":
        page_settings()
    elif page == "howto":
        page_how_it_works()


if __name__ == "__main__":
    main()
