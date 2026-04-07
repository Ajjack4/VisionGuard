# VisionGuard AI — Claude Code Instructions

## Project Overview

Real-time violence detection system. Python 3.10+, YOLOv8 tracking, SlowFast/R3D-18 classifier,
Streamlit dashboard, FastAPI REST API, SQLite incident log.

**Entry points:**
- `streamlit run streamlit_app.py` — primary UI (recommended)
- `python main.py` — headless/OpenCV mode

---

## Architecture Rules

- All config comes from `.env` via `config.py`. Never hardcode paths, ports, or thresholds.
- All paths use `pathlib.Path`. No raw string paths.
- The classifier is **swappable** — `ClipClassifier` accepts `str | nn.Module | None`. Any new model
  must accept `[B, C, T, H, W]` float32 and return `[B, 2]` softmax probabilities.
- Group keys are always `tuple(sorted(track_ids))` — normalised before use as dict keys.
- `PairBuffer`/`TemporalBuffer` keys are canonical (min, max) — never assume insertion order.
- `gate.evaluate_groups()` returns `List[List[Track]]` via DFS connected components.
  `gate.evaluate()` is a backward-compat shim; prefer `evaluate_groups()` in new code.
- Streamlit state uses `@st.cache_resource` for the pipeline singleton. Do NOT use
  `session_state` or module-level globals for the pipeline thread.
- Thread safety: all shared state in `streamlit_app.py` is guarded by `AppState.lock`.
- Track IDs from DeepSORT can be `int` or `str`. Always use `abs(hash(str(track_id)))` for
  modulo operations.

---

## File Map

```
main.py                     Headless pipeline + FastAPI daemon + OpenCV display
config.py                   All env vars with defaults
streamlit_app.py            Streamlit dashboard (primary UI)
pipeline/
  stream_reader.py          Video capture: webcam, file, RTSP, YouTube
  detector.py               YOLOv8 person detection + crop extraction
  tracker.py                DeepSORT with IoU fallback
  gate.py                   N-person proximity gate (connected components)
  buffer.py                 TemporalBuffer + PairBuffer (ring buffers)
  classifier.py             ClipClassifier — swappable 3D-CNN wrapper
  alert.py                  AlertEngine — sound, clips, snapshots, SQLite
models/
  r3d_classifier.py         R3D-18 binary head (fine-tunable)
  kinetics_heuristic.py     Zero-shot Kinetics-400 heuristic scorer
  slowfast_wrapper.py       SlowFastWrapper — PyTorchVideo SlowFast R50
api/
  server.py                 FastAPI REST server (7 endpoints)
train_slowfast.ipynb        17-cell training notebook for SlowFast fine-tuning
data/alerts/                Saved clips (MP4) and snapshots (JPEG) — auto-created
data/visionguard.db         SQLite incident log — auto-created
assets/alert.wav            Alert beep — auto-generated if missing
```

---

## Key Environment Variables

| Variable | Default | Notes |
|---|---|---|
| `SOURCE_TYPE` | `webcam` | `webcam` / `file` / `youtube` |
| `MODEL_TYPE` | `kinetics_heuristic` | See classifier backends below |
| `MODEL_PATH` | _(blank)_ | Path to fine-tuned weights |
| `BUFFER_SIZE` | `16` | Set to `32` for SlowFast |
| `GATE_ALPHA` | `2.5` | Proximity multiplier |
| `CLASSIFIER_THRESHOLD` | `0.65` | P(violent) cutoff |
| `PERSISTENCE_COUNT` | `2` | Consecutive positives before alert |
| `DEBUG_CROPS` | `false` | Show classifier contact sheet window |

Classifier backends: `kinetics_heuristic` | `r3d18` | `x3d_xs` | `slowfast_r50` | `slowfast_violence`

---

## Common Tasks

**Add a new classifier backend:**
1. Create `models/my_model.py` with a class accepting `[B,C,T,H,W]` → `[B,2]`
2. Add a branch in `pipeline/classifier.py` `_load_model()` for the new `MODEL_TYPE` string
3. Add the option to the `MODEL_TYPE` comment block in `config.py` and `.env.example`

**Change proximity logic:**
Edit `pipeline/gate.py` `_are_proximate()`. The rest of the pipeline reads `evaluate_groups()`
and adapts automatically.

**Add a new alert channel (email, SMS, Telegram):**
Add a method to `pipeline/alert.py` `AlertEngine` and call it from `update()` after `fired = True`.

**Add a new REST endpoint:**
Add a route to `api/server.py`. State is injected via `set_pipeline_state()` — use `_state` dict.

**Run training:**
```bash
jupyter notebook train_slowfast.ipynb   # local GPU
# or open on Kaggle with dataset already mounted (free T4 GPU)
```
After training, set `MODEL_TYPE=slowfast_violence`, `MODEL_PATH=models/slowfast_violence.pt`,
`BUFFER_SIZE=32` in `.env`.

---

## Production Readiness Backlog

The following features are needed to move from demo to production. Tackle in priority order.

### P0 — Must-have before any real deployment

- [ ] **Multi-camera support** — loop over multiple `StreamReader` instances, each with its own
  `Detector`, `Tracker`, `ProximityGate`, and `TemporalBuffer`. Share one `AlertEngine` and
  `ClipClassifier`. `camera_id` column already exists in the SQLite schema.

- [ ] **Authentication on the REST API** — `api/server.py` currently has no auth. Add an
  `API_KEY` env var and a `verify_api_key` FastAPI dependency on all non-health endpoints.
  Use `secrets.compare_digest` for constant-time comparison.

- [ ] **HTTPS / TLS for the API** — run uvicorn behind a reverse proxy (nginx/caddy) or pass
  `ssl_certfile` / `ssl_keyfile` to `uvicorn.run()` in `main.py`.

- [ ] **Fine-tuned model weights** — `kinetics_heuristic` is heuristic-only. Train
  `train_slowfast.ipynb` on RWF-2000 or UCF Crime dataset and set `MODEL_TYPE=slowfast_violence`.
  Target: ≥85% F1 on a held-out test set.

- [ ] **Graceful shutdown** — add `signal.signal(SIGTERM, ...)` handler in `main.py` and
  `streamlit_app.py` that flushes the alert buffer, closes the DB connection, and releases
  the VideoCapture before exit.

### P1 — Important for reliability

- [ ] **Persistent stream reconnection** — `stream_reader.py` already reconnects on frame read
  failure, but the retry interval is fixed. Add exponential backoff (max 30 s) and a
  `MAX_RECONNECT_ATTEMPTS` config var. Log each attempt.

- [ ] **Alert notification integrations** — add at least one push channel in `alert.py`:
  - Email via `smtplib` (SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS, ALERT_EMAIL env vars)
  - Telegram bot via `requests.post` to Bot API (TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
  - Twilio SMS (optional)

- [ ] **Clip storage management** — `data/alerts/` grows unbounded. Add a
  `MAX_ALERT_STORAGE_MB` env var. On each save, check total dir size and delete oldest clips
  if over limit. Add a `GET /storage/status` API endpoint.

- [ ] **Test suite** — add `tests/` with:
  - Unit tests for `gate.py` connected-component logic (edge cases: 1 person, chain of 3, fully connected graph)
  - Unit tests for `buffer.py` ring-buffer correctness and `cleanup_stale()`
  - Integration smoke test: feed a synthetic black frame through the full pipeline, assert no crash
  - Use `pytest` + `pytest-cov`

- [ ] **Docker support** — add `Dockerfile` and `docker-compose.yml`. Base image:
  `pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime`. Mount `/data` and `/models` as volumes.
  Expose ports 8000 (API) and 8501 (Streamlit).

### P2 — Quality of life

- [ ] **Streamlit live incident table** — add a second tab in `streamlit_app.py` that polls
  `GET /incidents` every 5 s and renders a `st.dataframe` of recent events with thumbnail
  previews of saved snapshots.

- [ ] **Runtime model swap via API** — add `POST /model/swap` endpoint that accepts a `.pt`
  file path and calls `classifier.swap_model(new_model)`. Useful for A/B testing weights
  without restarting.

- [ ] **Confidence history chart** — store a rolling deque of (timestamp, confidence) per
  group key in `AppState`. Render as `st.line_chart` below each group panel in Streamlit.

- [ ] **Export incidents to CSV** — add `GET /incidents/export.csv` endpoint in `server.py`
  using `csv.writer` on the SQLite query result. Useful for review sessions.

- [ ] **Re-identification across tracks** — DeepSORT resets IDs when a person leaves and
  re-enters frame. Add an appearance embedding cache (torchreid or OSNet) to re-assign
  consistent IDs. This prevents the same person forming a new pair buffer unnecessarily.

- [ ] **ONVIF PTZ control on alert** — when `alert.py` fires, command a compatible PTZ
  camera to zoom to the incident bounding box. Use `onvif-zeep` library.

- [ ] **Web-based incident review UI** — replace the raw FastAPI docs with a React or HTMX
  single-page app served at `/ui`. Show incident timeline, video player for saved clips,
  false-alarm button. The REST API already provides all necessary endpoints.

### P3 — Performance & scale

- [ ] **Async pipeline with asyncio** — the current pipeline is synchronous in a thread.
  Refactor to `asyncio` with `asyncio.Queue` between stages (reader → detector → tracker →
  gate → classifier) to reduce head-of-line blocking and improve GPU utilisation.

- [ ] **Batch inference** — `ClipClassifier.predict()` currently processes one clip at a time.
  Accumulate clips across active groups and call `predict_batch()` once per cycle for better
  GPU throughput.

- [ ] **TensorRT / ONNX export** — export `SlowFastWrapper` to ONNX and optimise with
  TensorRT for 2–4× inference speedup on NVIDIA hardware. Keep `nn.Module` path as fallback.

- [ ] **Multi-GPU support** — add `DEVICE_IDS` env var. Use `nn.DataParallel` for classifier
  and split YOLO across devices if multiple cameras are active.

- [ ] **Horizontal scaling** — containerise the pipeline and add a Redis pub/sub bus for
  alert events so multiple instances can share a single alert sink and dashboard.

---

## Data Preparation for Training

Expected folder structure for `train_slowfast.ipynb`:
```
data/
  train/
    violence/        *.mp4, *.avi — fight / assault clips
    nonviolence/     *.mp4, *.avi — normal activity clips
  val/
    violence/
    nonviolence/
```

Recommended datasets (binary violence mapping):
- **RWF-2000** — 2000 fight clips from surveillance footage (best match for this use case)
- **UCF Crime** — map Fighting + Assault → violence; Normal_Videos → nonviolence
- **Hockey Fight** — 1000 hockey fight clips
- **RLVS** — real-world violence from CCTV

For UCF Crime (Kaggle): categories to use as `violence/`:
`Fighting`, `Assault`, `Abuse`, `Shooting`, `Explosion`, `Robbery`

Categories to use as `nonviolence/`:
`Normal_Videos_event`, `Stealing`, `Shoplifting`, `Burglary`

---

## Coding Conventions

- Format with `black` (default settings) before committing
- Type-annotate all function signatures
- No `print()` outside pipeline startup — use `[ClassName]` prefix pattern: `print("[Gate] ...")`
- Do not catch bare `Exception` — catch specific exceptions and log the type
- Keep `config.py` as the single source of truth — no magic numbers in pipeline files
- Test changes with `DEBUG_CROPS=true` to visually verify what the classifier receives
