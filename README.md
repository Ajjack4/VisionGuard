# VisionGuard AI

VisionGuard AI is a real-time surveillance system that uses computer vision and deep learning to detect violent incidents in live video feeds. It ingests video from a webcam, Android IP Webcam app, iPhone (via Continuity Camera or EpocCam), or any RTSP-compatible IP camera. YOLOv8 detects and tracks every person in the scene, and a proximity gate ensures the expensive 3D-CNN classifier only runs when two or more people are close together — saving compute on quiet scenes.

When the classifier detects violence across multiple consecutive clips, the system fires an alert: an audio tone plays, a snapshot and MP4 clip are saved to disk, the incident is logged to a local SQLite database, and the video window flashes red. A lightweight FastAPI server runs alongside the pipeline so you can query incidents, tune the classifier threshold at runtime, and check stream status from any HTTP client or browser.

---

## Prerequisites

- **Python 3.10 or newer**
- **pip** (comes with Python)
- A CUDA-capable GPU is recommended for real-time performance, but **CPU-only** and **Apple Silicon (MPS)** are fully supported with automatic fallback and speed warnings
- No Docker required — everything runs directly with `pip install`

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-org/visionguard.git
cd visionguard

# 2. (Recommended) create a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Copy the example config and edit it
cp .env.example .env
# Open .env in your editor and set SOURCE, thresholds, etc.
```

> **Note:** On first run, YOLOv8 (`yolov8n.pt`) and torchvision R3D-18 weights
> are downloaded automatically (~25 MB + ~120 MB).

---

## Running

### Webcam (default)

```bash
python main.py
```

Or override the source at the command line without editing `.env`:

```bash
python main.py --source 0        # built-in webcam
python main.py --source 1        # second webcam / Continuity Camera
python main.py --source video.mp4  # offline video file for testing
```

Press **`q`** in the video window to quit gracefully.

---

## Connecting an Android Phone (IP Webcam app)

1. Install **IP Webcam** by Pavel Khlebovich from the Google Play Store.
2. Open the app, scroll to the bottom, and tap **"Start server"**.
3. Note the IP address shown (e.g. `192.168.1.42`) and port (default `8080`).
4. Confirm your phone and PC are on the **same Wi-Fi network**.
5. Test in your browser: `http://192.168.1.42:8080` — you should see a live feed.
6. Set `SOURCE` in `.env` (or use `--source`):
   ```
   SOURCE=http://192.168.1.42:8080/video
   ```
7. Run: `python main.py`

Or print the full guide at any time:

```bash
python main.py --mobile-setup
```

---

## Connecting an iPhone

### Option A — Continuity Camera (macOS 13+ / iOS 16+, no extra software)

1. Sign both your iPhone and Mac into the **same Apple ID**.
2. On Mac: **System Settings → General → AirPlay & Handoff** → enable *Continuity Camera*.
3. On iPhone: **Settings → General → AirPlay & Handoff** → enable *Continuity Camera*.
4. Your iPhone appears automatically as an additional camera device.
5. Try source index `1` or `2` in VisionGuard:
   ```
   SOURCE=1
   ```

### Option B — EpocCam (Windows or Mac)

1. Install **EpocCam Webcam** from the App Store (free tier available).
2. Download and install the **EpocCam driver** for your PC or Mac from the Kinoni website.
3. Connect iPhone and computer to the same Wi-Fi network (or via USB).
4. Open the EpocCam app — it begins broadcasting as a virtual webcam device.
5. Set `SOURCE=1` (or `2`) in `.env` and run `python main.py`.

---

## Connecting an IP Camera via RTSP

Most network cameras expose an RTSP stream. The URL format varies by vendor:

```
rtsp://username:password@192.168.1.100:554/stream
rtsp://admin:admin@192.168.1.100:554/h264Preview_01_main   # Reolink example
rtsp://192.168.1.100:554/live/ch0                          # Hikvision example
```

Set in `.env`:

```
SOURCE=rtsp://admin:admin@192.168.1.100:554/h264Preview_01_main
```

---

## Swapping in Your Own Fine-Tuned Model

The classifier is intentionally designed to be swappable. The default model
uses an R3D-18 backbone with Kinetics-400 pretrained weights — it will produce
**random-ish violence predictions** until you fine-tune it on labelled data.

### Step 1 — Prepare labelled data

Collect short video clips (≈1–2 s, 16 frames) labelled as `violent` or
`non_violent`. Datasets to consider:

- **RWF-2000** — 2 000 real-world fight clips (recommended for starting out)
- **UCF-Crime** — broader anomaly detection
- **Hockey Fight** / **Movies Fight** datasets

### Step 2 — Fine-tune

Use [MMAction2](https://github.com/open-mmlab/mmaction2) or
[PySlowFast](https://github.com/facebookresearch/SlowFast), or write a simple
PyTorch training loop against `models.r3d_classifier.ViolenceClassifier`.

Quick training loop example:

```python
from models.r3d_classifier import ViolenceClassifier
import torch, torch.nn as nn

model = ViolenceClassifier(pretrained=True)
opt   = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = nn.CrossEntropyLoss()

for clips, labels in dataloader:     # clips: [B,3,16,112,112]  labels: [B]
    logits = model(clips)
    loss   = loss_fn(logits, labels)
    opt.zero_grad(); loss.backward(); opt.step()
```

### Step 3 — Save and deploy

```python
torch.save(model.state_dict(), "my_violence_model.pth")
```

### Step 4 — Point VisionGuard at the new weights

In `.env`:

```
MODEL_PATH=my_violence_model.pth
```

VisionGuard's `pipeline/classifier.py` loads the state dict automatically on
the next `python main.py`.  No other code changes are needed.

---

## REST API Reference

The API server starts automatically on `http://localhost:8000` (configurable
via `API_PORT` in `.env`).

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/health` | Returns `{"status": "ok", "uptime_seconds": N}` |
| `GET`  | `/incidents` | Last 20 incidents from SQLite |
| `GET`  | `/incidents/{id}` | Single incident record |
| `PATCH`| `/incidents/{id}/false-alarm` | Mark as false alarm |
| `GET`  | `/config` | Current configuration values (read-only) |
| `POST` | `/config/threshold` | Update `CLASSIFIER_THRESHOLD` at runtime — body: `{"threshold": 0.7}` |
| `GET`  | `/stream/status` | Camera connection, FPS, track count, gate status |

Interactive API docs: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## Tuning Thresholds

| Parameter | Default | Effect |
|-----------|---------|--------|
| `CLASSIFIER_THRESHOLD` | `0.65` | P(violent) required to count a clip as positive. Raise to reduce false alarms; lower to catch more events. |
| `PERSISTENCE_COUNT` | `2` | Consecutive positive clips before alert fires. Increase for fewer spurious alerts. |
| `ALERT_COOLDOWN` | `60` s | Minimum seconds between repeated alerts for the same pair. |
| `GATE_ALPHA` | `2.5` | Proximity multiplier. Lower values = tighter proximity required to open the gate. |
| `YOLO_CONF` | `0.45` | YOLO detection confidence. Lower catches more people; higher reduces ghost detections. |

You can also update `CLASSIFIER_THRESHOLD` live without restarting:

```bash
curl -X POST http://localhost:8000/config/threshold \
     -H "Content-Type: application/json" \
     -d '{"threshold": 0.70}'
```

---

## Project Structure

```
visionguard/
├── main.py                  # Entry point — starts pipeline + API
├── config.py                # All tuneable parameters (dotenv)
├── .env.example             # Template config file
│
├── pipeline/
│   ├── stream_reader.py     # Video capture: webcam, IP cam, RTSP, file
│   ├── detector.py          # YOLOv8 human detection wrapper
│   ├── tracker.py           # DeepSORT wrapper + IoU fallback
│   ├── gate.py              # Multi-person proximity gate logic
│   ├── buffer.py            # Per-track temporal frame ring buffers
│   ├── classifier.py        # 3D-CNN classifier — SWAPPABLE module
│   └── alert.py             # Alert logic: sound, file save, DB log
│
├── models/
│   └── r3d_classifier.py    # R3D-18 model definition — fine-tunable
│
├── api/
│   └── server.py            # FastAPI app — incidents + config API
│
├── data/
│   ├── alerts/              # Saved alert clips (MP4) and snapshots (JPEG)
│   └── visionguard.db       # SQLite incident database (auto-created)
│
├── assets/
│   └── alert.wav            # Alert beep (auto-generated if missing)
│
├── requirements.txt
└── README.md
```

---

## Known Limitations & Next Steps

- **Classifier accuracy**: The default R3D-18 backbone produces near-random
  violence predictions until fine-tuned on labelled data. This is expected —
  the pipeline, gate logic, and infrastructure are complete; accuracy depends
  entirely on your training data.

- **CPU performance**: On a modern laptop CPU expect ≈2–5 fps end-to-end. A
  mid-range NVIDIA GPU (RTX 3060+) reaches 15+ fps comfortably.

- **Single camera**: The current implementation handles one stream. Adding
  multi-camera support requires a loop over multiple `StreamReader` instances
  with a shared `AlertEngine` (the API already supports multiple `camera_id`
  values in the database).

- **No re-identification**: DeepSORT track IDs reset when people leave and
  re-enter frame. Cross-camera re-ID is a natural extension.

- **Audio on headless servers**: `pygame` requires an audio device. On headless
  Linux, install `libasound2-dev` or redirect audio with PulseAudio.

### Suggested next steps

1. Fine-tune on RWF-2000 — expected to reach ≥85% accuracy
2. Add a web dashboard (React + WebSocket) consuming the REST API
3. Add email / SMS / Telegram notifications in `alert.py`
4. Integrate with ONVIF-compatible cameras for PTZ control on alert
5. Export incident reports to PDF or CSV
