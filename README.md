# VisionGuard AI

VisionGuard AI is a real-time surveillance system that uses computer vision and deep learning to detect violent incidents in live video feeds. It ingests video from a webcam, Android IP Webcam app, iPhone (via Continuity Camera or EpocCam), YouTube live stream, or any RTSP-compatible IP camera.

YOLOv8 detects and tracks every person in the scene. A proximity gate — using connected-component clustering — ensures the expensive 3D-CNN classifier only runs when two or more people are close together. When multiple people form a cluster, **one classifier handles the entire group** rather than duplicating work across pairs. When the classifier detects violence across multiple consecutive clips, the system fires an alert: an audio tone plays, a snapshot and MP4 clip are saved to disk, the incident is logged to a local SQLite database, and the live view flashes red.

A Streamlit dashboard provides the primary UI: a full detection view with unique per-track bounding-box colours, live FPS/track metrics, and per-group classifier panels showing exactly what the model sees. A lightweight FastAPI server runs alongside for programmatic access to incidents and runtime threshold changes.

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
# Open .env in your editor and set SOURCE_TYPE, thresholds, etc.
```

> **Note:** On first run, YOLOv8 (`yolov8n.pt`) and the classifier backbone weights
> are downloaded automatically (~25 MB + ~120 MB).

---

## Running

### Streamlit dashboard (recommended)

```bash
streamlit run streamlit_app.py
```

Opens at [http://localhost:8501](http://localhost:8501). The dashboard shows:

- **Top** — main YOLO detection view with per-person bounding boxes and unique track-ID colours
- **Bottom** — one panel per active proximate group showing the merged crop region fed to the classifier, a temporal contact sheet of the frame buffer, and a live confidence progress bar

The entire pipeline (detection → tracking → gate → classifier → alerts) runs in a background thread; the UI refreshes at ~20 fps without blocking.

### Headless / terminal mode

```bash
python main.py
```

Uses OpenCV windows instead of Streamlit. Useful for servers without a browser or for lower overhead. Press **`q`** in the video window to quit.

### Source override examples

Set `SOURCE_TYPE` in `.env` (preferred) or use the env variables directly:

```env
# Webcam
SOURCE_TYPE=webcam
WEBCAM_INDEX=0

# Local video file
SOURCE_TYPE=file
VIDEO_FILE=C:/path/to/video.mp4

# YouTube video or live stream
SOURCE_TYPE=youtube
YOUTUBE_URL=https://www.youtube.com/watch?v=XXXXXXXXXX
YOUTUBE_QUALITY=best[height<=480]
```

---

## Connecting a Camera

### Android phone (IP Webcam app)

1. Install **IP Webcam** by Pavel Khlebovich from the Google Play Store.
2. Open the app and tap **"Start server"** at the bottom.
3. Note the IP address shown (e.g. `192.168.1.42`) — phone and PC must share the same Wi-Fi.
4. Test in a browser: `http://192.168.1.42:8080` — you should see the live feed.
5. In `.env`:
   ```
   SOURCE=http://192.168.1.42:8080/video
   ```

### iPhone

**Option A — Continuity Camera (macOS 13+ / iOS 16+, no extra software)**

1. Sign both devices into the same Apple ID.
2. Enable *Continuity Camera* on both: **System Settings → General → AirPlay & Handoff**.
3. iPhone appears as a camera device; use `SOURCE=1` or `SOURCE=2`.

**Option B — EpocCam (Windows or Mac)**

1. Install **EpocCam Webcam** from the App Store and the matching driver from the Kinoni website.
2. Connect phone and PC to the same Wi-Fi (or via USB).
3. Set `SOURCE=1` (or `2`) in `.env`.

### IP camera via RTSP

```
rtsp://username:password@192.168.1.100:554/stream
rtsp://admin:admin@192.168.1.100:554/h264Preview_01_main   # Reolink
rtsp://192.168.1.100:554/live/ch0                          # Hikvision
```

In `.env`:
```
SOURCE=rtsp://admin:admin@192.168.1.100:554/h264Preview_01_main
```

---

## Classifier Backends

The classifier is **swappable** — set `MODEL_TYPE` in `.env`:

| `MODEL_TYPE` | Description | Training needed? |
|---|---|---|
| `kinetics_heuristic` | Zero-shot R3D-18 with heuristic violence scoring (default) | No |
| `r3d18` | Binary R3D-18 head — random until fine-tuned | Yes |
| `x3d_xs` / `x3d_s` / `x3d_m` | Facebook X3D models via torch.hub | No (PoC) |
| `slowfast_r50` | SlowFast R50 via torch.hub | No (PoC) |
| `slowfast_violence` | Fine-tuned SlowFast R50 from `train_slowfast.ipynb` | Yes — see below |

---

## Fine-Tuning SlowFast R50 (Recommended)

The repository ships a complete Jupyter training notebook:

```bash
jupyter notebook train_slowfast.ipynb
```

The notebook handles everything end-to-end:

1. **Data loading** — reads from `data/{train,val}/{violence,nonviolence}/` (any video format)
2. **Augmentation** — random crop, horizontal flip, colour jitter, ImageNet normalisation
3. **Class balancing** — `WeightedRandomSampler` handles imbalanced datasets automatically
4. **Model** — `SlowFastWrapper`: wraps PyTorchVideo SlowFast R50, replaces the 400-class Kinetics head with a binary output
5. **Two-phase training**:
   - Phase 1 (5 epochs) — backbone frozen, head only — fast initial convergence
   - Phase 2 (15 epochs) — full fine-tune with cosine LR decay — maximises accuracy
6. **Evaluation** — classification report, confusion matrix, ROC curve, Youden-optimal threshold
7. **Export** — saves `models/slowfast_violence.pt` + `models/slowfast_violence.json` metadata

After training, activate the model in `.env`:

```env
MODEL_TYPE=slowfast_violence
MODEL_PATH=models/slowfast_violence.pt
BUFFER_SIZE=32
```

### GPU kernel for training

Select a kernel that has **PyTorch with CUDA** installed. To verify:

```python
import torch
print(torch.cuda.is_available())       # must be True
print(torch.cuda.get_device_name(0))   # e.g. "NVIDIA GeForce RTX 3080"
```

To create a dedicated environment:

```bash
conda create -n visionguard python=3.10 -y
conda activate visionguard
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install pytorchvideo decord torchmetrics scikit-learn matplotlib seaborn ipykernel
python -m ipykernel install --user --name visionguard --display-name "VisionGuard (GPU)"
```

Then select **"VisionGuard (GPU)"** as the Jupyter kernel.

### Training with an older R3D-18 model

For a lighter-weight option (no SlowFast dependency):

```python
from models.r3d_classifier import ViolenceClassifier
import torch, torch.nn as nn

model   = ViolenceClassifier(pretrained=True)
opt     = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = nn.CrossEntropyLoss()

for clips, labels in dataloader:     # clips: [B, 3, 16, 112, 112]
    logits = model(clips)
    loss   = loss_fn(logits, labels)
    opt.zero_grad(); loss.backward(); opt.step()

torch.save(model.state_dict(), "models/r3d18_violence.pth")
```

Set `MODEL_TYPE=r3d18` and `MODEL_PATH=models/r3d18_violence.pth` in `.env`.

### Recommended datasets

- **RWF-2000** — 2 000 real-world fight clips (best starting point)
- **UCF-Crime** — broader anomaly detection
- **Hockey Fight** / **Movies Fight** datasets

---

## REST API Reference

The API server starts automatically on `http://localhost:8000` (configurable via `API_PORT` in `.env`).

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/health` | `{"status": "ok", "uptime_seconds": N}` |
| `GET`  | `/incidents` | Last 20 incidents from SQLite |
| `GET`  | `/incidents/{id}` | Single incident record |
| `PATCH`| `/incidents/{id}/false-alarm` | Mark as false alarm |
| `GET`  | `/config` | Current configuration values (read-only) |
| `POST` | `/config/threshold` | Update `CLASSIFIER_THRESHOLD` at runtime — body: `{"threshold": 0.7}` |
| `GET`  | `/stream/status` | Camera connection, FPS, track count, gate status |

Interactive docs: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## Tuning Thresholds

| Parameter | Default | Effect |
|-----------|---------|--------|
| `CLASSIFIER_THRESHOLD` | `0.65` | P(violent) required to count a clip as positive. Raise to reduce false alarms; lower to catch more events. |
| `PERSISTENCE_COUNT` | `2` | Consecutive positive clips before alert fires. Increase to suppress spurious alerts. |
| `ALERT_COOLDOWN` | `60` s | Minimum seconds between repeated alerts for the same group. |
| `GATE_ALPHA` | `2.5` | Proximity multiplier. Lower = tighter proximity required to open the gate. |
| `GATE_MIN_PEOPLE` | `2` | Minimum people in scene before any gate check runs. |
| `YOLO_CONF` | `0.45` | YOLO detection confidence. Lower catches more people; higher reduces ghost detections. |
| `BUFFER_SIZE` | `16` | Frames per clip fed to the classifier. Use `32` with SlowFast. |
| `FRAME_SKIP` | `2` | Run detection every Nth frame (intermediate frames reuse last annotations). |

Update `CLASSIFIER_THRESHOLD` live without restarting:

```bash
curl -X POST http://localhost:8000/config/threshold \
     -H "Content-Type: application/json" \
     -d '{"threshold": 0.70}'
```

---

## Project Structure

```
visionguard/
├── streamlit_app.py         # Streamlit dashboard (primary UI)
├── main.py                  # Headless entry point — OpenCV + FastAPI
├── config.py                # All tuneable parameters (dotenv)
├── .env.example             # Template config file
├── train_slowfast.ipynb     # End-to-end SlowFast R50 fine-tuning notebook
│
├── pipeline/
│   ├── stream_reader.py     # Video capture: webcam, file, YouTube, RTSP
│   ├── detector.py          # YOLOv8 human detection wrapper
│   ├── tracker.py           # DeepSORT wrapper + IoU fallback
│   ├── gate.py              # Connected-component proximity gate
│   ├── buffer.py            # Per-track / per-group temporal ring buffers
│   ├── classifier.py        # 3D-CNN classifier — swappable backend
│   └── alert.py             # Alert logic: sound, file save, DB log
│
├── models/
│   ├── r3d_classifier.py    # R3D-18 binary head
│   ├── slowfast_wrapper.py  # SlowFast R50 wrapper (binary output)
│   └── kinetics_heuristic.py# Zero-shot Kinetics heuristic scorer
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

- **Classifier accuracy**: The default `kinetics_heuristic` backbone is zero-shot and tuned for general action recognition, not violence specifically. For production accuracy, fine-tune using `train_slowfast.ipynb` on RWF-2000 or similar data.

- **CPU performance**: On a modern laptop CPU expect ≈2–5 fps end-to-end. A mid-range NVIDIA GPU (RTX 3060+) reaches 15+ fps comfortably.

- **Single camera**: The current implementation handles one stream. Multi-camera support would require looping over multiple `StreamReader` instances with a shared `AlertEngine` (the API already supports multiple `camera_id` values in the database).

- **No re-identification**: DeepSORT track IDs reset when people leave and re-enter the frame. Cross-camera re-ID is a natural extension.

- **Audio on headless servers**: `pygame` requires an audio device. On headless Linux, install `libasound2-dev` or redirect audio with PulseAudio.

### Suggested next steps

1. Fine-tune on RWF-2000 with `train_slowfast.ipynb` — expected to reach ≥85% accuracy
2. Add email / SMS / Telegram notifications in `pipeline/alert.py`
3. Integrate with ONVIF-compatible cameras for PTZ control on alert
4. Export incident reports to PDF or CSV from the REST API
