"""
VisionGuard AI — Human Detector
YOLOv8 wrapper: detects people, returns bounding boxes + crops.
"""

from __future__ import annotations

from collections import namedtuple
from typing import List

import cv2
import numpy as np

Detection = namedtuple("Detection", ["bbox", "confidence", "class_id"])
# bbox: [x1, y1, x2, y2] in pixel coordinates (integers)


def _select_device() -> str:
    """Auto-select the best available compute device."""
    import torch

    if torch.cuda.is_available():
        print("[Detector] Using device: CUDA")
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("[Detector] Using device: MPS (Apple Silicon)")
        return "mps"
    print(
        "[Detector] CUDA not available — running on CPU. "
        "Expect ~2–5 fps on older hardware. "
        "A GPU is strongly recommended for real-time performance."
    )
    return "cpu"


class HumanDetector:
    """
    YOLOv8 person detector.

    Parameters
    ----------
    model_path : str
        YOLO model file, e.g. "yolov8n.pt".  Downloaded automatically on first
        use if not present.
    conf_threshold : float
        Minimum detection confidence (0–1).
    device : str | None
        "cuda", "mps", "cpu", or None (auto).
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        conf_threshold: float = 0.45,
        device: str | None = None,
    ):
        from ultralytics import YOLO

        self.conf_threshold = conf_threshold
        self.device = device or _select_device()

        self._model = YOLO(model_path)
        # Warm-up to avoid latency on first real frame
        dummy = np.zeros((64, 64, 3), dtype=np.uint8)
        self._model.predict(
            dummy,
            classes=[0],
            conf=conf_threshold,
            device=self.device,
            verbose=False,
        )
        print(f"[Detector] YOLOv8 ready — model={model_path} device={self.device}")

    # ── Public API ──────────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Run YOLOv8 on *frame* (BGR numpy array) and return person detections.

        Returns
        -------
        List[Detection]  where Detection.bbox = [x1, y1, x2, y2] (int pixels).
        """
        results = self._model.predict(
            frame,
            classes=[0],           # person class only
            conf=self.conf_threshold,
            device=self.device,
            verbose=False,
        )
        detections: List[Detection] = []
        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                xyxy = boxes.xyxy[i].cpu().numpy().astype(int)
                conf = float(boxes.conf[i].cpu().numpy())
                cls = int(boxes.cls[i].cpu().numpy())
                if cls == 0:  # person
                    detections.append(
                        Detection(
                            bbox=xyxy.tolist(),  # [x1, y1, x2, y2]
                            confidence=conf,
                            class_id=cls,
                        )
                    )
        return detections

    def extract_crop(
        self,
        frame: np.ndarray,
        bbox: list,
        padding: float = 0.15,
        clip_size: int = 112,
    ) -> np.ndarray:
        """
        Crop and normalise a region from *frame*.

        Parameters
        ----------
        frame     : BGR numpy array [H, W, 3]
        bbox      : [x1, y1, x2, y2] in pixel coords
        padding   : fractional padding added on each side
        clip_size : output spatial resolution (square)

        Returns
        -------
        Normalised float32 array [clip_size, clip_size, 3] with values in
        [0, 1] after ImageNet normalisation.
        """
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox

        # Compute padding in pixels
        bw = x2 - x1
        bh = y2 - y1
        px = int(bw * padding)
        py = int(bh * padding)

        # Clamp to frame bounds
        x1c = max(0, x1 - px)
        y1c = max(0, y1 - py)
        x2c = min(w, x2 + px)
        y2c = min(h, y2 + py)

        crop = frame[y1c:y2c, x1c:x2c]
        if crop.size == 0:
            crop = frame  # fallback: use the whole frame

        # Resize to clip_size × clip_size
        crop = cv2.resize(crop, (clip_size, clip_size), interpolation=cv2.INTER_LINEAR)

        # BGR → RGB, normalise to [0, 1]
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        # ImageNet normalisation
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        crop = (crop - mean) / std

        return crop  # [clip_size, clip_size, 3]  float32
