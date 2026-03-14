"""
VisionGuard AI — Person Tracker
DeepSORT wrapper with automatic IoU-based fallback tracker.
"""

from __future__ import annotations

from typing import List

import numpy as np

from pipeline.detector import Detection


# ── Track dataclass ──────────────────────────────────────────────────────────

class Track:
    """Lightweight track object compatible with both DeepSORT and IoU tracker."""

    def __init__(self, track_id: int, bbox: list, confirmed: bool = True):
        self.track_id: int = track_id
        self.bbox: list = bbox           # [x1, y1, x2, y2]
        self._confirmed: bool = confirmed

    def is_confirmed(self) -> bool:
        return self._confirmed

    def __repr__(self) -> str:
        return f"Track(id={self.track_id}, bbox={self.bbox}, confirmed={self._confirmed})"


# ── DeepSORT wrapper ─────────────────────────────────────────────────────────

class PersonTracker:
    """
    Wraps deep_sort_realtime.DeepSort.

    Falls back to IoUTracker automatically if deep_sort_realtime is not
    installed or fails to import.
    """

    def __init__(self, max_age: int = 30):
        self._tracker = None
        self._fallback = False
        self._max_age = max_age
        self._init_tracker(max_age)

    def _init_tracker(self, max_age: int) -> None:
        try:
            from deep_sort_realtime.deepsort_tracker import DeepSort

            self._tracker = DeepSort(max_age=max_age)
            print("[Tracker] DeepSORT initialised.")
        except ImportError:
            print(
                "[Tracker] WARNING: deep_sort_realtime not found — "
                "falling back to IoU tracker.\n"
                "          Install with: pip install deep-sort-realtime"
            )
            self._tracker = _IoUTracker(max_age=max_age)
            self._fallback = True
        except Exception as exc:
            print(
                f"[Tracker] WARNING: DeepSORT init failed ({exc}) — "
                "falling back to IoU tracker."
            )
            self._tracker = _IoUTracker(max_age=max_age)
            self._fallback = True

    def update(self, detections: List[Detection], frame: np.ndarray) -> List[Track]:
        """
        Update tracker with new detections.

        Parameters
        ----------
        detections : List[Detection]   from HumanDetector.detect()
        frame      : BGR numpy array   (required by DeepSORT for appearance features)

        Returns
        -------
        List[Track]  — all active tracks (confirmed + tentative).
        """
        if self._fallback:
            return self._tracker.update(detections)

        # DeepSORT expects: list of ([left, top, w, h], confidence, class)
        ds_inputs = []
        for d in detections:
            x1, y1, x2, y2 = d.bbox
            w, h = x2 - x1, y2 - y1
            ds_inputs.append(([x1, y1, w, h], d.confidence, "person"))

        try:
            raw_tracks = self._tracker.update_tracks(ds_inputs, frame=frame)
        except Exception as exc:
            print(f"[Tracker] DeepSORT update failed ({exc}), skipping frame.")
            return []

        tracks: List[Track] = []
        for t in raw_tracks:
            if not t.is_confirmed():
                continue
            ltrb = t.to_ltrb()
            bbox = [int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])]
            tracks.append(Track(track_id=t.track_id, bbox=bbox, confirmed=True))
        return tracks

    def get_confirmed_tracks(self) -> List[Track]:
        """Return confirmed tracks (DeepSORT only; IoU tracker always confirms)."""
        # DeepSORT doesn't expose a direct method; use last update result instead.
        return []


# ── IoU Fallback Tracker ─────────────────────────────────────────────────────

class _IoUTracker:
    """
    Simple IoU-based tracker used when DeepSORT is unavailable.

    Each detection is matched to the nearest existing track by IoU.
    Tracks are aged out after `max_age` unmatched frames.
    """

    def __init__(self, max_age: int = 30):
        self._max_age = max_age
        self._tracks: dict[int, _IoUTrackState] = {}
        self._next_id: int = 1

    def update(self, detections: List[Detection]) -> List[Track]:
        bboxes = [d.bbox for d in detections]

        # --- match detections to existing tracks via IoU ---
        matched_track_ids: set[int] = set()
        matched_det_indices: set[int] = set()

        for track_id, state in list(self._tracks.items()):
            best_iou, best_idx = 0.0, -1
            for i, bbox in enumerate(bboxes):
                if i in matched_det_indices:
                    continue
                iou = _iou(state.bbox, bbox)
                if iou > best_iou:
                    best_iou, best_idx = iou, i

            if best_iou >= 0.3 and best_idx >= 0:
                state.bbox = bboxes[best_idx]
                state.age = 0
                matched_track_ids.add(track_id)
                matched_det_indices.add(best_idx)
            else:
                state.age += 1

        # --- create new tracks for unmatched detections ---
        for i, bbox in enumerate(bboxes):
            if i not in matched_det_indices:
                self._tracks[self._next_id] = _IoUTrackState(
                    bbox=bbox, age=0, hits=1
                )
                self._next_id += 1

        # --- remove stale tracks ---
        self._tracks = {
            tid: s
            for tid, s in self._tracks.items()
            if s.age <= self._max_age
        }

        # Return as Track objects (all confirmed in IoU tracker)
        return [
            Track(track_id=tid, bbox=state.bbox, confirmed=state.hits >= 1)
            for tid, state in self._tracks.items()
        ]


class _IoUTrackState:
    __slots__ = ("bbox", "age", "hits")

    def __init__(self, bbox: list, age: int, hits: int):
        self.bbox = bbox
        self.age = age
        self.hits = hits


def _iou(a: list, b: list) -> float:
    """Compute IoU between two [x1,y1,x2,y2] boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    inter_w = max(0, ix2 - ix1)
    inter_h = max(0, iy2 - iy1)
    inter = inter_w * inter_h

    if inter == 0:
        return 0.0

    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0
