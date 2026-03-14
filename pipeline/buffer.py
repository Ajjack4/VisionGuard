"""
VisionGuard AI — Temporal Frame Buffers
Ring buffers that accumulate per-track or per-pair crops into classifier clips.
"""

from __future__ import annotations

from collections import deque
from typing import Dict, Set, Tuple, Union

import numpy as np


TrackKey = Union[int, Tuple[int, int]]


class TemporalBuffer:
    """
    Per-track ring buffer of normalised crop frames.

    Frames are stored as numpy arrays [H, W, C] float32 (ImageNet-normalised).
    Once the buffer has accumulated `buffer_size` frames the clip is ready for
    classification.

    Parameters
    ----------
    buffer_size : int   number of temporal frames per clip (e.g. 16)
    clip_size   : int   spatial resolution expected by classifier (e.g. 112)
    """

    def __init__(self, buffer_size: int = 16, clip_size: int = 112):
        self.buffer_size = buffer_size
        self.clip_size = clip_size
        # keyed by track_id (int) or pair tuple
        self._buffers: Dict[TrackKey, deque] = {}

    # ── Public API ──────────────────────────────────────────────────────────

    def update(self, track_id: TrackKey, crop_frame: np.ndarray) -> None:
        """
        Push a new crop frame into the ring buffer for `track_id`.

        Parameters
        ----------
        track_id   : int or (int, int) track / pair key
        crop_frame : float32 array [H, W, C], already normalised
        """
        if track_id not in self._buffers:
            self._buffers[track_id] = deque(maxlen=self.buffer_size)
        self._buffers[track_id].append(crop_frame)

    def is_ready(self, track_id: TrackKey) -> bool:
        """Return True when the buffer holds exactly buffer_size frames."""
        buf = self._buffers.get(track_id)
        return buf is not None and len(buf) == self.buffer_size

    def get_clip(self, track_id: TrackKey) -> np.ndarray:
        """
        Return a clip tensor suitable for the 3D-CNN classifier.

        Shape: [C, T, H, W]  float32  (PyTorch video convention)
              C=3, T=buffer_size, H=W=clip_size
        """
        buf = self._buffers[track_id]
        # Stack frames: list of [H, W, C] → [T, H, W, C]
        frames = np.stack(list(buf), axis=0)  # [T, H, W, C]
        # Transpose to [C, T, H, W]
        clip = np.transpose(frames, (3, 0, 1, 2))  # [C, T, H, W]
        return clip.astype(np.float32)

    def clear(self, track_id: TrackKey) -> None:
        """Clear the buffer for a specific track."""
        if track_id in self._buffers:
            self._buffers[track_id].clear()

    def cleanup_stale(self, active_track_ids: Set[TrackKey]) -> None:
        """
        Remove buffers for tracks that are no longer active.

        Parameters
        ----------
        active_track_ids : set of currently visible track IDs / pair keys
        """
        stale = [k for k in self._buffers if k not in active_track_ids]
        for k in stale:
            del self._buffers[k]

    def __len__(self) -> int:
        return len(self._buffers)


class PairBuffer(TemporalBuffer):
    """
    Manages merged bounding-box clips keyed by (track_id_1, track_id_2) tuples.

    Inherits the full TemporalBuffer interface; keys are automatically
    normalised so (1, 2) and (2, 1) map to the same buffer.
    """

    @staticmethod
    def _canonical_key(key: Tuple[int, int]) -> Tuple[int, int]:
        return (min(key), max(key))

    def update(self, track_pair: Tuple[int, int], crop_frame: np.ndarray) -> None:
        super().update(self._canonical_key(track_pair), crop_frame)

    def is_ready(self, track_pair: Tuple[int, int]) -> bool:
        return super().is_ready(self._canonical_key(track_pair))

    def get_clip(self, track_pair: Tuple[int, int]) -> np.ndarray:
        return super().get_clip(self._canonical_key(track_pair))

    def clear(self, track_pair: Tuple[int, int]) -> None:
        super().clear(self._canonical_key(track_pair))

    def cleanup_stale(self, active_pairs: Set[Tuple[int, int]]) -> None:
        canonical = {self._canonical_key(p) for p in active_pairs}
        super().cleanup_stale(canonical)
