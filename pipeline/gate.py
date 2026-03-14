"""
VisionGuard AI — Proximity Gate
Gates the expensive classifier: only fires when 2+ people are close together.
"""

from __future__ import annotations

import math
from typing import List, Tuple

from pipeline.tracker import Track


class ProximityGate:
    """
    Evaluates whether any pair of tracked people are in close proximity.

    Two people are considered proximate when the distance between their
    bounding-box centroids is less than:

        alpha × max(diagonal(bbox_1), diagonal(bbox_2))

    This makes the threshold scale-invariant: people far from the camera
    (small bounding boxes) require less absolute distance to trigger the gate.

    Parameters
    ----------
    alpha      : float   proximity multiplier (default 2.5)
    min_people : int     minimum number of people in the scene to even check
    """

    def __init__(self, alpha: float = 2.5, min_people: int = 2):
        self.alpha = alpha
        self.min_people = min_people
        self._proximate_pairs: List[Tuple[Track, Track]] = []

    # ── Public API ──────────────────────────────────────────────────────────

    def evaluate(self, tracks: List[Track]) -> List[Tuple[Track, Track]]:
        """
        Check all confirmed track pairs for proximity.

        Returns
        -------
        List of (Track, Track) tuples that are proximate.
        Empty list when the gate is closed.
        """
        confirmed = [t for t in tracks if t.is_confirmed()]
        self._proximate_pairs = []

        if len(confirmed) < self.min_people:
            return self._proximate_pairs

        for i in range(len(confirmed)):
            for j in range(i + 1, len(confirmed)):
                t1, t2 = confirmed[i], confirmed[j]
                if self._are_proximate(t1.bbox, t2.bbox):
                    self._proximate_pairs.append((t1, t2))

        return self._proximate_pairs

    @property
    def gate_open(self) -> bool:
        """True if at least one proximate pair was found in the last evaluate()."""
        return len(self._proximate_pairs) > 0

    # ── Geometry helpers ────────────────────────────────────────────────────

    def _centroid(self, bbox: list) -> Tuple[float, float]:
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def _centroid_distance(self, bbox1: list, bbox2: list) -> float:
        cx1, cy1 = self._centroid(bbox1)
        cx2, cy2 = self._centroid(bbox2)
        return math.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)

    def _bounding_box_diagonal(self, bbox: list) -> float:
        x1, y1, x2, y2 = bbox
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    def _are_proximate(self, bbox1: list, bbox2: list) -> bool:
        dist = self._centroid_distance(bbox1, bbox2)
        max_diag = max(
            self._bounding_box_diagonal(bbox1),
            self._bounding_box_diagonal(bbox2),
        )
        threshold = self.alpha * max_diag
        return dist < threshold

    def get_merged_bbox(self, bbox1: list, bbox2: list) -> list:
        """
        Return the union (merged) bounding box covering both inputs.

        Returns [x1, y1, x2, y2] as integers.
        """
        x1 = min(bbox1[0], bbox2[0])
        y1 = min(bbox1[1], bbox2[1])
        x2 = max(bbox1[2], bbox2[2])
        y2 = max(bbox1[3], bbox2[3])
        return [x1, y1, x2, y2]
