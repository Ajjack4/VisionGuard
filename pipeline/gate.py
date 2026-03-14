"""
VisionGuard AI — Proximity Gate
Gates the expensive classifier: only fires when 2+ people are close together.

Groups people using connected-components so any number of people in the same
cluster share one classifier, instead of the old pair-of-2 constraint.
"""

from __future__ import annotations

import math
from typing import List, Tuple

from pipeline.tracker import Track


class ProximityGate:
    """
    Finds groups of tracked people that are in close proximity.

    Two people are considered proximate when the distance between their
    bounding-box centroids is less than:

        alpha × max(diagonal(bbox_1), diagonal(bbox_2))

    Connected-components: if A is close to B and B is close to C, then A, B
    and C form one group — the classifier sees all three at once.

    Parameters
    ----------
    alpha      : float   proximity multiplier (default 2.5)
    min_people : int     minimum people in scene before any gate check
    """

    def __init__(self, alpha: float = 2.5, min_people: int = 2):
        self.alpha = alpha
        self.min_people = min_people
        self._groups: List[List[Track]] = []

    # ── Public API ──────────────────────────────────────────────────────────

    def evaluate_groups(self, tracks: List[Track]) -> List[List[Track]]:
        """
        Return connected components of proximate people.

        Each component (group) has >= 2 confirmed tracks.
        People who are not close to anyone are excluded.

        Returns
        -------
        List[List[Track]] — each inner list is one group, sorted by track_id.
        """
        confirmed = [t for t in tracks if t.is_confirmed()]
        self._groups = []

        if len(confirmed) < self.min_people:
            return self._groups

        n = len(confirmed)

        # Build adjacency graph
        adj: List[set] = [set() for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                if self._are_proximate(confirmed[i].bbox, confirmed[j].bbox):
                    adj[i].add(j)
                    adj[j].add(i)

        # Connected components via DFS
        visited: set = set()
        for start in range(n):
            if start in visited:
                continue
            component: List[int] = []
            stack = [start]
            while stack:
                node = stack.pop()
                if node in visited:
                    continue
                visited.add(node)
                component.append(node)
                stack.extend(adj[node] - visited)
            if len(component) >= 2:
                group = sorted(
                    [confirmed[k] for k in component],
                    key=lambda t: t.track_id,
                )
                self._groups.append(group)

        return self._groups

    def evaluate(self, tracks: List[Track]) -> List[Tuple[Track, Track]]:
        """
        Backward-compatible: return pairs derived from groups.

        Calls evaluate_groups() then greedily flattens each group into
        non-overlapping nearest-first pairs (each person in at most one pair).
        """
        groups = self.evaluate_groups(tracks)
        pairs: List[Tuple[Track, Track]] = []

        for group in groups:
            candidates: List[Tuple[float, Track, Track]] = []
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    dist = self._centroid_distance(group[i].bbox, group[j].bbox)
                    candidates.append((dist, group[i], group[j]))
            candidates.sort(key=lambda x: x[0])
            used: set = set()
            for _, t1, t2 in candidates:
                if t1.track_id not in used and t2.track_id not in used:
                    pairs.append((t1, t2))
                    used.add(t1.track_id)
                    used.add(t2.track_id)

        return pairs

    @property
    def gate_open(self) -> bool:
        """True if at least one proximate group was found in the last call."""
        return bool(self._groups)

    # ── Geometry helpers ────────────────────────────────────────────────────

    def get_group_bbox(self, group: List[Track]) -> list:
        """Union bounding box covering all tracks in *group*."""
        x1 = min(t.bbox[0] for t in group)
        y1 = min(t.bbox[1] for t in group)
        x2 = max(t.bbox[2] for t in group)
        y2 = max(t.bbox[3] for t in group)
        return [x1, y1, x2, y2]

    def get_merged_bbox(self, bbox1: list, bbox2: list) -> list:
        """Backward-compat: union of two bounding boxes."""
        return [
            min(bbox1[0], bbox2[0]),
            min(bbox1[1], bbox2[1]),
            max(bbox1[2], bbox2[2]),
            max(bbox1[3], bbox2[3]),
        ]

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
        return dist < self.alpha * max_diag
