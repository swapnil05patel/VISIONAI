"""
gallery.py
----------
Track Gallery for persistent person re-identification.

When a track is retired (due to MAX_AGE or frame boundary exit), its
appearance embedding and spatial metadata are saved here. When a new
detection cannot be matched to any active track, the gallery is queried
to check if this person was seen before.

Retrieval uses a gated pipeline to avoid expensive cosine computation
on irrelevant entries:

    Gate 1 (TTL)     : Skip entries older than GALLERY_TTL_SECONDS.
    Gate 2 (Spatial) : Optional; skip entries whose last known center is too far.
    Gate 3 (Area)    : Skip entries with too different a bounding box area.
    Gate 4 (Cosine)  : Compare mean, EMA, or exemplar embeddings.

Locked entries (pending revival confirmation) are skipped by query.
"""

import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from tracker.config import (
    GALLERY_COSINE_THRESH,
    GALLERY_TTL_SECONDS,
    GALLERY_MAX_SIZE,
    GALLERY_MIN_HITS,
    GALLERY_USE_SPATIAL_GATE,
    GALLERY_SPATIAL_MAX_DIST,
    GALLERY_AREA_RATIO_MAX,
    GALLERY_REPRESENTATION,
    GALLERY_MAX_EXEMPLARS,
    GALLERY_EMA_ALPHA,
)
from tracker.utils import cosine_distance_vec, center_of, box_area


class TrackGallery:
    """
    Stores appearance representations of retired tracks and provides
    gated retrieval for new detections.
    """

    def __init__(
        self,
        cosine_thresh: float = GALLERY_COSINE_THRESH,
        use_spatial_gate: bool = GALLERY_USE_SPATIAL_GATE,
        spatial_max_dist: float = GALLERY_SPATIAL_MAX_DIST,
        area_ratio_max: float = GALLERY_AREA_RATIO_MAX,
        representation: str = GALLERY_REPRESENTATION,
    ):
        self._entries: Dict[int, dict] = {}
        self._locked: set = set()
        self.cosine_thresh = cosine_thresh
        self.use_spatial_gate = use_spatial_gate
        self.spatial_max_dist = spatial_max_dist
        self.area_ratio_max = area_ratio_max
        self.representation = representation

    def save(self, track, frame_idx: int) -> bool:
        """
        Save a retiring track to the gallery.

        Skips the track if it was too short-lived to be reliable.
        Evicts the oldest entry if the gallery is at capacity.
        """
        if track.hits < GALLERY_MIN_HITS:
            return False

        embedding = None
        exemplars: List[np.ndarray] = []
        if self.representation == "ema":
            embedding = track.get_ema_embedding(alpha=GALLERY_EMA_ALPHA)
        elif self.representation == "exemplars":
            exemplars = track.get_exemplars(max_exemplars=GALLERY_MAX_EXEMPLARS)
            embedding = track.get_average_embedding()
        else:
            embedding = track.get_weighted_embedding()

        if embedding is None and not exemplars:
            return False

        cx, cy = center_of(track.smoothed_box)
        area = box_area(track.smoothed_box)

        entry = {
            "embedding": embedding,
            "exemplars": exemplars,
            "last_position": (cx, cy),
            "last_area": max(1.0, area),
            "retired_frame": frame_idx,
            "retired_time": time.time(),
            "hit_count": track.hits,
        }

        if len(self._entries) >= GALLERY_MAX_SIZE and track.id not in self._entries:
            self._evict_oldest()

        self._entries[track.id] = entry
        self._locked.discard(track.id)
        return True

    def query(
        self,
        det_box: List[float],
        det_embed: np.ndarray,
        current_time: float,
    ) -> Optional[int]:
        """
        Find the best matching gallery entry for a new detection.
        """
        best_tid = None
        best_dist = self.cosine_thresh

        det_cx, det_cy = center_of(det_box)
        det_area = max(1.0, box_area(det_box))

        for tid, entry in self._entries.items():
            if tid in self._locked:
                continue

            if current_time - entry["retired_time"] > GALLERY_TTL_SECONDS:
                continue

            if self.use_spatial_gate:
                ecx, ecy = entry["last_position"]
                if math.hypot(det_cx - ecx, det_cy - ecy) > self.spatial_max_dist:
                    continue

            area_ratio = max(det_area, entry["last_area"]) / min(det_area, entry["last_area"])
            if area_ratio > self.area_ratio_max:
                continue

            dist = self._appearance_distance(entry, det_embed)
            if dist < best_dist:
                best_dist = dist
                best_tid = tid

        return best_tid

    def lock(self, tid: int):
        """Prevent additional queries from matching a pending revival."""
        if tid in self._entries:
            self._locked.add(tid)

    def unlock(self, tid: int):
        self._locked.discard(tid)

    def remove(self, tid: int):
        """Remove a gallery entry after it has been successfully re-attached."""
        self._entries.pop(tid, None)
        self._locked.discard(tid)

    def evict_expired(self, current_time: float):
        """Remove all entries older than GALLERY_TTL_SECONDS."""
        expired_ids = [
            tid for tid, entry in self._entries.items()
            if current_time - entry["retired_time"] > GALLERY_TTL_SECONDS
        ]
        for tid in expired_ids:
            self.remove(tid)

    def __len__(self) -> int:
        return len(self._entries)

    def _appearance_distance(self, entry: dict, det_embed: np.ndarray) -> float:
        exemplars = entry.get("exemplars") or []
        if self.representation == "exemplars" and exemplars:
            return min(cosine_distance_vec(ex, det_embed) for ex in exemplars)
        embedding = entry.get("embedding")
        if embedding is None:
            return 1.0
        return cosine_distance_vec(embedding, det_embed)

    def _evict_oldest(self):
        """Remove the unlocked entry with the earliest retirement time."""
        candidates = [tid for tid in self._entries if tid not in self._locked]
        if not candidates:
            candidates = list(self._entries.keys())
        if not candidates:
            return
        oldest = min(candidates, key=lambda k: self._entries[k]["retired_time"])
        self.remove(oldest)
