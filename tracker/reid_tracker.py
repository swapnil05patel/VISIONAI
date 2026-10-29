"""
reid_tracker.py
---------------
PersonReIDTracker — orchestrates the full per-frame tracking pipeline.

Pipeline per frame (in order):
  1.  Predict all active tracks (Kalman forward pass).
  2.  Mark all tracks as missed for this frame.
  3.  Retire tracks whose predicted center has exited the frame boundary.
      Retired tracks are saved to the gallery before deletion.
  4.  Build N x M cost matrix from configurable IoU / ReID / direction weights.
  5.  Run Hungarian algorithm for globally optimal track-detection assignment.
  6.  Gate each accepted assignment:
          Accept if: IoU >= iou_thr  OR  (strict ReID + spatial gates pass)
  7.  Update all accepted (track, detection) pairs.
  8.  ByteTrack-style second stage: associate remaining tracks with low-conf
      detections using IoU only (before lost-track fallback / gallery recovery).
  9.  Fallback IoU attach for unmatched high-conf detections.
  10. Gallery query for still-unmatched detections (with revival confirmation).
  11. Create new tracks for truly unmatched high-conf detections.
  12. Retire tracks that have exceeded MAX_AGE. Save them to gallery.
"""

import math
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from scipy.optimize import linear_sum_assignment

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from tracker.config import (
    DEVICE,
    NUM_PARTS,
    IOU_THR,
    REID_STRICT_THRESH,
    REID_REATTACH_MAX_AGE,
    REID_CENTER_MAX_DIST_BASE,
    VELOCITY_GATE_MULTIPLIER,
    REID_AREA_RATIO_MAX,
    MATCH_WEIGHT_IOU,
    MATCH_WEIGHT_REID,
    MATCH_WEIGHT_DIRECTION,
    DIRECTION_SPEED_THRESH,
    FALLBACK_IOU_FOR_NEW,
    FALLBACK_CENTER_MAX_DIST_BASE,
    MAX_FALLBACK_MISSES,
    MAX_AGE,
    MIN_HITS_TO_CONFIRM,
    EXIT_MIN_MISSES,
    HARD_RETIRE_ON_EXIT,
    GALLERY_EVICT_INTERVAL,
    ENABLE_BYTETRACK_STAGE2,
    STAGE2_IOU_THR,
    ENABLE_GALLERY,
    ENABLE_STATIC_PRUNING,
    REVIVAL_CONFIRMATION_FRAMES,
    REID_BACKBONE,
    USE_PART_EMBEDDING,
)
from tracker.embedder import PartEmbedder
from tracker.track import Track
from tracker.gallery import TrackGallery
from tracker.utils import iou, cosine_distance_vec, center_of, box_area, l2_normalize_vec


class PersonReIDTracker:
    """
    Full person tracking pipeline combining detections, appearance embeddings,
    Kalman filtering, Hungarian assignment, optional ByteTrack stage-2, and a
    persistent Track Gallery for long-term re-identification.
    """

    def __init__(
        self,
        device: str = DEVICE,
        frame_size: tuple = (1920, 1080),
        fps: float = 30.0,
        iou_thr: float = IOU_THR,
        reid_strict_thresh: float = REID_STRICT_THRESH,
        max_age: int = MAX_AGE,
        min_hits: int = MIN_HITS_TO_CONFIRM,
        embedder: Optional[PartEmbedder] = None,
        backbone: str = REID_BACKBONE,
        use_part_embedding: bool = USE_PART_EMBEDDING,
        enable_gallery: bool = ENABLE_GALLERY,
        enable_bytetrack_stage2: bool = ENABLE_BYTETRACK_STAGE2,
        enable_static_pruning: bool = ENABLE_STATIC_PRUNING,
        match_weight_iou: float = MATCH_WEIGHT_IOU,
        match_weight_reid: float = MATCH_WEIGHT_REID,
        match_weight_direction: float = MATCH_WEIGHT_DIRECTION,
        stage2_iou_thr: float = STAGE2_IOU_THR,
        revival_confirmation_frames: int = REVIVAL_CONFIRMATION_FRAMES,
    ):
        self.device = device
        self.frame_size = frame_size
        self.fps = fps
        self.dt = 1.0 / (fps if fps > 0 else 30.0)

        self.iou_thr = iou_thr
        self.reid_strict_thresh = reid_strict_thresh
        self.max_age = max_age
        self.min_hits = min_hits
        self.enable_gallery = enable_gallery
        self.enable_bytetrack_stage2 = enable_bytetrack_stage2
        self.enable_static_pruning = enable_static_pruning
        self.match_weight_iou = match_weight_iou
        self.match_weight_reid = match_weight_reid
        self.match_weight_direction = match_weight_direction
        self.stage2_iou_thr = stage2_iou_thr
        self.revival_confirmation_frames = max(1, int(revival_confirmation_frames))

        if embedder is not None:
            self.embedder = embedder
        else:
            self.embedder = PartEmbedder(
                device=device,
                num_parts=NUM_PARTS,
                backbone=backbone,
                use_part_embedding=use_part_embedding,
            )
        self.tracks: Dict[int, Track] = {}
        self.next_id = 1
        self.gallery = TrackGallery()
        self._frames_processed = 0
        self._pending_revivals: Dict[int, int] = {}

    def reset(self):
        """Clear all tracks, gallery, and reset the ID counter."""
        self.tracks = {}
        self.next_id = 1
        self.gallery = TrackGallery()
        self._frames_processed = 0
        self._pending_revivals = {}

    def step(
        self,
        detections: List[List[float]],
        parts_features: List[List[np.ndarray]],
        frame_idx: int,
        low_conf_detections: Optional[List[List[float]]] = None,
    ) -> Dict[int, Track]:
        """
        Process one frame and update all tracks.

        Args:
            detections:     High-confidence boxes [[x1,y1,x2,y2], ...].
            parts_features: Part embeddings per high-confidence detection.
            frame_idx:      Current frame number (1-indexed).
            low_conf_detections: Optional ByteTrack stage-2 boxes (IoU only).
        """
        self._frames_processed += 1
        current_time = time.time()
        low_conf_detections = low_conf_detections or []

        for tid in list(self.tracks.keys()):
            tr = self.tracks[tid]
            tr.predict()
            tr.mark_missed()

            should_retire = (
                (HARD_RETIRE_ON_EXIT and tr.misses_outside > 0) or
                (not HARD_RETIRE_ON_EXIT and tr.misses_outside >= EXIT_MIN_MISSES)
            )
            if should_retire:
                self._retire_track(tid, frame_idx)
                continue

            if self.enable_static_pruning and tr.is_static:
                del self.tracks[tid]
                self._pending_revivals.pop(tid, None)
                continue

        track_ids = list(self.tracks.keys())
        N = len(track_ids)
        M = len(detections)

        if N == 0 and M == 0:
            self._finish_frame(current_time, frame_idx)
            return self.tracks

        cost = np.ones((max(N, 1), max(M, 1)), dtype=np.float32)
        if N > 0 and M > 0:
            cost = np.ones((N, M), dtype=np.float32)
            for i, tid in enumerate(track_ids):
                tr = self.tracks[tid]
                tr_emb = tr.get_average_embedding()
                tr_vx = float(tr.x[4])
                tr_vy = float(tr.x[5])
                speed = math.hypot(tr_vx, tr_vy)
                tr_cx, tr_cy = center_of(tr.smoothed_box)

                for j in range(M):
                    det_box = detections[j]
                    det_parts = parts_features[j] if j < len(parts_features) else []
                    iou_score = iou(tr.box, det_box)
                    iou_cost = 1.0 - iou_score

                    reid_cost = 1.0
                    if tr_emb is not None and det_parts:
                        dists = [cosine_distance_vec(tr_emb, p) for p in det_parts]
                        reid_cost = float(min(dists)) if dists else 1.0

                    direction_cost = 0.0
                    if speed >= DIRECTION_SPEED_THRESH:
                        det_cx, det_cy = center_of(det_box)
                        dx = det_cx - tr_cx
                        dy = det_cy - tr_cy
                        dist = math.hypot(dx, dy)
                        if dist > 0:
                            cos_angle = (tr_vx * dx + tr_vy * dy) / (speed * dist)
                            direction_cost = (1.0 - cos_angle) / 2.0

                    cost[i, j] = (
                        self.match_weight_iou * iou_cost
                        + self.match_weight_reid * reid_cost
                        + self.match_weight_direction * direction_cost
                    )

        matches = []
        unmatched_track_ii = list(range(N))
        unmatched_det_jj = list(range(M))

        if N > 0 and M > 0:
            row_ind, col_ind = linear_sum_assignment(cost)
            assigned_tracks = set()
            assigned_dets = set()

            for r, c in zip(row_ind, col_ind):
                tid = track_ids[r]
                tr = self.tracks[tid]
                det_box = detections[c]
                det_parts = parts_features[c] if c < len(parts_features) else []

                iou_score = iou(tr.box, det_box)
                tr_emb = tr.get_average_embedding()
                best_reid = 1.0
                if tr_emb is not None and det_parts:
                    dists = [cosine_distance_vec(tr_emb, p) for p in det_parts]
                    best_reid = float(min(dists)) if dists else 1.0

                tcx, tcy = center_of(tr.box)
                dcx, dcy = center_of(det_box)
                center_dist = math.hypot(tcx - dcx, tcy - dcy)

                speed = math.hypot(float(tr.x[4]), float(tr.x[5]))
                dynamic_max_dist = REID_CENTER_MAX_DIST_BASE + (
                    VELOCITY_GATE_MULTIPLIER * speed * tr.time_since_update
                )

                tarea = max(1.0, box_area(tr.box))
                darea = max(1.0, box_area(det_box))
                area_ratio = max(tarea / darea, darea / tarea)
                recent_enough = tr.time_since_update <= REID_REATTACH_MAX_AGE

                reid_ok = (
                    tr_emb is not None
                    and det_parts
                    and recent_enough
                    and best_reid <= self.reid_strict_thresh
                    and center_dist <= dynamic_max_dist
                    and area_ratio <= REID_AREA_RATIO_MAX
                )

                if iou_score >= self.iou_thr or reid_ok:
                    matches.append((r, c))
                    assigned_tracks.add(r)
                    assigned_dets.add(c)

            unmatched_track_ii = [i for i in range(N) if i not in assigned_tracks]
            unmatched_det_jj = [j for j in range(M) if j not in assigned_dets]

        for r, c in matches:
            self._update_track(track_ids[r], detections[c], parts_features[c], frame_idx)

        if self.enable_bytetrack_stage2 and low_conf_detections:
            unmatched_track_ii = self._bytetrack_stage2(
                track_ids,
                unmatched_track_ii,
                low_conf_detections,
                frame_idx,
            )

        fallback_attached = set()
        for di in list(unmatched_det_jj):
            det_box = detections[di]
            best_tid = None
            best_iou = 0.0

            for tid, tr in self.tracks.items():
                if tr.time_since_update == 0:
                    continue
                if tr.time_since_update > MAX_FALLBACK_MISSES:
                    continue
                if tr.retired:
                    continue
                i_score = iou(tr.smoothed_box, det_box)
                if i_score > best_iou:
                    best_iou = i_score
                    best_tid = tid

            if best_tid is not None and best_iou >= FALLBACK_IOU_FOR_NEW:
                tr = self.tracks[best_tid]
                tcx, tcy = center_of(tr.smoothed_box)
                dcx, dcy = center_of(det_box)
                speed = math.hypot(float(tr.x[4]), float(tr.x[5]))
                dynamic_fallback_dist = FALLBACK_CENTER_MAX_DIST_BASE + (
                    VELOCITY_GATE_MULTIPLIER * speed * tr.time_since_update
                )
                if math.hypot(tcx - dcx, tcy - dcy) <= dynamic_fallback_dist:
                    det_parts = parts_features[di] if di < len(parts_features) else []
                    self._update_track(best_tid, det_box, det_parts, frame_idx)
                    fallback_attached.add(di)

        unmatched_det_jj = [d for d in unmatched_det_jj if d not in fallback_attached]

        gallery_revived = set()
        if self.enable_gallery:
            for di in list(unmatched_det_jj):
                det_box = detections[di]
                det_parts = parts_features[di] if di < len(parts_features) else []
                if not det_parts:
                    continue

                valid_parts = [p for p in det_parts if p is not None and np.isfinite(p).all()]
                if not valid_parts:
                    continue
                arr = np.stack(valid_parts, axis=0)
                det_embed = l2_normalize_vec(np.mean(arr, axis=0))

                matched_gallery_tid = self.gallery.query(det_box, det_embed, current_time)
                if matched_gallery_tid is None:
                    continue
                if matched_gallery_tid in self.tracks:
                    continue

                revived = Track(
                    init_box=det_box,
                    part_feats=det_parts,
                    track_id=matched_gallery_tid,
                    frame_idx=frame_idx,
                    dt=self.dt,
                    frame_size=self.frame_size,
                )
                if self.revival_confirmation_frames <= 1:
                    revived.hits = max(self.min_hits, revived.hits)
                    revived.confirmed = True
                    revived.status = "Confirmed"
                    self.gallery.remove(matched_gallery_tid)
                else:
                    revived.confirmed = False
                    revived.status = "PendingRevival"
                    self.gallery.lock(matched_gallery_tid)
                    self._pending_revivals[matched_gallery_tid] = 1

                self.tracks[matched_gallery_tid] = revived
                if matched_gallery_tid >= self.next_id:
                    self.next_id = matched_gallery_tid + 1
                gallery_revived.add(di)

        unmatched_det_jj = [d for d in unmatched_det_jj if d not in gallery_revived]

        for tid in list(self._pending_revivals):
            tr = self.tracks.get(tid)
            if tr is None or tr.time_since_update > 0:
                self._abort_pending_revival(tid)

        for di in unmatched_det_jj:
            det_box = detections[di]
            det_parts = parts_features[di] if di < len(parts_features) else []

            overlapping = any(
                tr.confirmed and iou(tr.smoothed_box, det_box) > 0.45
                for tr in self.tracks.values()
            )
            if overlapping:
                continue

            new_track = Track(
                init_box=det_box,
                part_feats=det_parts,
                track_id=self.next_id,
                frame_idx=frame_idx,
                dt=self.dt,
                frame_size=self.frame_size,
            )
            self.tracks[self.next_id] = new_track
            self.next_id += 1

        self._finish_frame(current_time, frame_idx)
        return self.tracks

    def _update_track(self, tid: int, det_box, det_parts, frame_idx: int):
        self.tracks[tid].update(det_box, det_parts, frame_idx)
        if tid in self._pending_revivals:
            self._pending_revivals[tid] += 1
            if self._pending_revivals[tid] >= self.revival_confirmation_frames:
                tr = self.tracks[tid]
                tr.confirmed = True
                tr.status = "Confirmed"
                self.gallery.remove(tid)
                del self._pending_revivals[tid]

    def _bytetrack_stage2(
        self,
        track_ids: List[int],
        unmatched_track_ii: List[int],
        low_conf_detections: List[List[float]],
        frame_idx: int,
    ) -> List[int]:
        """Associate remaining tracks with low-confidence detections using IoU."""
        if not unmatched_track_ii or not low_conf_detections:
            return unmatched_track_ii

        n_u = len(unmatched_track_ii)
        n_l = len(low_conf_detections)
        cost = np.ones((n_u, n_l), dtype=np.float32)
        for i, ti in enumerate(unmatched_track_ii):
            tr = self.tracks[track_ids[ti]]
            for j, det_box in enumerate(low_conf_detections):
                cost[i, j] = 1.0 - iou(tr.box, det_box)

        row_ind, col_ind = linear_sum_assignment(cost)
        still_unmatched = set(unmatched_track_ii)
        used_low = set()
        for r, c in zip(row_ind, col_ind):
            if c in used_low:
                continue
            iou_score = 1.0 - float(cost[r, c])
            if iou_score < self.stage2_iou_thr:
                continue
            ti = unmatched_track_ii[r]
            tid = track_ids[ti]
            self._update_track(tid, low_conf_detections[c], [], frame_idx)
            still_unmatched.discard(ti)
            used_low.add(c)
        return sorted(still_unmatched)

    def _abort_pending_revival(self, tid: int):
        self._pending_revivals.pop(tid, None)
        self.gallery.unlock(tid)
        self.tracks.pop(tid, None)

    def _retire_track(self, tid: int, frame_idx: int):
        tr = self.tracks[tid]
        tr.exit_time = datetime.now()
        if tid in self._pending_revivals:
            self._abort_pending_revival(tid)
            return
        if self.enable_gallery:
            self.gallery.save(tr, frame_idx)
        del self.tracks[tid]

    def _finish_frame(self, current_time: float, frame_idx: int):
        for tid in [t for t, tr in list(self.tracks.items()) if tr.time_since_update > self.max_age]:
            self._retire_track(tid, frame_idx)

        if self.enable_gallery and self._frames_processed % GALLERY_EVICT_INTERVAL == 0:
            self.gallery.evict_expired(current_time)
