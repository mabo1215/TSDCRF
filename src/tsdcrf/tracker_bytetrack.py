from __future__ import annotations
from dataclasses import dataclass
from typing import List
import numpy as np


@dataclass(frozen=True)
class ByteTrackConfig:
    """
    Field names kept for config.yaml compatibility;
    implementation is a lightweight IoU multi-object tracker (SORT-like),
    no longer depending on YOLOX's built-in ByteTrack.
    """

    track_thresh: float = 0.5  # Confidence threshold; detections below this are not tracked
    track_buffer: int = 30  # Max frames to keep a track without match before removal
    match_thresh: float = 0.8  # IoU match threshold
    frame_rate: int = 30  # Kept for semantics only


class _Track:
    def __init__(self, track_id: int, bbox_xyxy: np.ndarray, score: float, cls_id: int, frame_id: int):
        self.track_id = int(track_id)
        self.bbox = bbox_xyxy.astype(np.float32)
        self.score = float(score)
        self.cls_id = int(cls_id)
        self.last_frame = int(frame_id)


class ByteTrackWrapper:
    """
    Lightweight multi-object tracking:
      - Per-frame input: detections (xyxy), scores, class ids
      - IoU matching between detections and existing tracks
      - IoU >= match_thresh: same object, reuse track_id
      - Unmatched detections get new track_id
      - Tracks unmatched for track_buffer frames are removed

    Output format unchanged: list[dict] with track_id, bbox(xyxy), score, cls_id
    so main.py and privacy/smoothing modules need no changes.
    """

    def __init__(self, cfg: ByteTrackConfig):
        self.cfg = cfg
        self.tracks: List[_Track] = []
        self.next_id: int = 1
        self.frame_id: int = 0

    def _cleanup_dead_tracks(self):
        """Remove tracks that have not been updated for too long."""
        alive = []
        for t in self.tracks:
            if self.frame_id - t.last_frame <= self.cfg.track_buffer:
                alive.append(t)
        self.tracks = alive

    def update(self, bboxes_xyxy: np.ndarray, scores: np.ndarray, cls_ids: np.ndarray) -> list[dict]:
        """
        Input:
          - bboxes_xyxy: [N,4], xyxy format
          - scores: [N]
          - cls_ids: [N]

        Returns:
          list[dict(track_id, bbox, score, cls_id)]
        """
        self.frame_id += 1

        # 1) No detections: cleanup and return currently alive tracks (or return [] if preferred)
        if bboxes_xyxy.size == 0:
            self._cleanup_dead_tracks()
            return [
                dict(
                    track_id=t.track_id,
                    bbox=t.bbox.copy(),
                    score=t.score,
                    cls_id=t.cls_id,
                )
                for t in self.tracks
            ]

        # 2) Filter low-confidence detections
        mask = scores >= float(self.cfg.track_thresh)
        if not np.any(mask):
            self._cleanup_dead_tracks()
            return []

        det_bboxes = bboxes_xyxy[mask].astype(np.float32)
        det_scores = scores[mask].astype(np.float32)
        det_cls_ids = cls_ids[mask].astype(np.int32)

        # 3) IoU matching between alive tracks and new detections
        self._cleanup_dead_tracks()
        tracks = self.tracks

        assigned_det = np.full(det_bboxes.shape[0], -1, dtype=np.int32)  # track index per detection

        if tracks:
            track_bboxes = np.stack([t.bbox for t in tracks], axis=0).astype(np.float32)
            iou_mat = pairwise_iou(track_bboxes, det_bboxes)  # [T, D]

            # Greedy match: for each track pick detection with max IoU
            for ti, t in enumerate(tracks):
                det_i = int(iou_mat[ti].argmax())
                if iou_mat[ti, det_i] >= float(self.cfg.match_thresh):
                    if assigned_det[det_i] == -1:
                        assigned_det[det_i] = ti

        # 4) Update existing tracks / create new tracks
        for di in range(det_bboxes.shape[0]):
            bbox = det_bboxes[di]
            score = float(det_scores[di])
            cls_id = int(det_cls_ids[di])

            ti = assigned_det[di]
            if ti >= 0:
                # Update existing track
                trk = tracks[ti]
                trk.bbox = bbox
                trk.score = score
                trk.cls_id = cls_id
                trk.last_frame = self.frame_id
            else:
                # Create new track
                new_t = _Track(self.next_id, bbox, score, cls_id, self.frame_id)
                self.next_id += 1
                self.tracks.append(new_t)

        # 5) Cleanup expired tracks again
        self._cleanup_dead_tracks()

        # Sort by track_id for stable output
        self.tracks.sort(key=lambda t: t.track_id)

        out: list[dict] = []
        for t in self.tracks:
            out.append(
                dict(
                    track_id=t.track_id,
                    bbox=t.bbox.copy(),
                    score=t.score,
                    cls_id=t.cls_id,
                )
            )
        return out


def pairwise_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute IoU matrix between two sets of bboxes.

    a: [M,4], b: [N,4], both xyxy format.
    Returns: [M,N]
    """
    M, N = a.shape[0], b.shape[0]
    out = np.zeros((M, N), dtype=np.float32)
    if M == 0 or N == 0:
        return out

    ax1, ay1, ax2, ay2 = a[:, 0:1], a[:, 1:2], a[:, 2:3], a[:, 3:4]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]

    inter_x1 = np.maximum(ax1, bx1)
    inter_y1 = np.maximum(ay1, by1)
    inter_x2 = np.minimum(ax2, bx2)
    inter_y2 = np.minimum(ay2, by2)

    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h

    area_a = np.maximum(0.0, ax2 - ax1) * np.maximum(0.0, ay2 - ay1)
    area_b = np.maximum(0.0, bx2 - bx1) * np.maximum(0.0, by2 - by1)

    union = area_a + area_b - inter + 1e-6
    out = (inter / union).astype(np.float32)
    return out