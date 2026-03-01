from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class ByteTrackConfig:
    track_thresh: float = 0.5
    track_buffer: int = 30
    match_thresh: float = 0.8
    frame_rate: int = 30


class ByteTrackWrapper:
    def __init__(self, cfg: ByteTrackConfig):
        """
        Requires YOLOX installed (it bundles ByteTrack code):
          from yolox.tracker.byte_tracker import BYTETracker
        """
        self.cfg = cfg
        from yolox.tracker.byte_tracker import BYTETracker
        self.tracker = BYTETracker(
            dict(
                track_thresh=cfg.track_thresh,
                track_buffer=cfg.track_buffer,
                match_thresh=cfg.match_thresh,
                mot20=False,
            ),
            frame_rate=cfg.frame_rate,
        )

    def update(self, bboxes_xyxy: np.ndarray, scores: np.ndarray, cls_ids: np.ndarray) -> list[dict]:
        """
        Return list of tracks with fields:
          track_id, bbox(xyxy), score, cls_id
        """
        if bboxes_xyxy.size == 0:
            online_targets = self.tracker.update(np.zeros((0, 5), dtype=np.float32), (1, 1), (1, 1))
        else:
            # BYTETracker expects [x1,y1,x2,y2,score]
            dets = np.concatenate([bboxes_xyxy, scores.reshape(-1, 1)], axis=1).astype(np.float32)
            online_targets = self.tracker.update(dets, (1, 1), (1, 1))

        out = []
        for t in online_targets:
            tlwh = t.tlwh  # x,y,w,h
            x, y, w, h = tlwh
            bbox = np.array([x, y, x + w, y + h], dtype=np.float32)
            out.append(
                dict(
                    track_id=int(t.track_id),
                    bbox=bbox,
                    score=float(t.score),
                    cls_id=-1,  # ByteTrack doesn't keep class; we fill later by IoU match
                )
            )

        # class assignment: match each track to nearest det by IoU (simple)
        if bboxes_xyxy.size > 0 and len(out) > 0:
            iou = pairwise_iou(np.stack([o["bbox"] for o in out]), bboxes_xyxy)
            det_idx = iou.argmax(axis=1)
            for i, o in enumerate(out):
                j = int(det_idx[i])
                o["cls_id"] = int(cls_ids[j])
                o["score"] = float(scores[j])

        return out


def pairwise_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # a: [M,4], b: [N,4]
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