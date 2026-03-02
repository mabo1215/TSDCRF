from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch
import cv2
from loguru import logger
from ultralytics import YOLO


@dataclass(frozen=True)
class YOLOXConfig:
    """
    Keeps original config field names for compatibility; implementation uses Ultralytics YOLO11.

    - ckpt_path: YOLO11 weight path, e.g. "yolo11n.pt" or custom model path
    - input_size: Kept for compatibility; scaling is handled by Ultralytics
    - conf_thre: Confidence threshold for post-filtering
    - nms_thre: Kept for compatibility (Ultralytics has built-in NMS)
    - device / fp16: Handled by Ultralytics/torch; kept for config only
    """
    ckpt_path: str
    input_size: tuple[int, int] = (640, 640)
    conf_thre: float = 0.3
    nms_thre: float = 0.45
    device: str = "cuda"
    fp16: bool = True


class YOLOXDetector:
    """
    Public API unchanged: YOLOXDetector.load() + infer(frame_bgr);
    internally uses Ultralytics YOLO11 for detection.
    """

    def __init__(self, cfg: YOLOXConfig):
        self.cfg = cfg
        self.model: YOLO | None = None

    def load(self):
        """
        Load model using Ultralytics YOLO11 weights.

        Install first: pip install ultralytics
        Then set yolox.ckpt_path in config.yaml to YOLO11 weight path, e.g.:
          "yolo11n.pt" or "weights/yolo11n.pt"
        """
        self.model = YOLO(self.cfg.ckpt_path)
        logger.info(f"Loaded YOLO11 model from {self.cfg.ckpt_path}")

    @torch.no_grad()
    def infer(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return (bboxes_xyxy, scores, cls_ids) in original image scale.
        """
        assert self.model is not None, "call load() first"

        # Ultralytics expects RGB; convert BGR to RGB
        img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # results: List[Results]; single image so take index 0
        results = self.model(img_rgb, verbose=False)[0]
        boxes = results.boxes

        if boxes is None or boxes.shape[0] == 0:
            return (
                np.zeros((0, 4), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
                np.zeros((0,), dtype=np.int32),
            )

        bboxes = boxes.xyxy.cpu().numpy().astype(np.float32)
        scores = boxes.conf.cpu().numpy().astype(np.float32)
        cls_ids = boxes.cls.cpu().numpy().astype(np.int32)

        # Filter by conf_thre from config
        if self.cfg.conf_thre is not None:
            mask = scores >= float(self.cfg.conf_thre)
            bboxes = bboxes[mask]
            scores = scores[mask]
            cls_ids = cls_ids[mask]

        if bboxes.size == 0:
            return (
                np.zeros((0, 4), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
                np.zeros((0,), dtype=np.int32),
            )

        return bboxes, scores, cls_ids