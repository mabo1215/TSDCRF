from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
import torch
import cv2
from loguru import logger


@dataclass(frozen=True)
class YOLOXConfig:
    ckpt_path: str
    input_size: tuple[int, int] = (640, 640)
    conf_thre: float = 0.3
    nms_thre: float = 0.45
    device: str = "cuda"
    fp16: bool = True


class YOLOXDetector:
    def __init__(self, cfg: YOLOXConfig):
        self.cfg = cfg
        self.model = None
        self.cls_names = None

    def load(self):
        """
        Requires YOLOX installed:
          pip install git+https://github.com/Megvii-BaseDetection/YOLOX.git
        """
        from yolox.exp import get_exp
        from yolox.utils import postprocess
        from yolox.data.data_augment import preproc

        self._postprocess = postprocess
        self._preproc = preproc

        exp = get_exp(None, "yolox-s")  # default exp name
        model = exp.get_model()
        ckpt = torch.load(self.cfg.ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt, strict=False)
        model.eval()

        if self.cfg.device == "cuda" and torch.cuda.is_available():
            model = model.cuda()
            if self.cfg.fp16:
                model = model.half()

        self.model = model
        self.num_classes = exp.num_classes
        logger.info(f"Loaded YOLOX from {self.cfg.ckpt_path}, num_classes={self.num_classes}")

    @torch.no_grad()
    def infer(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return (bboxes_xyxy, scores, cls_ids) in original image scale.
        """
        assert self.model is not None, "call load() first"

        img = frame_bgr
        h0, w0 = img.shape[:2]
        inp_h, inp_w = self.cfg.input_size

        img_resized, ratio = self._preproc(img, (inp_h, inp_w), (0, 0, 0))
        img_tensor = torch.from_numpy(img_resized).unsqueeze(0)  # [1,3,H,W]
        img_tensor = img_tensor.float()
        if self.cfg.fp16 and self.cfg.device == "cuda" and torch.cuda.is_available():
            img_tensor = img_tensor.half()
        if self.cfg.device == "cuda" and torch.cuda.is_available():
            img_tensor = img_tensor.cuda()

        outputs = self.model(img_tensor)
        outputs = self._postprocess(
            outputs,
            self.num_classes,
            self.cfg.conf_thre,
            self.cfg.nms_thre,
            class_agnostic=True,
        )[0]

        if outputs is None or outputs.numel() == 0:
            return (
                np.zeros((0, 4), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
                np.zeros((0,), dtype=np.int32),
            )

        outputs = outputs.float().cpu().numpy()

        bboxes = outputs[:, 0:4]
        scores = outputs[:, 4] * outputs[:, 5]
        cls_ids = outputs[:, 6].astype(np.int32)

        # map back to original scale
        bboxes /= ratio
        bboxes[:, 0] = np.clip(bboxes[:, 0], 0, w0 - 1)
        bboxes[:, 2] = np.clip(bboxes[:, 2], 0, w0 - 1)
        bboxes[:, 1] = np.clip(bboxes[:, 1], 0, h0 - 1)
        bboxes[:, 3] = np.clip(bboxes[:, 3], 0, h0 - 1)

        return bboxes.astype(np.float32), scores.astype(np.float32), cls_ids