"""
Main pipeline for TSDCRF: Time Series Dynamic Conditional Random Field for
Privacy-preserving Object Tracking (IEEE TETCI).

Pipeline per paper (Fig. 3):
  1) Detect objects (YOLO11).
  2) Mark sensitive classes (e.g. person) for privacy protection.
  3) NCP: compute per-detection penalty weights (filter/weight classifications before noise).
  4) Add (epsilon, delta)-DP Gaussian noise to sensitive bboxes; sigma scaled by NCP.
  5) Multi-object tracking on noisy detections (associate targets across frames).
  6) Temporal smoothing (DCRF-style): iterative estimation to reduce position deviation
     and maintain trajectory consistency after noise; output smoothed bboxes.
"""
from __future__ import annotations
import argparse
import os
from dataclasses import dataclass
import yaml
import cv2
import numpy as np
from loguru import logger

from .yolox_detector import YOLOXDetector, YOLOXConfig
from .tracker_bytetrack import ByteTrackWrapper, ByteTrackConfig
from .privacy import GaussianDPConfig, add_gaussian_noise_bbox
from .ncp import NCPConfig, compute_ncp_weight
from .temporal_smoother import TemporalConfig, TrackSmoother
from .viz import draw_tracks


COCO_CLASSNAMES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    4: "airplane",
    5: "bus",
    6: "train",
    7: "truck",
    8: "boat",
    # ... extend as needed (COCO 80 classes)
}


def load_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config.yaml")
    ap.add_argument("--video", type=str, default="", help="video path; empty => webcam(0)")
    ap.add_argument("--save", type=str, default="", help="optional output video path")
    ap.add_argument("--show", action="store_true", help="imshow window")
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = load_cfg(args.config)

    device = cfg.get("device", "cuda")
    fp16 = bool(cfg.get("fp16", True))

    # detector
    ycfg = cfg["yolox"]
    det = YOLOXDetector(
        YOLOXConfig(
            ckpt_path=ycfg["ckpt_path"],
            input_size=tuple(ycfg["input_size"]),
            conf_thre=float(ycfg["conf_thre"]),
            nms_thre=float(ycfg["nms_thre"]),
            device=device,
            fp16=fp16,
        )
    )
    det.load()

    # tracker
    tcfg = cfg["bytetrack"]
    tracker = ByteTrackWrapper(
        ByteTrackConfig(
            track_thresh=float(tcfg["track_thresh"]),
            track_buffer=int(tcfg["track_buffer"]),
            match_thresh=float(tcfg["match_thresh"]),
            frame_rate=int(tcfg["frame_rate"]),
        )
    )

    # privacy + ncp + temporal
    pcfg_raw = cfg.get("privacy", {})
    ncfg_raw = cfg.get("ncp", {})
    tc_raw = cfg.get("temporal", {})

    dp_cfg = GaussianDPConfig(
        epsilon=float(pcfg_raw.get("epsilon", 1.0)),
        delta=float(pcfg_raw.get("delta", 1e-5)),
        sensitivity=float(pcfg_raw.get("sensitivity", 1.0)),
        sigma_cap=float(pcfg_raw.get("sigma_cap", 25.0)),
    )
    ncp_cfg = NCPConfig(
        enable=bool(ncfg_raw.get("enable", True)),
        mode=str(ncfg_raw.get("mode", "class_weight")),
        sensitive_weight=float(ncfg_raw.get("sensitive_weight", 1.0)),
        nonsensitive_weight=float(ncfg_raw.get("nonsensitive_weight", 0.2)),
        score_gate_thre=float(ncfg_raw.get("score_gate_thre", 0.35)),
    )
    temporal_cfg = TemporalConfig(
        enable=bool(tc_raw.get("enable", True)),
        method=str(tc_raw.get("method", "kalman_smooth")),
        alpha=float(tc_raw.get("alpha", 0.7)),
    )
    smoother = TrackSmoother(temporal_cfg)

    sensitive_names = set(pcfg_raw.get("sensitive_classes", ["person"]))
    privacy_enable = bool(pcfg_raw.get("enable", True))

    video_src = 0 if args.video == "" else args.video
    cap = cv2.VideoCapture(video_src)
    if not cap.isOpened():
        if args.video == "":
            raise RuntimeError(
                "Failed to open camera (device 0). "
                "Check Windows: Settings → Privacy → Camera → allow desktop apps. "
                "Or run with a file: --video path/to/video.mp4"
            )
        raise RuntimeError(f"Failed to open video: {args.video}")

    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        writer = cv2.VideoWriter(args.save, fourcc, fps, (w, h))

    show_gui = bool(args.show)
    if show_gui:
        try:
            cv2.namedWindow("TSDCRF-YOLO11", cv2.WINDOW_NORMAL)
        except cv2.error:
            logger.warning(
                "OpenCV GUI not available (headless build). "
                "Install GUI build: pip uninstall opencv-python-headless; pip install opencv-python. "
                "Skipping display."
            )
            show_gui = False

    if show_gui:
        logger.info("Press Q or ESC to exit.")
    logger.info("Start processing...")
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        h, w = frame.shape[:2]

        # 1) Detect (YOLO11)
        bboxes, scores, cls_ids = det.infer(frame)

        # 2) Sensitive class mask (paper: privacy categories to protect).
        cls_names = [COCO_CLASSNAMES.get(int(c), str(int(c))) for c in cls_ids.tolist()] if cls_ids.size else []
        is_sensitive = np.array([name in sensitive_names for name in cls_names], dtype=bool) if cls_ids.size else np.zeros((0,), bool)

        # 3) NCP: penalty weights before adding noise (paper: filter classifications, control accuracy vs privacy).
        ncp_w = compute_ncp_weight(scores, is_sensitive, ncp_cfg)

        # 4) (epsilon, delta)-DP Gaussian noise on sensitive bboxes; sigma = base_sigma * NCP weight.
        bboxes_noisy = bboxes
        if privacy_enable and bboxes.size > 0:
            base_sigma = dp_cfg.sigma()
            sigmas = base_sigma * ncp_w
            sigmas = np.where(is_sensitive, sigmas, 0.0).astype(np.float32)
            bboxes_noisy = add_gaussian_noise_bbox(bboxes, sigmas, (w, h))

        # 5) Multi-object tracking on noisy detections (associate targets in time series).
        tracks = tracker.update(bboxes_noisy, scores, cls_ids)

        # 6) DCRF-style temporal smoothing: reduce position deviation, maintain trajectory consistency (paper Eq. 16-19).
        out_tracks = []
        for t in tracks:
            cls_name = COCO_CLASSNAMES.get(int(t["cls_id"]), str(int(t["cls_id"])))
            sens = cls_name in sensitive_names
            smoothed_bbox = smoother.update(t["track_id"], t["bbox"])
            smoothed_bbox[0] = np.clip(smoothed_bbox[0], 0, w - 1)
            smoothed_bbox[2] = np.clip(smoothed_bbox[2], 0, w - 1)
            smoothed_bbox[1] = np.clip(smoothed_bbox[1], 0, h - 1)
            smoothed_bbox[3] = np.clip(smoothed_bbox[3], 0, h - 1)

            out_tracks.append(
                dict(
                    track_id=t["track_id"],
                    bbox=smoothed_bbox.astype(np.float32),
                    score=float(t["score"]),
                    cls_id=int(t["cls_id"]),
                    is_sensitive=bool(sens),
                )
            )

        vis = draw_tracks(frame, out_tracks, class_names=COCO_CLASSNAMES)

        if writer is not None:
            writer.write(vis)
        if show_gui:
            try:
                cv2.imshow("TSDCRF-YOLO11", vis)
                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord("q") or key == ord("Q"):
                    break
            except cv2.error:
                logger.warning("OpenCV imshow failed (headless). Skipping display.")
                show_gui = False

    cap.release()
    if writer is not None:
        writer.release()
    if show_gui:
        cv2.destroyAllWindows()
    logger.info("Done.")


if __name__ == "__main__":
    main()