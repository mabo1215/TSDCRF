"""
Temporal smoothing (DCRF-style) for privacy-preserving object tracking.

Reference: TSDCRF, IEEE TETCI. After adding DP noise to the bounding box, the tracking
position is deviated. The paper proposes "iterative estimation" (Eq. 16-19) to reduce
position deviation and defend against orbit hijacking: use a global time-series view
to correct wrong deviation of the target window and trajectory caused by attack or noise.
DCRF (Dynamic CRF) maintains trajectory consistency across frames. This module implements
a per-track Kalman filter on (center, size) plus optional blending with the noisy
observation, corresponding to the state update and covariance update in the paper.
"""
from __future__ import annotations
from dataclasses import dataclass
from filterpy.kalman import KalmanFilter
import numpy as np


@dataclass
class TemporalConfig:
    """
    Temporal smoothing parameters (DCRF / iterative estimation in the paper).

    - enable: if False, returns the noisy bbox unchanged.
    - method: "kalman_smooth" (paper Eq. 16-19 style) or "none".
    - alpha: blend factor; smoothed = alpha * kalman_state + (1-alpha) * noisy_bbox.
      Higher alpha = trust the smoothed trajectory more (stronger temporal consistency).
    """
    enable: bool = True
    method: str = "kalman_smooth"
    alpha: float = 0.7


class TrackSmoother:
    """
    Per-track Kalman smoother for bbox (cx, cy, w, h) with velocity state.

    Reduces position deviation after DP noise (paper: iterative estimation Eq. 16-19)
    and maintains trajectory consistency to mitigate perturbation and orbit hijacking.
    State: [cx, cy, w, h, vx, vy, vw, vh]; measurement: [cx, cy, w, h].
    """
    def __init__(self, cfg: TemporalConfig):
        self.cfg = cfg
        self._kf: dict[int, KalmanFilter] = {}

    def _init_kf(self, xyxy: np.ndarray) -> KalmanFilter:
        x1, y1, x2, y2 = xyxy.tolist()
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        w, h = max(x2 - x1, 1.0), max(y2 - y1, 1.0)

        kf = KalmanFilter(dim_x=8, dim_z=4)
        dt = 1.0

        # state transition
        F = np.eye(8, dtype=np.float32)
        for i in range(4):
            F[i, i + 4] = dt
        kf.F = F

        # measurement function
        H = np.zeros((4, 8), dtype=np.float32)
        H[0, 0] = 1
        H[1, 1] = 1
        H[2, 2] = 1
        H[3, 3] = 1
        kf.H = H

        kf.x = np.array([cx, cy, w, h, 0, 0, 0, 0], dtype=np.float32)

        kf.P *= 10.0
        kf.R *= 5.0
        kf.Q *= 0.01
        return kf

    def update(self, track_id: int, noisy_xyxy: np.ndarray) -> np.ndarray:
        if (not self.cfg.enable) or self.cfg.method != "kalman_smooth":
            return noisy_xyxy

        if track_id not in self._kf:
            self._kf[track_id] = self._init_kf(noisy_xyxy)

        kf = self._kf[track_id]
        kf.predict()

        x1, y1, x2, y2 = noisy_xyxy.astype(np.float32).tolist()
        z = np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0, max(x2 - x1, 1.0), max(y2 - y1, 1.0)], dtype=np.float32)
        kf.update(z)

        cx, cy, w, h = kf.x[0], kf.x[1], max(kf.x[2], 1.0), max(kf.x[3], 1.0)
        smoothed = np.array([cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0], dtype=np.float32)

        # Blend smoothed state with noisy observation (paper: fuse valid measurements to reduce deviation).
        out = self.cfg.alpha * smoothed + (1.0 - self.cfg.alpha) * noisy_xyxy.astype(np.float32)
        return out