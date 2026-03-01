from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class GaussianDPConfig:
    epsilon: float
    delta: float
    sensitivity: float = 1.0
    sigma_cap: float = 25.0  # prevent huge noise

    def sigma(self) -> float:
        """
        Gaussian mechanism (classic bound):
          sigma >= sensitivity * sqrt(2 ln(1.25/delta)) / epsilon
        This is a common practical bound; cap is applied for stability.
        """
        eps = max(self.epsilon, 1e-8)
        delt = min(max(self.delta, 1e-12), 1.0 - 1e-12)
        s = self.sensitivity * math.sqrt(2.0 * math.log(1.25 / delt)) / eps
        return float(min(s, self.sigma_cap))


def add_gaussian_noise_bbox(
    bboxes_xyxy: np.ndarray,
    sigmas: np.ndarray,
    image_wh: tuple[int, int],
) -> np.ndarray:
    """
    Add Gaussian noise to bbox coords (xyxy).
    bboxes_xyxy: [N,4] float
    sigmas: [N] float, per-box stddev
    image_wh: (w,h)
    """
    if bboxes_xyxy.size == 0:
        return bboxes_xyxy

    w, h = image_wh
    noise = np.random.randn(*bboxes_xyxy.shape).astype(np.float32)
    noise *= sigmas.reshape(-1, 1).astype(np.float32)

    out = bboxes_xyxy.astype(np.float32) + noise
    # clamp & fix order
    out[:, 0] = np.clip(out[:, 0], 0, w - 1)
    out[:, 2] = np.clip(out[:, 2], 0, w - 1)
    out[:, 1] = np.clip(out[:, 1], 0, h - 1)
    out[:, 3] = np.clip(out[:, 3], 0, h - 1)

    x1 = np.minimum(out[:, 0], out[:, 2])
    x2 = np.maximum(out[:, 0], out[:, 2])
    y1 = np.minimum(out[:, 1], out[:, 3])
    y2 = np.maximum(out[:, 1], out[:, 3])

    out[:, 0], out[:, 2], out[:, 1], out[:, 3] = x1, x2, y1, y2
    return out