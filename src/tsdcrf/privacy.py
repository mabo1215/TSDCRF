"""
Gaussian (epsilon, delta)-differential privacy for bounding box (ROI) coordinates.

Reference: TSDCRF (Time Series Dynamic Conditional Random Field for Privacy-preserving
Object Tracking), IEEE TETCI. The paper adds sampling noise to sensitive classes to
prevent feature extraction attacks (Threat Model 1). For (epsilon, delta)-DP, the Gaussian
output perturbation satisfies: sigma >= sensitivity * sqrt(2 ln(1.25/delta)) / epsilon
(Dwork et al.). Noise is applied to the region of interest (bbox) so that the tracking
task can still be performed while the attacker cannot restore the target's features.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class GaussianDPConfig:
    """
    (epsilon, delta)-differential privacy parameters for the Gaussian mechanism.

    - epsilon: privacy budget; smaller = stronger privacy, larger noise.
    - delta: target delta in (epsilon, delta)-DP; typically 1e-5.
    - sensitivity: L2 sensitivity of the bbox query (max change in one coordinate);
      default 1.0 for normalized or pixel-scale coordinates.
    - sigma_cap: upper bound on sigma to avoid numerical instability when epsilon is very small.
    """
    epsilon: float
    delta: float
    sensitivity: float = 1.0
    sigma_cap: float = 25.0

    def sigma(self) -> float:
        """
        Standard deviation for (epsilon, delta)-DP Gaussian mechanism.

        Paper bound: sigma >= sensitivity * sqrt(2 ln(1.25/delta)) / epsilon.
        Returns min(computed_sigma, sigma_cap) for stability when epsilon -> 0.
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
    Add independent Gaussian noise to each bbox coordinate (xyxy format).

    Per the paper: noise is added to the tracked target ROI so that the attacker
    cannot backtrack and extract the hidden object's features; tracking continues
    on the noisy observations and is corrected by the temporal (DCRF) stage.

    Args:
        bboxes_xyxy: [N,4] float, xyxy in image coordinates.
        sigmas: [N] float, per-box standard deviation (e.g. base_sigma * NCP_weight).
        image_wh: (width, height) for clipping outputs.

    Returns:
        Noisy bboxes [N,4] clipped to image and reordered so x1<=x2, y1<=y2.
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