"""
Normalized Control Penalty (NCP) for privacy-preserving object tracking.

Reference: TSDCRF, IEEE TETCI. Before adding privacy-preserving noises, NCP is used
to filter/weight the detected object classifications so that: (1) sensitive classes
receive a higher penalty (more noise), improving privacy; (2) similar tracking
targets can still be distinguished after encoding/decoding. The penalty weights
correspond to the mu term in the CRF (Eq. 3): P(l|t) propto exp(sum_i mu_a s_a(...)).
The paper's Classification Metric (CM) normalizes penalty by class; here we output
per-detection weights that scale the Gaussian noise (higher weight = larger sigma).
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class NCPConfig:
    """
    NCP parameters: control the relationship between tracking accuracy and privacy.

    - enable: if False, returns ones (no NCP scaling).
    - mode: "class_weight" | "score_gate" | "combined".
    - sensitive_weight: penalty weight for sensitive-class detections (more noise).
    - nonsensitive_weight: penalty weight for non-sensitive (less noise to preserve utility).
    - score_gate_thre: in score_gate/combined, detections below this get higher penalty.
    """
    enable: bool = True
    mode: str = "class_weight"
    sensitive_weight: float = 1.0
    nonsensitive_weight: float = 0.2
    score_gate_thre: float = 0.35


def compute_ncp_weight(
    scores: np.ndarray,
    is_sensitive: np.ndarray,
    cfg: NCPConfig,
) -> np.ndarray:
    """
    Compute per-detection NCP weights used to scale the DP noise (sigma).

    Higher weight -> stronger penalty -> larger noise for that detection.
    Paper: NCP "can better control the relationship between the accuracy and
    privacy of the tracking target" and helps distinguish similar targets after noise.
    """
    if (not cfg.enable) or scores.size == 0:
        return np.ones_like(scores, dtype=np.float32)

    if cfg.mode == "class_weight":
        # Paper: penalty coefficient by class; sensitive classes get higher penalty (more noise).
        w = np.where(is_sensitive, cfg.sensitive_weight, cfg.nonsensitive_weight).astype(np.float32)
        return w

    if cfg.mode == "score_gate":
        # Low-confidence detections: higher penalty (filter or more noise) per paper's score-gate idea.
        base = np.ones_like(scores, dtype=np.float32)
        base = np.where(scores < cfg.score_gate_thre, base * 1.5, base * 0.8)
        base = np.where(is_sensitive, base * 1.2, base)
        return base.astype(np.float32)

    if cfg.mode == "combined":
        # Class weight first, then scale by score gate (low score -> more penalty).
        w = np.where(is_sensitive, cfg.sensitive_weight, cfg.nonsensitive_weight).astype(np.float32)
        scale = np.where(scores < cfg.score_gate_thre, 1.5, 0.8).astype(np.float32)
        return (w * scale).astype(np.float32)

    return np.ones_like(scores, dtype=np.float32)