from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class NCPConfig:
    enable: bool = True
    mode: str = "class_weight"  # "class_weight" | "score_gate"
    sensitive_weight: float = 1.0
    nonsensitive_weight: float = 0.2
    score_gate_thre: float = 0.35


def compute_ncp_weight(
    scores: np.ndarray,
    is_sensitive: np.ndarray,
    cfg: NCPConfig,
) -> np.ndarray:
    """
    Return per-detection weight in [0, +inf).
    Higher weight -> stronger penalty / larger noise.
    """
    if (not cfg.enable) or scores.size == 0:
        return np.ones_like(scores, dtype=np.float32)

    if cfg.mode == "class_weight":
        w = np.where(is_sensitive, cfg.sensitive_weight, cfg.nonsensitive_weight).astype(np.float32)
        return w

    if cfg.mode == "score_gate":
        # low-confidence -> higher penalty
        base = np.ones_like(scores, dtype=np.float32)
        base = np.where(scores < cfg.score_gate_thre, base * 1.5, base * 0.8)
        # sensitive slightly higher
        base = np.where(is_sensitive, base * 1.2, base)
        return base.astype(np.float32)

    # fallback
    return np.ones_like(scores, dtype=np.float32)