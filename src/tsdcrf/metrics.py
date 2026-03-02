"""
MSE, PSNR, and RMSE for privacy-preserving tracking (paper Section VI).

Formulas (paper):
  MSE = (1/n) * sum_i (x_i - hat{x}_i)^2
  PSNR (dB) = 10 * log10(MAX^2 / MSE)
  RMSE = sqrt(MSE)
"""
from __future__ import annotations
import numpy as np


def compute_mse_psnr(
    bbox_orig: np.ndarray,
    bbox_noisy: np.ndarray,
    is_sensitive: np.ndarray,
    max_val: float = 1.0,
) -> tuple[float, float]:
    """
    MSE and PSNR between original and noisy bbox coordinates (sensitive detections only).

    Args:
        bbox_orig: [N,4] original bbox (xyxy).
        bbox_noisy: [N,4] perturbed bbox (xyxy).
        is_sensitive: [N] bool, True for sensitive-class detections.
        max_val: MAX in PSNR (e.g. image width/height or 1.0 if normalized).

    Returns:
        (mse, psnr_dB). If no sensitive detections, returns (0.0, 0.0).
    """
    if not np.any(is_sensitive):
        return 0.0, 0.0
    orig = bbox_orig[is_sensitive].astype(np.float64)
    noisy = bbox_noisy[is_sensitive].astype(np.float64)
    n = orig.size
    diff = noisy - orig
    mse = float(np.mean(diff ** 2))
    if mse <= 0:
        return 0.0, 0.0
    psnr = 10.0 * np.log10((max_val ** 2) / mse)
    return mse, float(psnr)


def compute_rmse(
    bbox_orig: np.ndarray,
    bbox_noisy: np.ndarray,
    is_sensitive: np.ndarray,
) -> float:
    """
    RMSE = sqrt(MSE) over sensitive bbox coordinates (paper: tracking deviation).
    """
    mse, _ = compute_mse_psnr(bbox_orig, bbox_noisy, is_sensitive, max_val=1.0)
    return float(np.sqrt(mse))


def aggregate_metrics(
    mse_list: list[float],
    psnr_list: list[float],
    rmse_list: list[float],
) -> tuple[float, float, float]:
    """Average MSE, PSNR (dB), and RMSE over frames (skip zeros)."""
    mse_arr = np.array([m for m in mse_list if m > 0])
    psnr_arr = np.array([p for p in psnr_list if p != 0])
    rmse_arr = np.array([r for r in rmse_list if r > 0])
    avg_mse = float(np.mean(mse_arr)) if mse_arr.size else 0.0
    avg_psnr = float(np.mean(psnr_arr)) if psnr_arr.size else 0.0
    avg_rmse = float(np.mean(rmse_arr)) if rmse_arr.size else 0.0
    return avg_mse, avg_psnr, avg_rmse
