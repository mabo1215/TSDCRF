# TSDCRF (Time Series Dynamic CRF for Privacy-preserving Object Tracking)

Pipeline (per paper, IEEE TETCI):
1. YOLO11 object detection
2. Multi-object tracking (IoU-based)
3. NCP (Normalized Control Penalty): class- and score-aware weights before noise
4. Gaussian (ε, δ)-DP noise on sensitive-class bboxes
5. Temporal smoothing (DCRF-style Kalman) to reduce position deviation and keep trajectory consistency

## Setup

### 1) Create venv and install dependencies
```bash
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# Linux/macOS:
# source .venv/bin/activate

pip install -r requirements.txt
pip install ultralytics
```

### 2) Weights
Place YOLO11 weights (e.g. `yolo11n.pt`) in `weights/` or set `yolox.ckpt_path` in `config.yaml` to a valid path. The first run will download the model if the path is a Ultralytics model name (e.g. `yolo11n.pt`).

### 3) Run

Webcam:
```bash
python -m src.tsdcrf.main --config config.yaml --show
```
Video file and save output:
```bash
python -m src.tsdcrf.main --config config.yaml --video input.mp4 --save output.mp4
```

---

## Configuration and parameter tuning

All parameters are in `config.yaml`. Below is how to use and tune them.

### Global
| Parameter | Description | Tuning |
|-----------|-------------|--------|
| `device` | `"cuda"` or `"cpu"` | Use `cuda` if a GPU is available. |
| `fp16` | Use half precision on GPU | `true` for faster inference on supported GPUs. |

### Detection (`yolox`)

Uses Ultralytics YOLO11; the section name is kept for compatibility.

| Parameter | Description | Tuning |
|-----------|-------------|--------|
| `ckpt_path` | Model weights path or name (e.g. `yolo11n.pt`, `weights/yolo11s.pt`) | Use `yolo11n.pt` for speed, `yolo11s.pt` / `yolo11m.pt` for better accuracy. |
| `input_size` | Resize before inference | Default `[640, 640]`; leave as-is unless you change the model. |
| `conf_thre` | Detection confidence threshold | **Lower** (e.g. 0.2): more detections, more false positives. **Higher** (e.g. 0.5): fewer detections, fewer false positives. |
| `nms_thre` | NMS IoU threshold | Usually 0.4–0.5; lower = more aggressive NMS. |

### Tracker (`bytetrack`)

Lightweight IoU-based multi-object tracker (no dependency on YOLOX).

| Parameter | Description | Tuning |
|-----------|-------------|--------|
| `track_thresh` | Min confidence for a detection to enter tracking | **Lower**: more tracks, more fragile IDs. **Higher**: only confident detections tracked. |
| `track_buffer` | Frames to keep a track without match before removal | **Higher**: IDs survive longer through short occlusions. **Lower**: IDs dropped sooner. |
| `match_thresh` | IoU threshold to match detection to track | **Higher** (e.g. 0.9): stricter match, fewer ID switches, more new IDs. **Lower** (e.g. 0.6): looser match, more switches. |
| `frame_rate` | Used for semantics only in current implementation | Set to your video FPS if you add time-based logic later. |

### Privacy (`privacy`)

(ε, δ)-differential privacy via Gaussian noise on **sensitive-class** bboxes only.

| Parameter | Description | Tuning |
|-----------|-------------|--------|
| `enable` | Turn privacy noise on/off | `true` to protect sensitive classes; `false` for no noise. |
| `epsilon` | Privacy budget | **Smaller** (e.g. 0.5): stronger privacy, **larger** noise, more tracking drift. **Larger** (e.g. 2.0): weaker privacy, smaller noise. |
| `delta` | Target δ in (ε, δ)-DP | Typically `1e-5`; keep fixed unless you know what you need. |
| `sensitivity` | L2 sensitivity of bbox query | Default 1.0; increase if your coordinate scale is larger. |
| `sigma_cap` | Max sigma (pixel) | Prevents huge noise when ε is very small; increase only if needed. |
| `sensitive_classes` | Class names to protect (e.g. person) | Add more names (e.g. `["person", "face"]`) to protect more classes. |

### NCP (`ncp`)

Normalized Control Penalty: weights applied **before** adding DP noise. Higher weight → more noise for that detection.

| Parameter | Description | Tuning |
|-----------|-------------|--------|
| `enable` | Use NCP weights | `false` → all detections use the same base sigma. |
| `mode` | `"class_weight"` \| `"score_gate"` \| `"combined"` | **class_weight**: only sensitive vs non-sensitive. **score_gate**: low-confidence gets more penalty. **combined**: both. |
| `sensitive_weight` | Weight for sensitive-class detections | **Higher** (e.g. 1.5): more noise on persons/sensitive. **Lower**: less noise, less privacy. |
| `nonsensitive_weight` | Weight for other classes | Usually &lt; 1 (e.g. 0.2) so non-sensitive get little noise and tracking stays stable. |
| `score_gate_thre` | Confidence below which penalty increases (score_gate / combined) | **Lower** (e.g. 0.25): only very low-confidence get extra penalty. **Higher** (e.g. 0.5): more detections get extra penalty. |

### Temporal smoother (`temporal`)

DCRF-style temporal smoothing: per-track Kalman filter on bbox (center + size) to reduce position deviation after DP noise and keep trajectory consistency (paper Eq. 16–19).

| Parameter | Description | Tuning |
|-----------|-------------|--------|
| `enable` | Turn temporal smoothing on/off | `true` to smooth trajectories; `false` to output raw tracked bboxes (noisy). |
| `method` | Smoothing method | `"kalman_smooth"`: Kalman + blend. Other values: no smoothing (pass-through). |
| `alpha` | Blend: `output = alpha * smoothed + (1 - alpha) * noisy_bbox` | **Higher** (e.g. 0.8–0.9): trust Kalman more → smoother, less jitter, may lag on fast motion. **Lower** (e.g. 0.4–0.5): trust current frame more → more responsive, more jitter. Default **0.7** is a middle ground. |

**When to use Temporal smoother**
- **Use** (`enable: true`, `method: "kalman_smooth"`) when you add DP noise: it reduces bbox jitter and helps maintain stable IDs.
- **Disable** (`enable: false`) if you do not use privacy noise and want raw tracker output with no extra smoothing.
- If trajectories look **too laggy**, decrease `alpha` (e.g. 0.5). If they are **too jittery**, increase `alpha` (e.g. 0.85).

---

## Notes

- COCO class names are partially listed in code; extend `COCO_CLASSNAMES` in `main.py` if you use more classes.
- Sensitive classes in `privacy.sensitive_classes` must match names in `COCO_CLASSNAMES` (e.g. `person`).
