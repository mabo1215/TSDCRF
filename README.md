# TSDCRF-YOLOX (Runnable Scaffold)

Pipeline:
1) YOLOX object detection
2) ByteTrack tracking
3) NCP penalty (class-aware / score-aware)
4) Gaussian (epsilon, delta)-DP noise injection on sensitive classes
5) Temporal smoothing (Kalman-based) as a lightweight approximation to time-series CRF consistency

## Setup

### 1) Create venv
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

###  2) Install YOLOX
```bash
pip install git+https://github.com/Megvii-BaseDetection/YOLOX.git
```

###  3) Run

Webcam:
```bash
python -m tsdcrf.main --config config.yaml --show
```
Video file + save:
```bash
python -m tsdcrf.main --config config.yaml --video input.mp4 --save output.mp4
```

###  Notes

COCO class names are partially included; extend if needed.

"Temporal smoother" is a lightweight proxy for time-series CRF. Replace temporal_smoother.py with a real DCRF inference module if required.
