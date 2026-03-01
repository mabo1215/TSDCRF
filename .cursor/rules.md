# Project rules (Cursor)

## Goals
- Provide a reproducible demo pipeline for:
  1) YOLOX detection
  2) ByteTrack association
  3) NCP penalty (class-aware / score-aware)
  4) Gaussian (epsilon, delta)-DP noise injection
  5) Temporal smoothing to reduce ID switching and trajectory drift

## Coding style
- Python 3.10+
- Explicit typing
- CLI must be stable
- Keep modules small and composable

## Output conventions
- All coordinates are xyxy in pixel space.
- Use float internally, clamp to image bounds before drawing/saving.