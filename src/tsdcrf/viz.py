from __future__ import annotations
import cv2
import numpy as np


def draw_tracks(
    frame: np.ndarray,
    tracks: list[dict],
    class_names: dict[int, str] | None = None,
) -> np.ndarray:
    """
    tracks item:
      {"track_id": int, "bbox": np.ndarray[4], "score": float, "cls_id": int, "is_sensitive": bool}
    """
    out = frame.copy()
    for t in tracks:
        x1, y1, x2, y2 = t["bbox"].astype(int).tolist()
        tid = t["track_id"]
        score = t["score"]
        cls_id = t["cls_id"]
        name = class_names.get(cls_id, str(cls_id)) if class_names else str(cls_id)
        tag = f"ID:{tid} {name} {score:.2f}"
        if t.get("is_sensitive", False):
            tag = "[S] " + tag

        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(out, tag, (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    return out