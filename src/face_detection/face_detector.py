# face_detector.py
# Face Detection Module
# Takes a cropped person image from human detection input and returns the cropped face

from __future__ import annotations

import os
from typing import Optional, Tuple

import cv2
import numpy as np

#makes sure the cropped image is within the bounds of the original image
def _clip_box(
    x: int, y: int, w: int, h: int, W: int, H: int
) -> Tuple[int, int, int, int]:
    x = max(0, x)
    y = max(0, y)
    w = max(1, min(w, W - x))
    h = max(1, min(h, H - y))
    return x, y, w, h


# Minimum height (px) the crop is upscaled to before running the cascade.
# Covers distant persons whose body crop may be only 80-120 px tall.
_MIN_DETECTION_HEIGHT = 200

# Pre-load cascades once at import time (avoid repeated disk I/O per call).
_CASCADE_FRONTAL = cv2.CascadeClassifier(
    os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
)
_CASCADE_PROFILE = cv2.CascadeClassifier(
    os.path.join(cv2.data.haarcascades, "haarcascade_profileface.xml")
)


def detect_face(
    person_bgr: np.ndarray,
    min_face_size: Tuple[int, int] = (20, 20),
    margin_ratio: float = 0.15,
) -> Optional[np.ndarray]: # Returns: a cropped face image or None if no face is detected in the image
    
    if person_bgr is None or person_bgr.size == 0:
        return None

    if person_bgr.ndim != 3 or person_bgr.shape[2] != 3:
        raise ValueError("Expected BGR image with shape (H, W, 3).")

    H, W = person_bgr.shape[:2]

    # ── Upscale small / distant crops so the cascade can find tiny faces ──
    # When a person is far away their bounding-box crop may be only 80-150 px
    # tall.  A face in that crop can be as small as 15-25 px, which is below
    # even a (20,20) minSize.  Upscaling to _MIN_DETECTION_HEIGHT keeps the
    # detection pipeline working at any distances the human detector handles.
    scale = 1.0
    work = person_bgr
    if H < _MIN_DETECTION_HEIGHT:
        scale = _MIN_DETECTION_HEIGHT / H
        work = cv2.resize(person_bgr, (int(W * scale), _MIN_DETECTION_HEIGHT),
                          interpolation=cv2.INTER_LINEAR)

    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    if _CASCADE_FRONTAL.empty():
        raise RuntimeError("Failed to load frontal face Haar cascade.")

    def _run_cascade(cascade, gray_img, min_nb):
        return cascade.detectMultiScale(
            gray_img,
            scaleFactor=1.1,    # 1.1 (default) is ~2x faster than 1.05 with minimal accuracy loss
            minNeighbors=min_nb,
            minSize=min_face_size,
            flags=cv2.CASCADE_SCALE_IMAGE,
        )

    # Try frontal first; fall back to profile for turned heads
    faces = _run_cascade(_CASCADE_FRONTAL, gray, min_nb=2)
    if len(faces) == 0 and not _CASCADE_PROFILE.empty():
        faces = _run_cascade(_CASCADE_PROFILE, gray, min_nb=2)

    if len(faces) == 0:
        return None

    # Select largest face (in upscaled coordinates)
    x, y, w, h = max(faces, key=lambda b: b[2] * b[3])

    # Map coordinates back to original image space
    if scale != 1.0:
        x = int(x / scale); y = int(y / scale)
        w = int(w / scale); h = int(h / scale)

    # Apply margin padding in original space
    mx = int(w * margin_ratio)
    my = int(h * margin_ratio)

    x2, y2, w2, h2 = _clip_box(x - mx, y - my, w + 2 * mx, h + 2 * my, W, H)

    face_crop = person_bgr[y2 : y2 + h2, x2 : x2 + w2]

    if face_crop.size == 0:
        return None

    return face_crop