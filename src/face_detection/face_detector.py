# face_detector.py
# Face Detection Module
# Detects and crops faces from images

from __future__ import annotations

import os
import time
import uuid
from typing import Any, List, Optional, Tuple

import cv2
import numpy as np


# change later once you have human detection output
def to_bgr(person_image: Any) -> np.ndarray:
    if isinstance(person_image, dict) and "image" in person_image:
        person_image = person_image["image"]

    if isinstance(person_image, str):
        img = cv2.imread(person_image)
        if img is None:
            raise ValueError(f"Could not read image from path: {person_image}")
        return img  # BGR

    if isinstance(person_image, np.ndarray):
        if person_image.ndim != 3 or person_image.shape[2] != 3:
            raise ValueError("Expected a color image with shape (H, W, 3).")
        return person_image

    raise TypeError("Unsupported person_image type.")


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _clip_box(x: int, y: int, w: int, h: int, W: int, H: int) -> Tuple[int, int, int, int]:
    x = max(0, x)
    y = max(0, y)
    w = max(1, min(w, W - x))
    h = max(1, min(h, H - y))
    return x, y, w, h


def detect_faces(
    person_image: Any,
    person_id: str | int,
    tmp_root: str = "tmp/faces",
    min_face_size: Tuple[int, int] = (30, 30),
    margin_ratio: float = 0.15,  # used as the "middle" crop
    max_faces: Optional[int] = None,
    # NEW: generate multiple crops per detected face from a single image
    multi_crop_margins: Tuple[float, ...] = (0.08, 0.15, 0.25),
    # If you want exactly 3 outputs for a single-person image, set max_faces=1 (in your test)
) -> List[str]:
    img = to_bgr(person_image)

    if img is None or img.size == 0:
        return []

    H, W = img.shape[:2]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        raise RuntimeError(f"Failed to load Haar cascade: {cascade_path}")

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=min_face_size,
        flags=cv2.CASCADE_SCALE_IMAGE,
    )

    if len(faces) == 0:
        return []

    faces = sorted(faces, key=lambda b: b[2] * b[3], reverse=True)
    if max_faces is not None:
        faces = faces[:max_faces]

    person_dir = os.path.join(tmp_root, str(person_id))
    _ensure_dir(person_dir)

    saved_paths: List[str] = []

    margins = list(multi_crop_margins)
    if margin_ratio not in margins:
        margins.append(margin_ratio)
    margins = sorted(set(margins))

    for (x, y, w, h) in faces:
        for m in margins:
            mx = int(w * m)
            my = int(h * m)

            x2, y2, w2, h2 = _clip_box(x - mx, y - my, w + 2 * mx, h + 2 * my, W, H)
            crop = img[y2 : y2 + h2, x2 : x2 + w2]
            if crop.size == 0:
                continue

            fname = f"face_{int(time.time()*1000)}_{uuid.uuid4().hex[:8]}_m{int(m*100)}.jpg"
            out_path = os.path.join(person_dir, fname)

            if cv2.imwrite(out_path, crop):
                saved_paths.append(out_path)

    return saved_paths
