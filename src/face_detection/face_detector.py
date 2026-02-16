# face_detector.py
# Face Detection Module
# Detects and crops faces from images or human-detector outputs

from __future__ import annotations

import os
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


def to_bgr(person_image: Any) -> np.ndarray:
    """
    Accepts:
      - str path to image
      - np.ndarray (H,W,3) BGR
      - dict from upstream containing one of: {"image": ...} or {"crop": ...}
    Returns:
      - BGR np.ndarray
    """
    # Support dict outputs from other modules
    if isinstance(person_image, dict):
        if "image" in person_image:
            person_image = person_image["image"]
        elif "crop" in person_image:
            person_image = person_image["crop"]

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


def _crop_from_bbox(frame_bgr: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
    """
    bbox format expected from human detection: (x, y, w, h)
    Returns cropped person image or None.
    """
    if frame_bgr is None or frame_bgr.size == 0 or bbox is None:
        return None

    H, W = frame_bgr.shape[:2]
    x, y, w, h = bbox

    # Be robust to floats/np types
    x = int(round(float(x)))
    y = int(round(float(y)))
    w = int(round(float(w)))
    h = int(round(float(h)))

    x, y, w, h = _clip_box(x, y, w, h, W, H)
    crop = frame_bgr[y : y + h, x : x + w]
    if crop.size == 0:
        return None
    return crop


def detect_faces(
    person_image: Any,
    person_id: str | int,
    tmp_root: str = "tmp/faces",
    min_face_size: Tuple[int, int] = (30, 30),
    margin_ratio: float = 0.15,
    max_faces: Optional[int] = None,
    # generate multiple crops per detected face from a single image
    multi_crop_margins: Tuple[float, ...] = (0.08, 0.15, 0.25),
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

    # Biggest faces first
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


def detect_faces_from_humans(
    frame_bgr: np.ndarray,
    human_results: List[Dict[str, Any]],
    tmp_root: str = "tmp/faces",
    min_face_size: Tuple[int, int] = (30, 30),
    # For your "3 face crops" requirement: detect 1 face, output 3 margin crops
    max_faces_per_person: int = 1,
    multi_crop_margins: Tuple[float, ...] = (0.08, 0.15, 0.25),
) -> List[Dict[str, Any]]:
 
    frame = to_bgr(frame_bgr)  # ensures it's a valid BGR ndarray
    outputs: List[Dict[str, Any]] = []

    for person in human_results:
        pid = person.get("id")
        bbox = person.get("bbox")

        if pid is None or bbox is None:
            continue

        person_crop = _crop_from_bbox(frame, bbox)
        if person_crop is None:
            continue

        face_paths = detect_faces(
            person_crop,
            person_id=pid,
            tmp_root=tmp_root,
            min_face_size=min_face_size,
            max_faces=max_faces_per_person,
            multi_crop_margins=multi_crop_margins,
        )

        outputs.append({"person_id": pid, "face_paths": face_paths})

    return outputs
