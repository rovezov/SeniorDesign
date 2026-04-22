# face_detector.py
# Lightweight face detection module.
# Takes a cropped person image from human detection input and returns the cropped face.

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


def _find_cascade_path() -> Optional[str]:
    """Find a local Haar cascade XML without relying on cv2.data."""
    candidates = [
        os.getenv("FACE_CASCADE_PATH"),
        os.path.join(os.path.dirname(cv2.__file__), "data", "haarcascade_frontalface_default.xml"),
        "/home/ubuntu/gearguard-env/lib/python3.12/site-packages/cv2/data/haarcascade_frontalface_default.xml",
        "/home/ubuntu/SeniorDesign/venv/lib/python3.11/site-packages/cv2/data/haarcascade_frontalface_default.xml",
        "/home/ubuntu/SeniorDesign/demo_venv/lib/python3.11/site-packages/cv2/data/haarcascade_frontalface_default.xml",
        "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
        "/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml",
        "/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
        "/usr/local/share/opencv/haarcascades/haarcascade_frontalface_default.xml",
    ]
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate
    return None


_CASCADE_PATH = _find_cascade_path()
_CASCADE_DETECTOR = cv2.CascadeClassifier(_CASCADE_PATH) if _CASCADE_PATH else None
if _CASCADE_DETECTOR is not None and _CASCADE_DETECTOR.empty():
    _CASCADE_DETECTOR = None


def _detect_with_cascade(
    person_bgr: np.ndarray,
    min_face_size: Tuple[int, int],
    margin_ratio: float,
) -> Optional[np.ndarray]:
    if _CASCADE_DETECTOR is None:
        return None

    H, W = person_bgr.shape[:2]
    gray = cv2.cvtColor(person_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    faces = _CASCADE_DETECTOR.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=min_face_size,
    )
    if len(faces) == 0:
        return None

    x, y, w, h = max(faces, key=lambda box: box[2] * box[3])

    mx = int(w * margin_ratio)
    my = int(h * margin_ratio)
    x2, y2, w2, h2 = _clip_box(x - mx, y - my, w + 2 * mx, h + 2 * my, W, H)
    face_crop = person_bgr[y2 : y2 + h2, x2 : x2 + w2]
    return face_crop if face_crop.size > 0 else None


def _detect_with_heuristic(
    person_bgr: np.ndarray,
    margin_ratio: float,
) -> Optional[np.ndarray]:
    """Return a cheap upper-center crop when no detector model is available."""
    H, W = person_bgr.shape[:2]
    if H < 40 or W < 40:
        return None

    face_h = max(32, int(H * 0.40))
    face_w = max(32, int(W * 0.55))
    center_x = W // 2
    x = max(0, center_x - face_w // 2)
    y = max(0, int(H * 0.02))
    x2, y2, w2, h2 = _clip_box(x, y, face_w, face_h, W, H)

    mx = int(w2 * margin_ratio)
    my = int(h2 * margin_ratio)
    x3, y3, w3, h3 = _clip_box(x2 - mx, y2 - my, w2 + 2 * mx, h2 + 2 * my, W, H)
    face_crop = person_bgr[y3 : y3 + h3, x3 : x3 + w3]
    return face_crop if face_crop.size > 0 else None


def _detect_with_insightface(
    person_bgr: np.ndarray,
    min_face_size: Tuple[int, int],
    margin_ratio: float,
    confidence_threshold: float,
) -> Optional[np.ndarray]:
    try:
        from insightface.app import FaceAnalysis
    except ImportError:
        return None

    detector = FaceAnalysis(name='buffalo_sc', providers=['CPUExecutionProvider'])
    detector.prepare(ctx_id=-1, det_size=(640, 640))

    H, W = person_bgr.shape[:2]
    person_rgb = cv2.cvtColor(person_bgr, cv2.COLOR_BGR2RGB)
    faces = detector.get(person_rgb)
    if not faces:
        return None

    best_face = max(faces, key=lambda f: f.det_score)
    if best_face.det_score < confidence_threshold:
        return None

    bbox = best_face.bbox.astype(int)
    x, y, x_end, y_end = bbox[0], bbox[1], bbox[2], bbox[3]
    w = x_end - x
    h = y_end - y

    mx = int(w * margin_ratio)
    my = int(h * margin_ratio)
    x2, y2, w2, h2 = _clip_box(x - mx, y - my, w + 2 * mx, h + 2 * my, W, H)
    face_crop = person_bgr[y2 : y2 + h2, x2 : x2 + w2]
    return face_crop if face_crop.size > 0 else None


def detect_face(
    person_bgr: np.ndarray,
    min_face_size: Tuple[int, int] = (20, 20),
    margin_ratio: float = 0.15,
    confidence_threshold: float = 0.5,
) -> Optional[np.ndarray]:
    """
    Detect face in a person crop using a lightweight OpenCV Haar cascade.
    
    Args:
        person_bgr: BGR image array (H, W, 3)
        min_face_size: Minimum face size (not strictly enforced, for compatibility)
        margin_ratio: Padding ratio around face bbox (0.0-1.0)
        confidence_threshold: Detection confidence threshold (0.0-1.0)
    
    Returns:
        Cropped face image or None if no face detected.

    If no Haar cascade XML is available locally, returns a cheap upper-center
    heuristic crop instead of loading a deep model.

    Set FACE_DETECTOR_ENABLE_INSIGHTFACE_FALLBACK=1 to fall back to the
    heavier InsightFace detector if the Haar cascade misses a face.
    """
    if person_bgr is None or person_bgr.size == 0:
        return None

    if person_bgr.ndim != 3 or person_bgr.shape[2] != 3:
        raise ValueError("Expected BGR image with shape (H, W, 3).")

    try:
        face_crop = _detect_with_cascade(person_bgr, min_face_size, margin_ratio)
        if face_crop is not None:
            return face_crop

        if _CASCADE_DETECTOR is None:
            return _detect_with_heuristic(person_bgr, margin_ratio)

        if os.getenv("FACE_DETECTOR_ENABLE_INSIGHTFACE_FALLBACK", "0") == "1":
            return _detect_with_insightface(
                person_bgr,
                min_face_size=min_face_size,
                margin_ratio=margin_ratio,
                confidence_threshold=confidence_threshold,
            )

        return None
    except Exception as e:
        # Log but don't crash on detector errors
        raise RuntimeError(f"Face detection failed: {str(e)}")