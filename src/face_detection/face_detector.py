# face_detector.py
# Face Detection Module
# Takes a cropped person image from human detection input and returns the cropped face

from __future__ import annotations

import os
from typing import Optional, Tuple

import cv2
import numpy as np

def _clip_box(
    x: int, y: int, w: int, h: int, W: int, H: int
) -> Tuple[int, int, int, int]:
    x = max(0, x)
    y = max(0, y)
    w = max(1, min(w, W - x))
    h = max(1, min(h, H - y))
    return x, y, w, h

def detect_face(
    person_bgr: np.ndarray,
    min_face_size: Tuple[int, int] = (30, 30),
    margin_ratio: float = 0.15,
) -> Optional[np.ndarray]: 
    
    if person_bgr is None or person_bgr.size == 0:
        return None

    if person_bgr.ndim != 3 or person_bgr.shape[2] != 3:
        raise ValueError("Expected BGR image with shape (H, W, 3).")

    H, W = person_bgr.shape[:2]

    gray = cv2.cvtColor(person_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    # --- THE FIX ---
    # Look for the file in the same directory as this script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    cascade_path = os.path.join(current_dir, "haarcascade_frontalface_default.xml")

    # If it's not there, check the current working directory (SeniorDesign folder)
    if not os.path.exists(cascade_path):
        cascade_path = os.path.join(os.getcwd(), "haarcascade_frontalface_default.xml")

    face_cascade = cv2.CascadeClassifier(cascade_path)

    if face_cascade.empty():
        raise RuntimeError(
            f"XML file not found at {cascade_path}. "
            "Please run: wget https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
        )
    # ---------------

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=min_face_size,
        flags=cv2.CASCADE_SCALE_IMAGE,
    )

    if len(faces) == 0:
        return None

    # Select largest face
    x, y, w, h = max(faces, key=lambda b: b[2] * b[3])

    # Apply margin padding
    mx = int(w * margin_ratio)
    my = int(h * margin_ratio)

    x2, y2, w2, h2 = _clip_box(x - mx, y - my, w + 2 * mx, h + 2 * my, W, H)

    face_crop = person_bgr[y2 : y2 + h2, x2 : x2 + w2]

    return face_crop if face_crop.size > 0 else None