# face_detector.py
# Face Detection Module
# Takes a cropped person image from human detection input and returns the cropped face

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

try:
    from insightface.app import FaceAnalysis
except ImportError:
    raise ImportError("insightface is required for face detection. Install with: pip install insightface>=0.7.3")

#makes sure the cropped image is within the bounds of the original image
def _clip_box(
    x: int, y: int, w: int, h: int, W: int, H: int
) -> Tuple[int, int, int, int]:
    x = max(0, x)
    y = max(0, y)
    w = max(1, min(w, W - x))
    h = max(1, min(h, H - y))
    return x, y, w, h


# Initialize insightface RetinaFace detector once at import time
# Detection plugin provides multi-task learning face detection with landmarks and angles
_face_detector = None

def _init_detector():
    """Initialize insightface detector (lazy load)"""
    global _face_detector
    if _face_detector is None:
        _face_detector = FaceAnalysis(name='buffalo_sc', providers=['CPUExecutionProvider'])
        _face_detector.prepare(ctx_id=-1, det_size=(640, 640))
    return _face_detector


def detect_face(
    person_bgr: np.ndarray,
    min_face_size: Tuple[int, int] = (20, 20),
    margin_ratio: float = 0.15,
    confidence_threshold: float = 0.5,
) -> Optional[np.ndarray]:
    """
    Detect face in a person crop using insightface RetinaFace detector.
    
    Args:
        person_bgr: BGR image array (H, W, 3)
        min_face_size: Minimum face size (not strictly enforced, for compatibility)
        margin_ratio: Padding ratio around face bbox (0.0-1.0)
        confidence_threshold: Detection confidence threshold (0.0-1.0)
    
    Returns:
        Cropped face image or None if no face detected
    """
    if person_bgr is None or person_bgr.size == 0:
        return None

    if person_bgr.ndim != 3 or person_bgr.shape[2] != 3:
        raise ValueError("Expected BGR image with shape (H, W, 3).")

    H, W = person_bgr.shape[:2]
    
    # Convert BGR to RGB for insightface
    person_rgb = cv2.cvtColor(person_bgr, cv2.COLOR_BGR2RGB)
    
    try:
        detector = _init_detector()
        # detect() returns list of Face objects with bboxes
        faces = detector.get(person_rgb)
        
        if not faces:
            return None
        
        # Select face with highest confidence
        best_face = max(faces, key=lambda f: f.det_score)
        
        # Skip if confidence is below threshold
        if best_face.det_score < confidence_threshold:
            return None
        
        # Extract bbox: [x1, y1, x2, y2]
        bbox = best_face.bbox.astype(int)
        x, y, x_end, y_end = bbox[0], bbox[1], bbox[2], bbox[3]
        
        # Convert to width/height format
        w = x_end - x
        h = y_end - y
        
        # Apply margin padding
        mx = int(w * margin_ratio)
        my = int(h * margin_ratio)
        
        # Clip to image bounds
        x2, y2, w2, h2 = _clip_box(x - mx, y - my, w + 2 * mx, h + 2 * my, W, H)
        
        face_crop = person_bgr[y2 : y2 + h2, x2 : x2 + w2]
        
        if face_crop.size == 0:
            return None
        
        return face_crop
        
    except Exception as e:
        # Log but don't crash on detector errors
        raise RuntimeError(f"Face detection failed: {str(e)}")