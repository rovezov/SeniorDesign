import argparse
import cv2
import os
import sys
import numpy as np

def _find_known_dir():
    """Find a directory containing training images.

    Tries several common locations so the script works when training
    data is placed in a sibling `trained_faces` directory or a
    `known_faces` directory next to this module.
    """
    base = os.path.dirname(__file__)
    candidates = [
        os.path.join(base, "known_faces"),
        os.path.join(base, "trained_faces"),
        os.path.join(base, os.pardir, "trained_faces"),
        os.path.join(base, os.pardir, os.pardir, "trained_faces"),
        os.path.join(os.getcwd(), "trained_faces"),
    ]

    for p in candidates:
        p = os.path.abspath(p)
        if os.path.isdir(p):
            return p

    # If nothing found, raise with helpful diagnostics
    raise FileNotFoundError(
        "Known faces directory not found. Tried: " + ", ".join(os.path.abspath(p) for p in candidates)
    )

KNOWN_DIR = _find_known_dir()
IMG_SIZE = (200, 200)   # all faces resized to same size for LBPH


def _load_training_data():
    """Load faces from KNOWN_DIR, detect/crop faces in each image and
    return trained recognizer and label map."""
    try:
        from cvzone.FaceDetectionModule import FaceDetector
    except Exception:
        FaceDetector = None

    faces = []
    labels = []
    label_map = {}
    current_label = 0

    detector = FaceDetector() if FaceDetector is not None else None

    if not os.path.isdir(KNOWN_DIR):
        raise FileNotFoundError(f"Known faces directory not found: {KNOWN_DIR}")

    for person_name in os.listdir(KNOWN_DIR):
        person_path = os.path.join(KNOWN_DIR, person_name)

        if not os.path.isdir(person_path):
            continue

        label_map[current_label] = person_name

        for filename in os.listdir(person_path):
            file_path = os.path.join(person_path, filename)
            img = cv2.imread(file_path)

            if img is None:
                continue

            # If we have a detector, crop the face; otherwise assume image is already a face
            if detector is not None:
                _, bboxs = detector.findFaces(img, draw=False)
                if bboxs:
                    x, y, w, h = bboxs[0]["bbox"]
                    face_crop = img[y:y+h, x:x+w]
                else:
                    continue
            else:
                face_crop = img

            gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, IMG_SIZE)

            faces.append(gray)
            labels.append(current_label)

        current_label += 1

    if not faces:
        raise ValueError("No training faces found in KNOWN_DIR")

    faces = np.array(faces)
    labels = np.array(labels)

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, labels)

    return recognizer, label_map


# Initialize recognizer at import so the module is ready to match images
try:
    recognizer, LABEL_MAP = _load_training_data()
except Exception as e:
    recognizer = None
    LABEL_MAP = {}


def match_face_image(face_image_path, threshold=100):
    """Match a single cropped face image file against the trained recognizer.

    Args:
        face_image_path (str): Path to the cropped face image file.
        threshold (float): Confidence threshold (lower means better match). Higher
            values are more permissive (default 100).

    Returns:
        dict: {"label_id": int, "name": str, "confidence": float, "match": bool}
    """
    if recognizer is None:
        raise RuntimeError("Recognizer not initialized. Check training data in KNOWN_DIR.")

    img = cv2.imread(face_image_path)
    if img is None:
        raise FileNotFoundError(f"Face image not found: {face_image_path}")

    # Convert to gray if necessary
    if len(img.shape) == 3 and img.shape[2] == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    gray = cv2.resize(gray, IMG_SIZE)

    label_id, confidence = recognizer.predict(gray)
    match = confidence < threshold
    name = LABEL_MAP.get(label_id, "Unknown") if match else "Unknown"

    return {"label_id": int(label_id), "name": name, "confidence": float(confidence), "match": bool(match)}


def match_image_group(image_paths, threshold=100):
    """Match multiple images (up to 3) that belong to the same test person.

    Returns a dict with per-image results and an overall verdict using a
    simple majority rule: the label with the most matching images wins; ties
    resolved by lowest average confidence. If no image produces a match,
    overall `match` is False.
    """
    if not image_paths:
        raise ValueError("No image paths provided to match_image_group")

    if len(image_paths) > 3:
        raise ValueError("A maximum of 3 images can be provided for a single test person")

    per_image = []
    for p in image_paths:
        res = match_face_image(p, threshold=threshold)
        per_image.append(res)

    # Count matched labels
    label_stats = {}
    for r in per_image:
        lid = r["label_id"]
        matched = r["match"]
        label_stats.setdefault(lid, {"matches": 0, "conf_sum": 0.0, "count": 0})
        if matched:
            label_stats[lid]["matches"] += 1
        label_stats[lid]["conf_sum"] += r["confidence"]
        label_stats[lid]["count"] += 1

    # Determine best label by most matches then lowest avg confidence
    best_label = None
    best_score = None
    for lid, s in label_stats.items():
        # Primary key: number of matches (higher is better)
        # Secondary key: average confidence (lower is better)
        avg_conf = s["conf_sum"] / s["count"]
        key = (s["matches"], -avg_conf)
        if best_score is None or key > best_score:
            best_score = key
            best_label = lid

    overall_match = False
    overall_name = "Unknown"
    overall_confidence = None

    if best_label is not None:
        stats = label_stats[best_label]
        overall_confidence = stats["conf_sum"] / stats["count"]
        overall_name = LABEL_MAP.get(best_label, "Unknown")
        overall_match = stats["matches"] > 0

    return {
        "per_image": per_image,
        "overall": {"label_id": int(best_label) if best_label is not None else None,
                    "name": overall_name,
                    "confidence": float(overall_confidence) if overall_confidence is not None else None,
                    "match": bool(overall_match)}
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Match cropped face images against trained faces and print a single label."
    )
    parser.add_argument("paths", nargs='+', help="One or more image paths, or a single directory of test images")
    parser.add_argument("--threshold", type=float, default=100.0,
                        help="Confidence threshold (lower = stricter). Default: 100")

    args = parser.parse_args()
    targets = args.paths

    overall_name = "Unknown"

    try:
        # Directory mode: single argument that is a directory
        if len(targets) == 1 and os.path.isdir(targets[0]):
            target = targets[0]
            exts = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff")
            label_counts = {}
            for root, _, files in os.walk(target):
                for fn in files:
                    if not fn.lower().endswith(exts):
                        continue
                    fp = os.path.join(root, fn)
                    try:
                        res = match_face_image(fp, threshold=args.threshold)
                        if res.get("match"):
                            lid = res.get("label_id")
                            label_counts[lid] = label_counts.get(lid, 0) + 1
                    except Exception:
                        # ignore unreadable files or prediction errors
                        continue

            if label_counts:
                # choose label with highest count; tie-breaker: lowest average id (stable)
                best_label = max(label_counts.items(), key=lambda x: (x[1], -x[0]))[0]
                overall_name = LABEL_MAP.get(best_label, "Unknown")
            else:
                overall_name = "Unknown"
        else:
            # File mode: one or more file paths (treat multiple as same person group)
            if len(targets) > 3:
                overall_name = "Unknown"
            elif len(targets) == 1:
                try:
                    result = match_face_image(targets[0], threshold=args.threshold)
                    overall_name = result.get("name", "Unknown")
                except Exception:
                    overall_name = "Unknown"
            else:
                try:
                    res = match_image_group(targets, threshold=args.threshold)
                    overall_name = res["overall"].get("name", "Unknown")
                except Exception:
                    overall_name = "Unknown"

    except Exception:
        overall_name = "Unknown"

    # Final output: single label only
    print(overall_name)