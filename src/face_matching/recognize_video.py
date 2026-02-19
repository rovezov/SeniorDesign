import cv2
import os
import sys
import numpy as np

BASE_DIR = os.path.dirname(__file__)

# Candidate locations for training data (search in this order)
_CANDIDATE_DIRS = [
    os.path.join(BASE_DIR, "known_faces"),
    os.path.join(BASE_DIR, "trained_faces"),
    os.path.join(os.path.dirname(BASE_DIR), "trained_faces"),
]


def _find_known_dir():
    for d in _CANDIDATE_DIRS:
        if os.path.isdir(d):
            return d
    return None


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

    if KNOWN_DIR is None:
        raise FileNotFoundError(f"Known faces directory not found. Searched: {', '.join(_CANDIDATE_DIRS)}")

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
def match_face_image(face_image):
    """Minimal: accept a cropped face image (NumPy BGR) and return (name, confidence).

    Returns (name, confidence) where `name` is None when not recognized (confidence still provided).
    """
    if recognizer is None:
        raise RuntimeError("Recognizer not initialized. Check training data in KNOWN_DIR.")

    if face_image is None:
        raise ValueError("face_image is None")

    img = face_image

    # Convert to gray if necessary
    if len(img.shape) == 3 and img.shape[2] == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    gray = cv2.resize(gray, IMG_SIZE)

    label_id, confidence = recognizer.predict(gray)
    name = LABEL_MAP.get(label_id, "Unknown")
    return (name, float(confidence))


def match_image_group(images):
    """Minimal: accept 1-3 face images and return (name, confidence).

    Aggregation: choose the label closest on average among images that produced predictions.
    Returns (name, confidence) with `name` None if no consensus match.
    """
    if not images:
        raise ValueError("No images provided to match_image_group")

    if len(images) > 3:
        raise ValueError("A maximum of 3 images can be provided")

    # collect (label_id, confidence) for each image
    votes = []
    for img in images:
        img_name, conf = match_face_image(img)
        # If match_face_image returns None name, still record predicted label via recognizer
        # We'll call recognizer directly to get label for aggregation to avoid extra outputs
        if img is None:
            continue
        # convert img to gray and predict
        if len(img.shape) == 3 and img.shape[2] == 3:
            g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            g = img
        g = cv2.resize(g, IMG_SIZE)
        lid, conf_raw = recognizer.predict(g)
        votes.append((int(lid), float(conf_raw)))

    if not votes:
        return (None, None)

    # aggregate: group by label id
    stats = {}
    for lid, conf in votes:
        s = stats.setdefault(lid, {"sum": 0.0, "count": 0})
        s["sum"] += conf
        s["count"] += 1

    # pick label with lowest average confidence
    best = min(stats.items(), key=lambda kv: kv[1]["sum"] / kv[1]["count"]) if stats else None
    if not best:
        return (None, None)
    best_lid, best_s = best[0], best[1]
    avg_conf = best_s["sum"] / best_s["count"]
    name = LABEL_MAP.get(best_lid, "Unknown")
    return (name, float(avg_conf))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python recognize_video.py /path/to/cropped_face.jpg")
        sys.exit(1)

    face_path = sys.argv[1]
    try:
        img = cv2.imread(face_path)
        result = match_face_image(img)
        print(result)
    except Exception as exc:
        print(f"Error: {exc}")
        sys.exit(2)