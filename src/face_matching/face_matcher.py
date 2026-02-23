import os
import cv2
import numpy as np
import pickle
import tempfile

BASE_DIR = os.path.dirname(__file__)
IMG_SIZE = (200, 200)

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


class FaceMatcher:
    """Encapsulates an LBPH face recognizer and label map.

    Usage:
    - Create instance: `m = FaceMatcher()`
    - Call `m.load(model_path, labels_path)` to load saved model
    - Or call `m.train_and_save(...)` to build from known faces and persist
    - Use `m.match_face_image(img)` and `m.match_image_group([imgs...])`
    """

    def __init__(self):
        self.recognizer = None
        self.label_map = {}
        self.known_dir = _find_known_dir()
        # Try to load existing PKL model, otherwise train and persist one
        try:
            self.load_pkl()
        except Exception:
            self.train_and_save_pkl()

    def _load_training_data(self):
        try:
            from cvzone.FaceDetectionModule import FaceDetector
        except Exception:
            FaceDetector = None

        faces = []
        labels = []
        label_map = {}
        current_label = 0

        detector = FaceDetector() if FaceDetector is not None else None

        if self.known_dir is None:
            raise FileNotFoundError(f"Known faces directory not found. Searched: {', '.join(_CANDIDATE_DIRS)}")

        for person_name in os.listdir(self.known_dir):
            person_path = os.path.join(self.known_dir, person_name)

            if not os.path.isdir(person_path):
                continue

            label_map[current_label] = person_name

            for filename in os.listdir(person_path):
                file_path = os.path.join(person_path, filename)
                img = cv2.imread(file_path)

                if img is None:
                    continue

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
            raise ValueError("No training faces found in known faces directory")

        faces = np.array(faces)
        labels = np.array(labels)

        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.train(faces, labels)

        return recognizer, label_map

    # NOTE: legacy separate-file save/load removed. Use PKL helpers below.

    def train_and_save_pkl(self, pkl_path=None):
        """Train and save a single .pkl containing model bytes and label_map."""
        if pkl_path is None:
            pkl_path = os.path.join(BASE_DIR, "recognizer.pkl")

        recognizer, label_map = self._load_training_data()

        # write recognizer to a temp file and read bytes
        with tempfile.NamedTemporaryFile(suffix=".yml", delete=False) as tf:
            tmp_name = tf.name
        try:
            recognizer.write(tmp_name)
            with open(tmp_name, "rb") as f:
                model_bytes = f.read()
        finally:
            try:
                os.remove(tmp_name)
            except Exception:
                pass

        data = {"model_bytes": model_bytes, "label_map": label_map}
        with open(pkl_path, "wb") as f:
            pickle.dump(data, f)

        self.recognizer = recognizer
        self.label_map = label_map
        return pkl_path

    def load_pkl(self, pkl_path=None):
        if pkl_path is None:
            pkl_path = os.path.join(BASE_DIR, "recognizer.pkl")

        if not os.path.exists(pkl_path):
            raise FileNotFoundError("PKL model not found. Call train_and_save_pkl() first.")

        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        model_bytes = data.get("model_bytes")
        label_map = data.get("label_map")

        # write bytes to a temp file and read with recognizer
        with tempfile.NamedTemporaryFile(suffix=".yml", delete=False) as tf:
            tmp_name = tf.name
            tf.write(model_bytes)
            tf.flush()
        try:
            recognizer = cv2.face.LBPHFaceRecognizer_create()
            recognizer.read(tmp_name)
        finally:
            try:
                os.remove(tmp_name)
            except Exception:
                pass

        self.recognizer = recognizer
        self.label_map = label_map
        return True

    def match_face_image(self, face_image):
        """Accept a cropped face image (NumPy BGR) and return (name, confidence).

        `name` will be None if no recognizer is loaded.
        """
        if self.recognizer is None:
            raise RuntimeError("Recognizer not loaded. Call load_pkl() or train_and_save_pkl() first.")

        if face_image is None:
            raise ValueError("face_image is None")

        img = face_image
        if len(img.shape) == 3 and img.shape[2] == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        gray = cv2.resize(gray, IMG_SIZE)

        label_id, confidence = self.recognizer.predict(gray)
        name = self.label_map.get(label_id, None)
        return (name, float(confidence))

    def match_image_group(self, images):
        if not images:
            raise ValueError("No images provided to match_image_group")
        if len(images) > 3:
            raise ValueError("A maximum of 3 images can be provided")

        votes = []
        for img in images:
            if img is None:
                continue
            if len(img.shape) == 3 and img.shape[2] == 3:
                g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                g = img
            g = cv2.resize(g, IMG_SIZE)
            lid, conf_raw = self.recognizer.predict(g)
            votes.append((int(lid), float(conf_raw)))

        if not votes:
            return (None, None)

        stats = {}
        for lid, conf in votes:
            s = stats.setdefault(lid, {"sum": 0.0, "count": 0})
            s["sum"] += conf
            s["count"] += 1

        best = min(stats.items(), key=lambda kv: kv[1]["sum"] / kv[1]["count"]) if stats else None
        if not best:
            return (None, None)
        best_lid, best_s = best[0], best[1]
        avg_conf = best_s["sum"] / best_s["count"]
        name = self.label_map.get(best_lid, None)
        return (name, float(avg_conf))
