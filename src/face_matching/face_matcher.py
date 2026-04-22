"""
Face matching with a lightweight default backend.

Backend priority (first available wins):
        1. OpenCV LBPH recognizer        – ultralight, best for edge stability
        2. insightface buffalo_sc        – MobileFaceNet (still lightweight)
        3. Raw ONNX session              – place any 112×112 → 512-d ArcFace model
                                                                             at <module_dir>/arcface.onnx

LBPH confidence is mapped to a similarity score in [0, 1] so the public API
stays consistent with the InsightFace backend.

Public API (backward-compatible with LBPH version):
    match_face_image(face_image)          → (name, confidence)
    match_image_group([img, ...])         → (name, confidence)
    train_and_save_pkl(pkl_path=None)     → pkl_path
    load_pkl(pkl_path=None)               → True
"""

import os
import sys
import math
import cv2
import numpy as np
import pickle

# Suppress OnnxRuntime shape-mismatch warnings that SCRFD emits on every call.
# Must be set before onnxruntime is imported (insightface imports it internally).
os.environ.setdefault("ORT_LOGGING_LEVEL", "3")  # 0=Verbose … 3=Error only
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="insightface")
try:
    import onnxruntime as _ort
    _ort.set_default_logger_severity(3)  # 3 = ERROR only (silences shape warnings)
except Exception:
    pass

# Make the face_detection module importable when this file is run directly
# (i.e. when BASE_DIR is src/face_matching, we need src/ on sys.path)
_SRC_DIR = os.path.dirname(os.path.dirname(__file__))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

BASE_DIR = os.path.dirname(__file__)

# ArcFace / MobileFaceNet standard input size
_IMG_SIZE = (112, 112)

# LBPH uses a smaller grayscale working size to keep runtime low.
_LBPH_SIZE = (96, 96)

# Cosine-similarity threshold: raise to be stricter, lower to be more lenient
DEFAULT_THRESHOLD = 0.25

# Where to look for per-person training image folders
_CANDIDATE_DIRS = [
    os.path.join(BASE_DIR, "known_faces"),
    os.path.join(BASE_DIR, "trained_faces"),
    os.path.join(os.path.dirname(BASE_DIR), "trained_faces"),
]

# Where the serialised embeddings live
_EMBEDDINGS_PKL = os.path.join(BASE_DIR, "embeddings.pkl")

# Where the serialised LBPH training data lives
_LBPH_PKL = os.path.join(BASE_DIR, "lbph_model.pkl")

# Bump this whenever the LBPH training pipeline changes so stale caches rebuild.
_LBPH_MODEL_VERSION = 2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_known_dir():
    for d in _CANDIDATE_DIRS:
        if os.path.isdir(d):
            return d
    return None


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D float vectors."""
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 1e-10 else 0.0


def _l2_normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-10 else v


def _preprocess_onnx(img_bgr: np.ndarray) -> np.ndarray:
    """Resize + normalize a BGR face crop to (1, 3, 112, 112) float32 [-1, 1]."""
    img = cv2.resize(img_bgr, _IMG_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    img = (img - 127.5) / 128.0
    img = np.transpose(img, (2, 0, 1))[np.newaxis]   # CHW + batch
    return img


def _preprocess_lbph(img_bgr: np.ndarray) -> np.ndarray:
    """Resize + normalize a BGR face crop for LBPH training / inference."""
    gray = cv2.cvtColor(cv2.resize(img_bgr, _LBPH_SIZE), cv2.COLOR_BGR2GRAY)
    return cv2.equalizeHist(gray)


def _lbph_distance_to_similarity(distance: float) -> float:
    """Map LBPH distance to a stable similarity score in [0, 1]."""
    return float(math.exp(-max(0.0, float(distance)) / 45.0))


# ---------------------------------------------------------------------------
# FaceMatcher
# ---------------------------------------------------------------------------

class FaceMatcher:
    """Face matcher for edge deployment (Rubik Pi 3 / QCS6490).

    The default backend is OpenCV LBPH because it is dramatically lighter than
    deep embedding inference. If that backend is unavailable, the matcher
    falls back to insightface buffalo_sc, then to a raw ONNX session.

    Confidence semantics:
        confidence ∈ [0, 1]  —  higher means a better match.
        A result below DEFAULT_THRESHOLD is treated as "not recognised".

    Usage::

        m = FaceMatcher()                          # loads / builds embeddings
        name, conf = m.match_face_image(crop)      # single face BGR ndarray
        name, conf = m.match_image_group([c1, c2]) # up to 3 crops, majority vote
    """

    def __init__(self, prefer_lightweight: bool = True, use_alignment: bool = False):
        self._mode: str = "none"
        self._session = None           # onnxruntime.InferenceSession (ONNX mode)
        self._input_name: str = ""
        self._rec_model = None         # insightface recognition model
        self._det_model = None         # SCRFD face detector for 5-pt alignment
        self._lbph_model = None        # OpenCV LBPH face recognizer
        self._lbph_label_to_name = {}  # label id -> person name
        self._lbph_name_to_label = {}  # person name -> label id
        self._lbph_samples_by_person = {}  # person name -> [grayscale face crops]
        self._prefer_lightweight = bool(prefer_lightweight)
        self._use_alignment = bool(use_alignment)
        self.embeddings: dict = {}     # {person_name: [np.ndarray(512,), ...]}
        self.known_dir = _find_known_dir()

        self._load_model()

        # Try to load cached training data; rebuild if stale / missing.
        try:
            if self._mode == "lbph":
                self._load_lbph_pkl()
            else:
                self._load_embeddings_pkl()
        except Exception:
            try:
                if self._mode == "lbph":
                    self._build_and_save_lbph_model()
                else:
                    self._build_and_save_embeddings()
            except Exception:
                pass  # No training data yet – caller must add faces and retrain

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self):
        """Load best available recognition back-end."""
        # ── 0. OpenCV LBPH – ultralight default on edge devices ─────────────
        if self._prefer_lightweight and hasattr(cv2, "face") and hasattr(cv2.face, "LBPHFaceRecognizer_create"):
            try:
                self._lbph_model = cv2.face.LBPHFaceRecognizer_create(
                    radius=1,
                    neighbors=8,
                    grid_x=8,
                    grid_y=8,
                )
                self._mode = "lbph"
                return
            except Exception:
                self._lbph_model = None
                self._mode = "none"

        # ── 1. insightface model_zoo – prefer lightweight buffalo_sc ─────────
        try:
            import insightface  # noqa: F401
            from insightface.model_zoo import get_model as _igfz_get_model

            model_name = "buffalo_sc" if self._prefer_lightweight else "buffalo_l"
            rec_file = "w600k_mbf.onnx" if self._prefer_lightweight else "w600k_r50.onnx"

            _buffalo_dir = os.path.join(
                os.path.expanduser("~"), ".insightface", "models", model_name
            )
            _rec_path = os.path.join(_buffalo_dir, rec_file)

            # Trigger download only when the selected file is missing
            if not os.path.exists(_rec_path):
                from insightface.app import FaceAnalysis as _FA
                _fa = _FA(
                    name=model_name,
                    allowed_modules=["recognition"],
                    providers=["CPUExecutionProvider"],
                )
                _fa.prepare(ctx_id=-1, det_size=_IMG_SIZE)

            if os.path.exists(_rec_path):
                rec = _igfz_get_model(_rec_path, providers=["CPUExecutionProvider"])
                rec.prepare(ctx_id=-1)
                self._rec_model = rec
                self._mode = "insightface"

                # Optional alignment: keep disabled by default for edge stability.
                _det_path = os.path.join(_buffalo_dir, "det_500m.onnx")
                if self._use_alignment and os.path.exists(_det_path):
                    try:
                        det = _igfz_get_model(_det_path,
                                              providers=["CPUExecutionProvider"])
                        det.prepare(ctx_id=-1, input_size=(128, 128),
                                    det_thresh=0.4)
                        self._det_model = det
                    except Exception:
                        pass  # alignment unavailable; fall back to raw get_feat
                return
        except Exception:
            pass

        # ── 2. FaceAnalysis fallback (iterate models for get_feat) ──────────
        try:
            import insightface  # noqa: F401
            from insightface.app import FaceAnalysis
            model_name = "buffalo_sc" if self._prefer_lightweight else "buffalo_l"
            app = FaceAnalysis(
                name=model_name,
                allowed_modules=["recognition"],
                providers=["CPUExecutionProvider"],
            )
            app.prepare(ctx_id=-1, det_size=_IMG_SIZE)

            rec = app.models.get("recognition")
            if rec is None:
                for _key, _m in list(
                    app.models.items()
                    if hasattr(app.models, "items") else []
                ):
                    if hasattr(_m, "get_feat"):
                        rec = _m
                        break
            if rec is None and hasattr(app, "rec_model"):
                rec = app.rec_model

            if rec is not None:
                self._rec_model = rec
                self._mode = "insightface"
                return
        except Exception:
            pass

        # ── 3. Raw ONNX session ──────────────────────────────────────────────
        # Check the module directory first, then the insightface cache.
        _onnx_candidates = [
            os.path.join(BASE_DIR, "arcface.onnx"),
            os.path.join(
                os.path.expanduser("~"),
                ".insightface", "models", "buffalo_sc", "w600k_mbf.onnx",
            ),
            os.path.join(
                os.path.expanduser("~"),
                ".insightface", "models", "buffalo_l", "w600k_r50.onnx",
            ),
        ]
        for onnx_path in _onnx_candidates:
            if os.path.exists(onnx_path):
                try:
                    import onnxruntime as ort
                    so = ort.SessionOptions()
                    so.intra_op_num_threads = 1
                    so.inter_op_num_threads = 1
                    self._session = ort.InferenceSession(
                        onnx_path,
                        sess_options=so,
                        providers=["CPUExecutionProvider"],
                    )
                    self._input_name = self._session.get_inputs()[0].name
                    self._mode = "onnx"
                    return
                except Exception:
                    pass

        onnx_path = os.path.join(BASE_DIR, "arcface.onnx")
        raise RuntimeError(
            "No face recognition back-end found.  "
            "Install insightface  OR  place a 112×112→512-d ArcFace ONNX "
            f"model at {onnx_path}."
        )

    def _collect_lbph_training_data(self):
        """Build training samples for the LBPH backend from known face folders."""
        if self.known_dir is None:
            raise FileNotFoundError(
                "Known faces directory not found. "
                f"Searched: {', '.join(_CANDIDATE_DIRS)}"
            )

        try:
            from face_detection.face_detector import detect_face as _detect_face
        except ImportError:
            _detect_face = None

        samples = []
        labels = []
        label_to_name = {}
        name_to_label = {}
        samples_by_person = {}

        for label_id, person_name in enumerate(sorted(os.listdir(self.known_dir))):
            person_path = os.path.join(self.known_dir, person_name)
            if not os.path.isdir(person_path):
                continue

            person_samples = []
            for filename in sorted(os.listdir(person_path)):
                file_path = os.path.join(person_path, filename)
                img = cv2.imread(file_path)
                if img is None:
                    continue

                crop = None
                if _detect_face is not None:
                    try:
                        crop = _detect_face(img)
                    except Exception:
                        crop = None
                if crop is None:
                    crop = img

                try:
                    person_samples.append(_preprocess_lbph(crop))
                except Exception:
                    continue

            if person_samples:
                label_to_name[label_id] = person_name
                name_to_label[person_name] = label_id
                samples_by_person[person_name] = list(person_samples)
                for sample in person_samples:
                    samples.append(sample)
                    labels.append(label_id)

        if not samples:
            raise ValueError("No training faces found in known faces directory.")

        return samples, labels, label_to_name, name_to_label, samples_by_person

    def _train_lbph_model(self):
        """Train the OpenCV LBPH recognizer from known-face images."""
        if self._lbph_model is None:
            raise RuntimeError("LBPH recognizer is not available.")

        samples, labels, label_to_name, name_to_label, samples_by_person = self._collect_lbph_training_data()
        self._lbph_model.train(samples, np.asarray(labels, dtype=np.int32))
        self._lbph_label_to_name = label_to_name
        self._lbph_name_to_label = name_to_label
        self._lbph_samples_by_person = samples_by_person
        self._mode = "lbph"
        self.embeddings = {}
        return samples, labels, label_to_name, name_to_label, samples_by_person

    def _build_and_save_lbph_model(self, pkl_path: str = None) -> str:
        """Train LBPH and persist the training bundle to pkl."""
        if pkl_path is None:
            pkl_path = _LBPH_PKL

        samples, labels, label_to_name, name_to_label, samples_by_person = self._train_lbph_model()
        with open(pkl_path, "wb") as fh:
            pickle.dump(
                {
                    "backend": "lbph",
                    "version": _LBPH_MODEL_VERSION,
                    "samples": samples,
                    "labels": labels,
                    "label_to_name": label_to_name,
                    "name_to_label": name_to_label,
                    "samples_by_person": samples_by_person,
                    "size": _LBPH_SIZE,
                },
                fh,
            )
        return pkl_path

    def _load_lbph_pkl(self, pkl_path: str = None) -> bool:
        """Load LBPH training data from pkl and retrain the recognizer."""
        if pkl_path is None:
            pkl_path = _LBPH_PKL

        if not os.path.exists(pkl_path):
            raise FileNotFoundError(f"LBPH pkl not found: {pkl_path}")

        with open(pkl_path, "rb") as fh:
            data = pickle.load(fh)

        if not isinstance(data, dict) or data.get("backend") != "lbph":
            raise ValueError("Unrecognised LBPH pkl format.")

        if data.get("version") != _LBPH_MODEL_VERSION:
            raise ValueError("Stale LBPH pkl format – retrain required.")

        samples = data.get("samples", [])
        labels = data.get("labels", [])
        label_to_name = data.get("label_to_name", {})
        name_to_label = data.get("name_to_label", {})
        samples_by_person = data.get("samples_by_person", {})

        if not samples_by_person and samples and labels and label_to_name:
            reconstructed = {}
            for sample, label_id in zip(samples, labels):
                person_name = label_to_name.get(int(label_id))
                if person_name is None:
                    continue
                reconstructed.setdefault(person_name, []).append(sample)
            samples_by_person = reconstructed

        if self._lbph_model is None:
            raise RuntimeError("LBPH recognizer is not available.")

        self._lbph_model.train(samples, np.asarray(labels, dtype=np.int32))
        self._lbph_label_to_name = label_to_name
        self._lbph_name_to_label = name_to_label
        self._lbph_samples_by_person = samples_by_person
        self._mode = "lbph"
        self.embeddings = {}
        return True

    # ------------------------------------------------------------------
    # Embedding extraction
    # ------------------------------------------------------------------

    def _align_face(self, face_bgr: np.ndarray) -> np.ndarray:
        """Return a 5-point landmark-aligned 112×112 crop, or a plain resize.

        When the SCRFD detector is available it runs on the incoming face crop
        to find landmarks, then uses insightface's ``norm_crop`` to produce
        the canonical aligned face that ArcFace/ResNet-50 was trained on.
        Without alignment, ResNet-50 embeddings degrade significantly.
        """
        if self._det_model is not None:
            try:
                from insightface.utils.face_align import norm_crop
                bboxes, kps = self._det_model.detect(face_bgr, max_num=1)
                if kps is not None and len(kps) > 0:
                    return norm_crop(face_bgr, landmark=kps[0])
            except Exception:
                pass
        # Fallback: plain resize (works well for MobileFaceNet)
        return cv2.resize(face_bgr, _IMG_SIZE)

    def _extract_embedding(self, face_bgr: np.ndarray) -> np.ndarray:
        """Return an L2-normalised 512-d embedding for a BGR face crop."""
        if face_bgr is None or face_bgr.size == 0:
            raise ValueError("Empty face image.")

        if self._mode == "insightface":
            img = self._align_face(face_bgr)
            emb = self._rec_model.get_feat(img).flatten()
        elif self._mode == "onnx":
            inp = _preprocess_onnx(face_bgr)
            emb = self._session.run(None, {self._input_name: inp})[0].flatten()
        else:
            raise RuntimeError("No recognition back-end loaded.")

        return _l2_normalize(emb)

    # ------------------------------------------------------------------
    # Training / persistence
    # ------------------------------------------------------------------

    def _build_and_save_embeddings(self, pkl_path: str = None) -> str:
        """Extract embeddings from all known-face images and save to pkl.

        Directory layout expected::

            known_faces/      (or trained_faces/)
                Alice/
                    photo1.jpg
                    photo2.jpg
                Bob/
                    photo1.jpg
        """
        if pkl_path is None:
            pkl_path = _EMBEDDINGS_PKL

        if self.known_dir is None:
            raise FileNotFoundError(
                "Known faces directory not found. "
                f"Searched: {', '.join(_CANDIDATE_DIRS)}"
            )

        # Use the project's face detector so that training crops are produced
        # by the same pipeline as inference crops (no cvzone dependency).
        try:
            from face_detection.face_detector import detect_face as _detect_face
        except ImportError:
            _detect_face = None

        embeddings: dict = {}
        for person_name in sorted(os.listdir(self.known_dir)):
            person_path = os.path.join(self.known_dir, person_name)
            if not os.path.isdir(person_path):
                continue

            person_embs = []
            for filename in os.listdir(person_path):
                file_path = os.path.join(person_path, filename)
                img = cv2.imread(file_path)
                if img is None:
                    continue

                # Try to extract a tight face crop; fall back to whole image
                # only as a last resort so training & inference embeddings
                # remain comparable.
                crop = None
                if _detect_face is not None:
                    try:
                        crop = _detect_face(img)
                    except Exception:
                        crop = None
                if crop is None:
                    crop = img  # whole-image fallback (better than skipping)

                try:
                    emb = self._extract_embedding(crop)
                    person_embs.append(emb)
                except Exception:
                    continue

            if person_embs:
                embeddings[person_name] = person_embs

        if not embeddings:
            raise ValueError("No training faces found in known faces directory.")

        with open(pkl_path, "wb") as fh:
            pickle.dump(embeddings, fh)

        self.embeddings = embeddings
        return pkl_path

    # Public alias (backward-compatible name)
    def train_and_save_pkl(self, pkl_path: str = None) -> str:
        """Build embeddings from known faces and persist to pkl. Returns pkl path."""
        if self._mode == "lbph":
            return self._build_and_save_lbph_model(pkl_path)
        return self._build_and_save_embeddings(pkl_path)

    def _load_embeddings_pkl(self, pkl_path: str = None) -> bool:
        if pkl_path is None:
            pkl_path = _EMBEDDINGS_PKL

        if not os.path.exists(pkl_path):
            raise FileNotFoundError(f"Embeddings pkl not found: {pkl_path}")

        with open(pkl_path, "rb") as fh:
            data = pickle.load(fh)

        # Reject old LBPH format automatically
        if isinstance(data, dict) and "label_map" in data:
            raise ValueError("Old LBPH pkl format – call train_and_save_pkl() to rebuild.")

        if not isinstance(data, dict):
            raise ValueError("Unrecognised pkl format.")

        self.embeddings = data
        return True

    # Public alias (backward-compatible name)
    def load_pkl(self, pkl_path: str = None) -> bool:
        """Load serialised embeddings from pkl. Returns True on success."""
        if self._mode == "lbph":
            return self._load_lbph_pkl(pkl_path)
        return self._load_embeddings_pkl(pkl_path)

    # ------------------------------------------------------------------
    # Matching
    # ------------------------------------------------------------------

    def match_face_image(self, face_image: np.ndarray,
                         threshold: float = DEFAULT_THRESHOLD):
        """Match a single pre-cropped face image.

        Parameters
        ----------
        face_image : np.ndarray
            BGR face crop (any size; will be resized internally).
        threshold : float
            Minimum cosine similarity to accept as a match (default 0.35).

        Returns
        -------
        (name, confidence) : (str | None, float)
            *confidence* is cosine similarity ∈ [0, 1] – **higher is better**.
            *name* is None when similarity is below *threshold*.
        """
        if face_image is None:
            raise ValueError("face_image is None.")
        if self._mode == "lbph":
            effective_threshold = min(float(threshold), 0.18)
            return self._match_face_image_lbph(face_image, effective_threshold)

        if not self.embeddings:
            raise RuntimeError(
                "No embeddings loaded. Call train_and_save_pkl() first."
            )
        query_emb = self._extract_embedding(face_image)
        return self._find_best_match([query_emb], threshold)

    def match_image_group(self, images: list,
                          threshold: float = DEFAULT_THRESHOLD):
        """Match 1–3 cropped face images of the same person (consensus vote).

        Parameters
        ----------
        images : list[np.ndarray]
            Between 1 and 3 BGR face crops.
        threshold : float
            Minimum average cosine similarity to accept as a match.

        Returns
        -------
        (name, confidence) : (str | None, float)
            Same semantics as :meth:`match_face_image`.
        """
        if not images:
            raise ValueError("No images provided to match_image_group.")
        if len(images) > 3:
            raise ValueError("A maximum of 3 images can be provided.")

        if self._mode == "lbph":
            return self._match_image_group_lbph(images, threshold)

        query_embs = []
        for img in images:
            if img is None:
                continue
            try:
                query_embs.append(self._extract_embedding(img))
            except Exception:
                continue

        if not query_embs:
            return (None, None)

        return self._find_best_match(query_embs, threshold)

    def _find_best_match(self, query_embs: list,
                         threshold: float):
        """Return (best_name, avg_similarity) using max-pooled cosine similarity.

        For each known person we compute, for every query embedding, the
        maximum cosine similarity against that person's stored embeddings, then
        average across queries.  This is robust to pose / lighting variation.
        """
        best_name: str | None = None
        best_sim: float = -1.0

        for person_name, stored_embs in self.embeddings.items():
            total = 0.0
            for q_emb in query_embs:
                # Best-match similarity for this query against all stored crops
                sims = [_cosine_similarity(q_emb, s) for s in stored_embs]
                total += max(sims)
            avg_sim = total / len(query_embs)

            if avg_sim > best_sim:
                best_sim = avg_sim
                best_name = person_name

        if best_sim < threshold:
            return (None, float(best_sim))
        return (best_name, float(best_sim))

    def _match_face_image_lbph(self, face_image: np.ndarray, threshold: float):
        """Match a single face image using the OpenCV LBPH backend."""
        scores = self._score_all_lbph_candidates(face_image)
        if not scores:
            return (None, None)

        best_name = next(iter(scores))
        best_confidence = float(scores[best_name])
        if best_confidence < threshold:
            return (None, best_confidence)
        return (best_name, best_confidence)

    def _match_image_group_lbph(self, images: list, threshold: float):
        """Match a small group of images using repeated LBPH predictions."""
        scores = {}
        votes = {}

        for image in images:
            if image is None:
                continue
            try:
                name, confidence = self._match_face_image_lbph(image, threshold)
            except Exception:
                continue
            if name is None:
                continue
            scores[name] = max(scores.get(name, 0.0), confidence)
            votes[name] = votes.get(name, 0) + 1

        if not scores:
            return (None, None)

        best_name = max(scores, key=lambda candidate: (votes.get(candidate, 0), scores[candidate]))
        return (best_name, float(scores[best_name]))

    def _score_all_lbph_candidates(self, face_image: np.ndarray) -> dict:
        """Score a face image against every trained LBPH identity."""
        if self._lbph_model is None:
            raise RuntimeError("LBPH recognizer is not loaded.")
        if face_image is None or not self._lbph_samples_by_person:
            return {}

        query_face = _preprocess_lbph(face_image)
        scores = {}
        for person_name, stored_faces in self._lbph_samples_by_person.items():
            if not stored_faces:
                continue
            best_sim = max(
                self._score_lbph_against_person(query_face, stored_face)
                for stored_face in stored_faces
            )
            scores[person_name] = float(best_sim)

        return dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))

    def _score_lbph_against_person(self, query_face: np.ndarray, stored_face: np.ndarray) -> float:
        """Cheap similarity score between two preprocessed LBPH face crops."""
        q = query_face.astype(np.float32)
        s = stored_face.astype(np.float32)
        diff = cv2.norm(q, s, cv2.NORM_L2)
        return float(math.exp(-diff / 4000.0))

    def get_all_match_scores(self, face_image: np.ndarray) -> dict:
        """Get similarity scores for a face against all known persons (for debugging).
        
        Returns dict with all person names and their similarity scores, sorted descending.
        Useful for understanding why a particular match was selected.
        
        Parameters
        ----------
        face_image : np.ndarray
            BGR face crop.
        
        Returns
        -------
        dict : {person_name: similarity_score, ...}
            All candidates sorted by score descending.
        """
        if face_image is None:
            return {}
        
        try:
            if self._mode == "lbph":
                return self._score_all_lbph_candidates(face_image)

            if not self.embeddings:
                return {}

            query_emb = self._extract_embedding(face_image)
            scores = {}
            
            # Calculate similarity for each person
            for person_name, stored_embs in self.embeddings.items():
                sims = [_cosine_similarity(query_emb, s) for s in stored_embs]
                best_sim = max(sims) if sims else 0.0
                scores[person_name] = float(best_sim)
            
            # Sort by score descending
            return dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))
        except Exception:
            return {}
