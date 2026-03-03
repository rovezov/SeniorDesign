"""
Face matching via deep-learning embeddings (ArcFace / MobileFaceNet).

Backend priority (first available wins):
  1. insightface buffalo_l   – ResNet-50, WebFace600K, ~166 MB ONNX
  2. Raw ONNX session        – place any 112×112 → 512-d ArcFace model at
                               <module_dir>/arcface.onnx

Confidence is **cosine similarity** in [0, 1] (higher = better match).
A value below DEFAULT_THRESHOLD means "not recognised".

Public API (backward-compatible with LBPH version):
  match_face_image(face_image)          → (name, confidence)
  match_image_group([img, ...])         → (name, confidence)
  train_and_save_pkl(pkl_path=None)     → pkl_path
  load_pkl(pkl_path=None)               → True
"""

import os
import sys
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


# ---------------------------------------------------------------------------
# FaceMatcher
# ---------------------------------------------------------------------------

class FaceMatcher:
    """Embedding-based face matcher for edge deployment (RubikPi 3 / QCS6490).

    Uses ArcFace (MobileFaceNet backbone via insightface buffalo_sc **or** a raw
    ONNX session) to extract 512-d face embeddings, then ranks candidates by
    cosine similarity.  Far more accurate than LBPH and runs efficiently via
    ONNX Runtime on-device.

    Confidence semantics (CHANGED from LBPH version):
        confidence ∈ [0, 1]  —  higher means a better match.
        A result below DEFAULT_THRESHOLD (0.35) is treated as "not recognised".

    Usage::

        m = FaceMatcher()                          # loads / builds embeddings
        name, conf = m.match_face_image(crop)      # single face BGR ndarray
        name, conf = m.match_image_group([c1, c2]) # up to 3 crops, majority vote
    """

    def __init__(self):
        self._mode: str = "none"
        self._session = None           # onnxruntime.InferenceSession (ONNX mode)
        self._input_name: str = ""
        self._rec_model = None         # insightface recognition model
        self._det_model = None         # SCRFD face detector for 5-pt alignment
        self.embeddings: dict = {}     # {person_name: [np.ndarray(512,), ...]}
        self.known_dir = _find_known_dir()

        self._load_model()

        # Try to load cached embeddings; rebuild if stale / missing
        try:
            self._load_embeddings_pkl()
        except Exception:
            try:
                self._build_and_save_embeddings()
            except Exception:
                pass  # No training data yet – caller must add faces and retrain

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self):
        """Load best available recognition back-end."""
        # ── 1. insightface model_zoo – load w600k_mbf directly ──────────────
        # Prefer this over FaceAnalysis because newer insightface versions key
        # app.models by filename (e.g. "w600k_mbf") rather than task name
        # ("recognition"), causing app.models.get("recognition") to return None.
        try:
            import insightface  # noqa: F401
            from insightface.model_zoo import get_model as _igfz_get_model

            # buffalo_l uses ResNet-50 (w600k_r50.onnx) – much more accurate
            # than buffalo_sc's MobileFaceNet (w600k_mbf.onnx).
            _buffalo_dir = os.path.join(
                os.path.expanduser("~"), ".insightface", "models", "buffalo_l"
            )
            _r50_path = os.path.join(_buffalo_dir, "w600k_r50.onnx")

            # Trigger download only when the file is missing
            if not os.path.exists(_r50_path):
                from insightface.app import FaceAnalysis as _FA
                _fa = _FA(
                    name="buffalo_l",
                    allowed_modules=["recognition"],
                    providers=["CPUExecutionProvider"],
                )
                _fa.prepare(ctx_id=-1, det_size=_IMG_SIZE)

            if os.path.exists(_r50_path):
                rec = _igfz_get_model(_r50_path, providers=["CPUExecutionProvider"])
                rec.prepare(ctx_id=-1)
                self._rec_model = rec
                self._mode = "insightface"

                # Load SCRFD detector for 5-point landmark alignment.
                # ResNet-50 ArcFace needs aligned crops; without this step
                # recognition accuracy drops significantly.
                _det_path = os.path.join(_buffalo_dir, "det_10g.onnx")
                if os.path.exists(_det_path):
                    try:
                        det = _igfz_get_model(_det_path,
                                              providers=["CPUExecutionProvider"])
                        # Use a compact input size; the crop is already a tight
                        # person bbox so 128×128 is sufficient.
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
            app = FaceAnalysis(
                name="buffalo_l",
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
                ".insightface", "models", "buffalo_l", "w600k_r50.onnx",
            ),
            # Fall back to the smaller model if buffalo_l was never downloaded
            os.path.join(
                os.path.expanduser("~"),
                ".insightface", "models", "buffalo_sc", "w600k_mbf.onnx",
            ),
        ]
        for onnx_path in _onnx_candidates:
            if os.path.exists(onnx_path):
                try:
                    import onnxruntime as ort
                    self._session = ort.InferenceSession(
                        onnx_path,
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
