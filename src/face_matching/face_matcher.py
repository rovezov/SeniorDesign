# face_matcher.py
"""
Face matching via deep-learning embeddings (MobileFaceNet).
Locked to pure ONNX Runtime (w600k_mbf.onnx) with strictly 1 thread.

Confidence is cosine similarity in [0, 1] (higher = better match).
A value below DEFAULT_THRESHOLD means "not recognised".
"""

import os
import sys
import cv2
import numpy as np
import pickle
import warnings

# Suppress OnnxRuntime noise
os.environ.setdefault("ORT_LOGGING_LEVEL", "3")
warnings.filterwarnings("ignore", category=FutureWarning)

# Ensure internal modules are importable
_SRC_DIR = os.path.dirname(os.path.dirname(__file__))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

BASE_DIR = os.path.dirname(__file__)

# Model config
_IMG_SIZE = (112, 112)
DEFAULT_THRESHOLD = 0.30

# Path config
_CANDIDATE_DIRS = [
    os.path.join(BASE_DIR, "known_faces"),
    os.path.join(BASE_DIR, "trained_faces"),
    os.path.join(os.path.dirname(BASE_DIR), "trained_faces"),
]
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
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 1e-10 else 0.0

def _l2_normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-10 else v

# ---------------------------------------------------------------------------
# FaceMatcher
# ---------------------------------------------------------------------------

class FaceMatcher:
    """
    Locked-down FaceMatcher using Pure ONNX Runtime.
    Strictly limited to 1 thread to prevent hardware bus lockups.
    """

    def __init__(self):
        self._session = None
        self._input_name = ""
        self.embeddings: dict = {}
        self.known_dir = _find_known_dir()

        self._load_pure_onnx_model()

        # Load or rebuild embeddings
        try:
            self._load_embeddings_pkl()
        except Exception:
            try:
                self._build_and_save_embeddings()
            except Exception:
                pass

    def _load_pure_onnx_model(self):
        """Bypasses insightface wrapper and clamps threads to 1."""
        import onnxruntime as ort

        # Path to where insightface previously downloaded the model
        _model_path = os.path.expanduser(
            "~/.insightface/models/buffalo_sc/w600k_mbf.onnx"
        )

        if not os.path.exists(_model_path):
            raise RuntimeError(
                f"Model missing at {_model_path}. "
                "Please ensure the ONNX file is located at this path."
            )

        # --- THE MAGIC FIX ---
        # Clamp the thread pool to 1 to stop the CPU/FastRPC power spike
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC

        # Force CPU execution to avoid accidental DSP mapping
        self._session = ort.InferenceSession(
            _model_path,
            sess_options=so,
            providers=["CPUExecutionProvider"]
        )
        self._input_name = self._session.get_inputs()[0].name

    def _extract_embedding(self, face_bgr: np.ndarray) -> np.ndarray:
        """Raw preprocessing and extraction (no insightface overhead)."""
        if face_bgr is None or face_bgr.size == 0:
            raise ValueError("Empty face image.")

        # Manual Preprocessing for ArcFace / MobileFaceNet
        img = cv2.resize(face_bgr, _IMG_SIZE)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        
        # Normalize to [-1, 1]
        img = (img - 127.5) / 128.0
        
        # Transpose to Channels-First (CHW) and add Batch dimension
        img = np.transpose(img, (2, 0, 1))[np.newaxis] 

        # Run Inference
        emb = self._session.run(None, {self._input_name: img})[0].flatten()
        
        return _l2_normalize(emb)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _build_and_save_embeddings(self, pkl_path: str = None) -> str:
        if pkl_path is None:
            pkl_path = _EMBEDDINGS_PKL

        if self.known_dir is None:
            raise FileNotFoundError("Known faces directory not found.")

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
                img = cv2.imread(os.path.join(person_path, filename))
                if img is None:
                    continue

                # Use project face detector if available
                crop = None
                if _detect_face:
                    try: crop = _detect_face(img)
                    except: pass
                
                if crop is None: crop = img

                try:
                    emb = self._extract_embedding(crop)
                    person_embs.append(emb)
                except:
                    continue

            if person_embs:
                embeddings[person_name] = person_embs

        with open(pkl_path, "wb") as fh:
            pickle.dump(embeddings, fh)

        self.embeddings = embeddings
        return pkl_path

    def train_and_save_pkl(self, pkl_path: str = None) -> str:
        return self._build_and_save_embeddings(pkl_path)

    def _load_embeddings_pkl(self, pkl_path: str = None) -> bool:
        if pkl_path is None:
            pkl_path = _EMBEDDINGS_PKL

        if not os.path.exists(pkl_path):
            raise FileNotFoundError(f"Embeddings pkl not found: {pkl_path}")

        with open(pkl_path, "rb") as fh:
            data = pickle.load(fh)

        if not isinstance(data, dict) or "label_map" in data:
            raise ValueError("Invalid or old LBPH pkl format.")

        self.embeddings = data
        return True

    def load_pkl(self, pkl_path: str = None) -> bool:
        return self._load_embeddings_pkl(pkl_path)

    # ------------------------------------------------------------------
    # Matching Logic
    # ------------------------------------------------------------------

    def match_face_image(self, face_image: np.ndarray, threshold: float = DEFAULT_THRESHOLD):
        if face_image is None:
            raise ValueError("face_image is None.")
        if not self.embeddings:
            raise RuntimeError("No embeddings loaded.")
        
        query_emb = self._extract_embedding(face_image)
        return self._find_best_match([query_emb], threshold)

    def match_image_group(self, images: list, threshold: float = DEFAULT_THRESHOLD):
        query_embs = []
        for img in images:
            if img is None: continue
            try: query_embs.append(self._extract_embedding(img))
            except: continue

        if not query_embs:
            return (None, None)

        return self._find_best_match(query_embs, threshold)

    def _find_best_match(self, query_embs: list, threshold: float):
        best_name: str | None = None
        best_sim: float = -1.0

        for person_name, stored_embs in self.embeddings.items():
            total = 0.0
            for q_emb in query_embs:
                sims = [_cosine_similarity(q_emb, s) for s in stored_embs]
                total += max(sims)
            avg_sim = total / len(query_embs)

            if avg_sim > best_sim:
                best_sim = avg_sim
                best_name = person_name

        if best_sim < threshold:
            return (None, float(best_sim))
        return (best_name, float(best_sim))

    def get_all_match_scores(self, face_image: np.ndarray) -> dict:
        if face_image is None or not self.embeddings:
            return {}
        
        try:
            query_emb = self._extract_embedding(face_image)
            scores = {}
            for person_name, stored_embs in self.embeddings.items():
                sims = [_cosine_similarity(query_emb, s) for s in stored_embs]
                scores[person_name] = float(max(sims)) if sims else 0.0
            return dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))
        except:
            return {}