Face Matching Module
=====================

## Algorithm

**ArcFace (MobileFaceNet backbone)** — deep-learning face embeddings via
[insightface](https://github.com/deepinsight/insightface) `buffalo_sc` pack.

| Property | Value |
|---|---|
| Backbone | MobileFaceNet |
| Training set | WebFace600K |
| Embedding dim | 512-d, L2-normalised |
| Input size | 112 × 112 px |
| ONNX model size | ~1.5 MB |
| Similarity metric | Cosine similarity ∈ [0, 1] |
| Default threshold | 0.35 (raise for stricter matching) |

This replaces the previous LBPH recogniser. ArcFace is significantly more
accurate across pose, lighting, and age variation, and runs efficiently via
ONNX Runtime on edge hardware (tested on Qualcomm QCS6490 / RubikPi 3).

**⚠ Confidence semantics changed**: confidence is now cosine similarity
(higher = better), **not** LBPH distance (lower = better).

---

## Installation

```bash
pip install insightface onnxruntime
# insightface auto-downloads buffalo_sc (~1.5 MB) on first run
```

No ONNX file needs to be placed manually — insightface handles model
download and caching automatically.

---

## Module API (minimal)

```python
from face_matching.face_matcher import FaceMatcher

m = FaceMatcher()  # loads/rebuilds embeddings automatically
```

### `match_face_image(face_image, threshold=0.35)`

- **Input**: single pre-cropped face as a NumPy BGR array.
- **Output**: `(name, confidence)`
  - `name` — folder name of the matched person, or `None`.
  - `confidence` — cosine similarity ∈ [0, 1]; higher is better.
    Returns `None` when similarity is below `threshold`.

### `match_image_group(images, threshold=0.35)`

- **Input**: list of 1–3 pre-cropped face images (NumPy BGR arrays).
- **Output**: `(name, confidence)` — consensus result across the group.

---

## Training data layout

Place per-person image folders inside `known_faces/` (or `trained_faces/`):

```
known_faces/
    Alice/
        photo1.jpg
        photo2.jpg
    Bob/
        photo1.jpg
```

Rebuild the embedding cache after adding new people:

```python
m = FaceMatcher()
m.train_and_save_pkl()   # writes embeddings.pkl next to this file
```

---

## Examples

```python
import cv2
from face_matching.face_matcher import FaceMatcher

m = FaceMatcher()

# Single image
img = cv2.imread("test_faces/Lebron_James/lebron-james-10.webp")
name, conf = m.match_face_image(img)
print(name, f"{conf:.3f}")        # e.g. "Lebron_James 0.612"

# Group vote (up to 3 crops of the same person → consensus)
faces = [cv2.imread("img1.jpg"), cv2.imread("img2.jpg")]
name, conf = m.match_image_group(faces)
if name:
    print("Recognised:", name, "similarity:", round(conf, 3))
else:
    print("Not recognised (similarity too low:", round(conf, 3), ")")
```

---

## Tuning

- **Threshold 0.30** — lenient, more matches, possible false positives.
- **Threshold 0.35** — default, good balance.
- **Threshold 0.45** — strict, fewer false positives, may miss marginal faces.

Provide ≥ 5 varied photos per person (different lighting / slight pose variation)
for best embedding coverage.

---

## Fallback ONNX model

If `insightface` is not available, place any ArcFace-compatible 112x112 -> 512-d
ONNX model at:

```
src/face_matching/arcface.onnx
```

The module will load it automatically via `onnxruntime`.