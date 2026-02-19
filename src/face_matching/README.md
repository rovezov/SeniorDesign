Face matching module
=====================
Module API (minimal)
----------
Import the module and call the minimal functions from Python code. Functions accept in-memory cropped face images (NumPy BGR arrays) and return a simple tuple `(name, confidence)`.

- `match_face_image(face_image)`
  - Input: single cropped face image as a NumPy array (BGR). Do not pass a file path.
  - Output: `(name, confidence)` where `name` is the recognized person's folder name or `None` when not recognized. `confidence` is the LBPH score (lower = more similar).

- `match_image_group(images)`
  - Input: list of 1–3 cropped face images (NumPy arrays, BGR).
  - Output: `(name, confidence)` with the aggregated decision for the group. `name` is `None` when no match.

Examples (programmatic)
-----------------------
```python
import cv2
from src.face_matching import recognize_video as rv

# single image
img = cv2.imread('test_faces/Lebron_James/lebron-james-10.webp')
name, conf = rv.match_face_image(img)
print(name, conf)

# group of multiple cropped faces
faces = [cv2.imread('img1.jpg'), cv2.imread('img2.jpg')]
name, conf = rv.match_image_group(faces)
if name:
    print('Recognized:', name, 'confidence:', conf)
else:
    print('Not recognized')
```
- OpenCV with contrib (for `cv2.face.LBPHFaceRecognizer_create`) — install via `pip install opencv-contrib-python`
- NumPy
- Optional: `cvzone` for improved face detection (`pip install cvzone`) — if missing the module will still work but expects cropped face images.

Quick install (example):

```bash
pip install opencv-contrib-python numpy
# optional detector
pip install cvzone
```

Module API
----------
Import the module or call functions directly from Python code:

- `match_face_image(face_image_path, threshold=100)`
  - Input: path to a single cropped face image
  - Output: `dict` {"label_id": int, "name": str, "confidence": float, "match": bool}
  - `confidence`: LBPH score (lower is better). `match` is True when `confidence < threshold`.

- `match_image_group(image_paths, threshold=100)`
  - Input: list of 1–3 image paths representing different photos of the same test person
  - Output: `dict` {
      "per_image": [ ... per-image dicts ... ],
      "overall": {"label_id": ..., "name": ..., "confidence": ..., "match": bool}
    }
  - Aggregation rule: majority wins (label with most matching images); ties broken by lowest average confidence. Overall `match` is True if at least one image matched the winning label.

Examples (programmatic)
-----------------------
```python
from recognize_video import match_face_image, match_image_group

# single image
r = match_face_image('test_faces/Lebron_James/lebron-james-10.webp', threshold=100)
print(r)

# group of multiple photos (up to 3)
res = match_image_group(['img1.jpg', 'img2.jpg'], threshold=100)
print(res['per_image'])
print(res['overall'])
```

Command line / CLI
------------------
The script supports three modes:

1. Directory mode (process all images under a folder):

```bash
python recognize_video.py test_faces --threshold 100
```

2. Single image mode:

```bash
python recognize_video.py path/to/image.jpg --threshold 100
```

3. Group mode (1–3 image paths for the same test person):

```bash
python recognize_video.py img1.jpg img2.jpg --threshold 100
```

CLI notes:
- `--threshold` controls how permissive the match is. Lower values are stricter. Default is `100`.
- Output: The script prints a Python-style `dict` for each image and, in grouped mode, an `Overall` summary. It also prints a summary in directory mode: `Processed N images, matches: M`.

Understanding results
---------------------
- `label_id`: numeric id assigned during training (0..N-1).
- `name`: folder name mapped to the `label_id`.
- `confidence`: LBPH distance (lower = more similar). 0 is perfect match.
- `match`: boolean (`True` if `confidence < threshold`). Tune `--threshold` to reduce false positives.

Exit codes
----------
- `0` — ran and printed results successfully.
- `1` — usage error (incorrect args).
- `2` — runtime error (e.g., missing training data, missing image file).

Tuning & recommendations
------------------------
- Provide multiple training images per person with varied lighting/pose for better robustness.
- If using raw photos (not cropped faces), install `cvzone` so the module can detect and crop faces automatically.
- Experiment with `--threshold` (try 50, 70, 100) and check false positive/negative tradeoffs.

Extending
---------
- The script can be adapted to save/load trained recognizer to disk instead of training at import time.
- You can change aggregation rules in `match_image_group` (e.g., require unanimous matches or use averaged confidence).

File locations
--------------
- Core script: `recognize_video.py`
- Place this README next to that file: `src/face_matching/README.md`

Contact
-------
If something doesn't work (missing `cv2.face` or missing training images), inspect printed `Error:` messages and ensure `trained_faces` or `known_faces` is present with subfolders per person.
