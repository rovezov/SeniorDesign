# PPE Detection Module
# Identifies PPE items and violations

import os
import cv2
import numpy as np
import onnxruntime as ort
from utils.verbose import log



class PPEDetector:
    """
    Detects personal protective equipment (PPE) in images using a pre-trained ONNX model.

    The model recognises two classes:
        0 -> helmet
        1 -> vest

    Attributes:
        requirements (set[str]): PPE items that must be present.
        conf_threshold (float): Minimum confidence score to accept a detection.
    """

    # Model input dimensions expected by gear_guard_net.onnx  (NCHW: H=320, W=192)
    _INPUT_W = 192
    _INPUT_H = 320

    _VALID_LABELS = {"vest", "helmet"}

    def __init__(self, requirements: list[str], model_path: str | None = None,
                 conf_threshold: float = 0.3, nms_iou_threshold: float = 0.45):
        """
        Initialise the PPE detector.

        Args:
            requirements: List of PPE item names that are required (e.g. ['helmet', 'vest']).
                          Names are normalised to lower-case.
            model_path:       Path to the ONNX model file.  Defaults to the bundled model.
            conf_threshold:   Confidence threshold for filtering detections (0-1).
            nms_iou_threshold: IoU threshold used during Non-Maximum Suppression (0-1).
        """
        self.requirements: set[str] = self._validate_requirements(requirements)
        self.conf_threshold = conf_threshold
        self.nms_iou_threshold = nms_iou_threshold

        if model_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(current_dir, "model", "gear_guard_net.onnx")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"PPE model not found at: {model_path}")

        # Load label map [true labels found in labels.txt]
        self._labels = list(self._VALID_LABELS)

        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        self._session = ort.InferenceSession(model_path, sess_options=so, providers=["CPUExecutionProvider"])
        self._input_name: str = self._session.get_inputs()[0].name

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, image: np.ndarray) -> dict:
        """
        Run PPE detection on a single person crop and return a structured result.

        Args:
            image: BGR image as a NumPy array (e.g. from cv2.imread or a cropped frame).

        Returns:
            A dict with the following keys:
                detected  (list[dict]): Each entry has 'label' (str), 'confidence' (float),
                                        and 'bbox' ([x, y, w, h] in original image pixels).
                missing   (list[str]):  PPE items from requirements that were not detected.
                all_present (bool):     True when no required items are missing.
        """
        boxes_raw, scores_raw, class_indices_raw, scale, (pad_left, pad_top) = self._run_inference(image)

        # Bucket candidates by class so NMS is applied independently per class.
        # Store boxes as [x, y, w, h] in model pixel space for cv2.dnn.NMSBoxes.
        # Model outputs (x1, y1, x2, y2) corner format in letterboxed model pixel coordinates.
        class_candidates: dict[int, dict] = {}

        for i in range(len(scores_raw)):
            if scores_raw[i] < self.conf_threshold:
                log("ppe_detection", f"Skipping low-confidence detection: {self._labels[int(class_indices_raw[i])]} with score {scores_raw[i]:.2f}")
                continue

            class_id = int(class_indices_raw[i])
            if class_id >= len(self._labels):
                continue

            x1, y1, x2, y2 = boxes_raw[i]
            if class_id not in class_candidates:
                class_candidates[class_id] = {"boxes": [], "scores": []}
            class_candidates[class_id]["boxes"].append(
                [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
            )
            class_candidates[class_id]["scores"].append(float(scores_raw[i]))

        detected: list[dict] = []
        detected_labels: set[str] = set()

        for class_id, data in class_candidates.items():
            nms_input = [[int(b[0]), int(b[1]), int(b[2]), int(b[3])] for b in data["boxes"]]
            indices = cv2.dnn.NMSBoxes(nms_input, data["scores"], 0.0, self.nms_iou_threshold)

            if len(indices) == 0:
                continue

            label = self._labels[class_id]
            for idx in indices.flatten():
                x_m, y_m, w_m, h_m = data["boxes"][idx]
                # Invert letterbox transform: remove padding then undo scale
                x = int((x_m - pad_left) / scale)
                y = int((y_m - pad_top)  / scale)
                w = int(w_m / scale)
                h = int(h_m / scale)
                detected.append({
                    "label": label,
                    "confidence": data["scores"][idx],
                    "bbox": [x, y, w, h],
                })
                detected_labels.add(label)

        missing = sorted(self.requirements - detected_labels)

        return {
            "detected": detected,
            "missing": missing,
            "all_present": len(missing) == 0,
        }

    def update_requirements(self, requirements: list[str]) -> None:
        """
        Replace the current PPE requirements.

        Args:
            requirements: New list of required PPE item names.
        """
        self.requirements = self._validate_requirements(requirements)

    @classmethod
    def _validate_requirements(cls, requirements: list[str]) -> set[str]:
        normalised = {item.lower() for item in requirements}
        invalid = normalised - cls._VALID_LABELS
        if invalid:
            raise ValueError(
                f"Invalid PPE requirement(s): {sorted(invalid)}. "
                f"Allowed values: {sorted(cls._VALID_LABELS)}"
            )
        return normalised

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _preprocess(self, image: np.ndarray) -> tuple[np.ndarray, float, tuple[int, int]]:
        """
        Resize and normalise the image into the model's expected input tensor.

        Scales so height exactly fills INPUT_H (preserving aspect ratio), which keeps the
        subject at full height in the canvas — critical for a person-oriented PPE model.
        Width is center-cropped if the scaled image is too wide, or padded with grey if
        too narrow.  The returned scale and pad_left let detect() map boxes back precisely
        to original image coordinates.

        Returns:
            tensor:   float32 NCHW array ready for inference.
            scale:    uniform scale factor applied to the original image.
            pad:      (pad_left, 0) — horizontal offset of the image within the canvas.
                      pad_left is negative when a center-crop was applied.
        """
        orig_h, orig_w = image.shape[:2]
        scale  = self._INPUT_H / orig_h          # fill canvas height exactly
        new_w  = int(orig_w * scale)
        new_h  = self._INPUT_H

        resized = cv2.resize(image, (new_w, new_h))
        canvas  = np.full((self._INPUT_H, self._INPUT_W, 3), 114, dtype=np.uint8)

        if new_w >= self._INPUT_W:
            # Wider than canvas — take the centre INPUT_W columns
            x_start  = (new_w - self._INPUT_W) // 2
            canvas[:] = resized[:, x_start:x_start + self._INPUT_W]
            pad_left  = -x_start          # negative so inverse formula stays uniform
        else:
            # Narrower than canvas — centre-pad with grey
            pad_left = (self._INPUT_W - new_w) // 2
            canvas[:, pad_left:pad_left + new_w] = resized

        rgb    = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        tensor = rgb.astype(np.float32) / 255.0
        tensor = np.transpose(tensor, (2, 0, 1))
        tensor = np.expand_dims(tensor, axis=0)
        return tensor, scale, (pad_left, 0)   # pad_top is always 0

    def _run_inference(self, image: np.ndarray):
        """Run the ONNX session and return (boxes, scores, class_indices, scale, pad)."""
        tensor, scale, pad = self._preprocess(image)
        outputs = self._session.run(None, {self._input_name: tensor})

        # outputs: [boxes (1,3780,4), scores (1,3780), class_idx (1,3780)]
        boxes        = outputs[0][0]  # (3780, 4) — (x1, y1, x2, y2) in model pixel coords
        scores       = outputs[1][0]  # (3780,)
        class_indices = outputs[2][0] # (3780,)

        return boxes, scores, class_indices, scale, pad

