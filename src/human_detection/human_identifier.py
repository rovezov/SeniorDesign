"""
Human Identification Service

Provides real-time person detection and tracking using YOLO.
Maintains persistent IDs across frames and outputs cropped person images.
"""

import cv2
import numpy as np
import onnxruntime as ort
import os
import time
from scipy.spatial import distance as dist
from collections import OrderedDict, deque


class HumanIdentificationService:
    """
    Service for detecting, tracking, and identifying humans in video frames.
    Maintains state across frames and provides high-quality cropped person images
    optimized for PPE detection.
    
    Features:
    - YOLO-based person detection
    - Centroid tracking with persistent IDs
    - Quality-based crop selection (chooses best frame for PPE visibility)
    - Automatic filtering of edge detections
    """
    
    def __init__(self, model_path=None, edge_margin=0, min_tracking_time=2.0, 
                 max_centroid_distance=150, conf_threshold=0.15):
        """
        Initialize the human identification service.
        
        Args:
            model_path: Path to YOLO ONNX model (auto-detected if None)
            edge_margin: Pixels from edge to filter detections (0 to disable)
            min_tracking_time: Minimum seconds to track before saving
            max_centroid_distance: Max pixel distance for tracking
            conf_threshold: Confidence threshold for YOLO detections
        """
        self.edge_margin = edge_margin
        self.conf_threshold = conf_threshold
        
        # Initialize YOLO session
        if model_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(current_dir, 'model', 'yolov8n.onnx')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"YOLO model not found at {model_path}")
        
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.input_size = 640
        
        self.tracker = HumanTracker(
            max_disappeared=10,
            persistence_window=10,
            min_tracking_time=min_tracking_time,
            max_centroid_distance=max_centroid_distance
        )
        self.saved_ids = set()
    
    def process_frame(self, frame):
        """Process frame and return detected/tracked persons with quality-based crop selection"""
        bboxes = self._detect_humans(frame)
        tracked_objects, deregistered_info = self.tracker.update(bboxes)
        frame_h, frame_w = frame.shape[:2]
        results = []
        
        # Handle departed persons
        for departed_id, info in deregistered_info.items():
            if departed_id not in self.saved_ids and info['crop'] is not None:
                if info['quality'] > 0.3 and info['duration'] >= self.tracker.min_tracking_time:
                    results.append({
                        'id': departed_id,
                        'centroid': (0, 0),
                        'bbox': None,
                        'crop': info['crop'],
                        'ready_to_save': True,
                        'tracking_duration': info['duration'],
                        'is_new': True,
                        'departed': True
                    })
                    self.saved_ids.add(departed_id)
        
        # Handle currently tracked persons
        for object_id, centroid in tracked_objects.items():
            result = {
                'id': object_id,
                'centroid': tuple(centroid),
                'bbox': None,
                'crop': None,
                'ready_to_save': self.tracker.is_ready_to_save(object_id),
                'tracking_duration': self.tracker.get_tracking_duration(object_id),
                'is_new': False,
                'departed': False
            }
            
            for (x, y, w, h) in bboxes:
                cx, cy = x + w // 2, y + h // 2
                if abs(cx - centroid[0]) < 20 and abs(cy - centroid[1]) < 20:
                    result['bbox'] = (x, y, w, h)
                    crop = self._crop_person(frame, (x, y, w, h))
                    if crop is not None and crop.size > 0:
                        self.tracker.update_best_crop(object_id, crop, (x, y, w, h), frame_w, frame_h)
                    break
            
            results.append(result)
        
        return results
    
    def _detect_humans(self, image, debug=False):
        """Detect humans in image and return bounding boxes (x, y, w, h)"""
        orig_h, orig_w = image.shape[:2]
        
        # Preprocess
        img, ratio, (dw, dh) = self._letterbox(image)
        img = img[:, :, ::-1].astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))[None].astype(np.float32)
        
        # Inference
        outputs = self.session.run(None, {self.input_name: img})
        pred = outputs[0] if isinstance(outputs, list) and len(outputs) == 1 else None
        if pred is None:
            for o in outputs:
                if isinstance(o, np.ndarray) and o.ndim == 3:
                    pred = o
                    break
        if pred is None:
            return []
        
        detections = self._decode_yolo_output(pred)
        person_boxes = []
        for box, score, cid in detections:
            if cid == 0:
                x1, y1, x2, y2 = box
                x1 = int(max(0, min(orig_w - 1, (x1 - dw) / ratio)))
                x2 = int(max(0, min(orig_w - 1, (x2 - dw) / ratio)))
                y1 = int(max(0, min(orig_h - 1, (y1 - dh) / ratio)))
                y2 = int(max(0, min(orig_h - 1, (y2 - dh) / ratio)))
                if x2 > x1 and y2 > y1:
                    person_boxes.append(([x1, y1, x2, y2], score))
        
        if not person_boxes:
            return []
        
        boxes_arr = np.array([p[0] for p in person_boxes])
        scores_arr = np.array([p[1] for p in person_boxes])
        keep = self._nms(boxes_arr, scores_arr)
        
        result = []
        for box in boxes_arr[keep]:
            x1, y1, x2, y2 = box
            x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
            if self.edge_margin > 0:
                if (x < self.edge_margin or y < self.edge_margin or
                    x + w > orig_w - self.edge_margin or y + h > orig_h - self.edge_margin):
                    continue
            result.append((x, y, w, h))
        
        return result
    
    def _letterbox(self, img):
        """Resize and pad image"""
        shape = img.shape[:2]
        new_shape = (self.input_size, self.input_size)
        r = min(new_shape[0] / shape[1], new_shape[1] / shape[0])
        new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
        dw = (new_shape[0] - new_unpad[0]) / 2
        dh = (new_shape[1] - new_unpad[1]) / 2
        img_resized = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img_padded = cv2.copyMakeBorder(img_resized, top, bottom, left, right, 
                                        cv2.BORDER_CONSTANT, value=(114, 114, 114))
        return img_padded, r, (left, top)
    
    def _decode_yolo_output(self, pred):
        """Decode YOLO output to detections"""
        if pred.ndim == 3:
            pred = pred[0]
        if pred.ndim == 2 and pred.shape[0] <= 200 and pred.shape[1] > pred.shape[0]:
            pred = pred.T
        
        xywh = pred[:, 0:4].astype(np.float32)
        num_classes = pred.shape[1] - 4
        
        if num_classes >= 20:
            conf = np.ones((pred.shape[0],), dtype=np.float32)
            class_scores = pred[:, 4:]
        else:
            conf = pred[:, 4].astype(np.float32)
            class_scores = pred[:, 5:]
        
        class_ids = np.argmax(class_scores, axis=1)
        class_conf = class_scores[np.arange(class_scores.shape[0]), class_ids]
        scores = conf * class_conf
        
        mask = scores > self.conf_threshold
        if not np.any(mask):
            return []
        
        xywh = xywh[mask]
        scores = scores[mask]
        class_ids = class_ids[mask]
        
        if np.max(xywh) <= 1.01:
            xywh = xywh * float(self.input_size)
        
        boxes = []
        for x, s, cid in zip(xywh, scores, class_ids):
            x, y, w, h = x.tolist()
            box = [x - w/2, y - h/2, x + w/2, y + h/2]
            boxes.append((box, float(s), int(cid)))
        return boxes
    
    def _nms(self, boxes, scores, iou_threshold=0.45):
        """Non-maximum suppression"""
        idxs = np.argsort(-scores)
        keep = []
        while idxs.size > 0:
            i = idxs[0]
            keep.append(i)
            if idxs.size == 1:
                break
            ious = self._bbox_iou(boxes[i], boxes[idxs[1:]])
            idxs = idxs[1:][ious <= iou_threshold]
        return keep
    
    def _bbox_iou(self, box, other_boxes):
        """Calculate IoU"""
        x1 = np.maximum(box[0], other_boxes[:, 0])
        y1 = np.maximum(box[1], other_boxes[:, 1])
        x2 = np.minimum(box[2], other_boxes[:, 2])
        y2 = np.minimum(box[3], other_boxes[:, 3])
        inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
        area1 = (box[2] - box[0]) * (box[3] - box[1])
        area2 = (other_boxes[:, 2] - other_boxes[:, 0]) * (other_boxes[:, 3] - other_boxes[:, 1])
        return inter / np.maximum(area1 + area2 - inter, 1e-8)
    
    def _crop_person(self, image, bbox):
        """Crop person from image"""
        x, y, w, h = bbox
        x1, y1 = int(x), int(y)
        x2, y2 = int(x + w), int(y + h)
        
        h_img, w_img = image.shape[:2]
        x1 = max(0, min(w_img - 1, x1))
        x2 = max(0, min(w_img, x2))
        y1 = max(0, min(h_img - 1, y1))
        y2 = max(0, min(h_img, y2))
        
        if x2 <= x1 or y2 <= y1:
            return None
        
        return image[y1:y2, x1:x2]
    
    def get_all_tracked_crops(self):
        """Get best crops for all currently tracked persons (for saving on exit)"""
        crops_to_save = []
        for object_id in list(self.tracker.objects.keys()):
            if object_id not in self.saved_ids:
                best_crop = self.tracker.get_best_crop(object_id)
                duration = self.tracker.get_tracking_duration(object_id)
                if best_crop is not None and duration >= self.tracker.min_tracking_time:
                    quality = self.tracker.best_crops[object_id]['quality']
                    if quality > 0.3:
                        crops_to_save.append({
                            'id': object_id,
                            'crop': best_crop,
                            'quality': quality,
                            'duration': duration
                        })
        return crops_to_save
    
    def reset(self):
        """Reset tracking state"""
        self.tracker = HumanTracker(
            max_disappeared=10,
            persistence_window=10,
            min_tracking_time=self.tracker.min_tracking_time,
            max_centroid_distance=self.tracker.max_centroid_distance
        )
        self.saved_ids = set()


class HumanTracker:
    """Tracks humans across frames and assigns persistent IDs"""
    
    def __init__(self, max_disappeared=10, persistence_window=10, min_tracking_time=2.0, max_centroid_distance=150):
        self.next_id = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.max_disappeared = max_disappeared
        self.persistence_window = persistence_window
        self.detection_history = OrderedDict()
        self.first_seen_time = OrderedDict()
        self.min_tracking_time = min_tracking_time
        self.max_centroid_distance = max_centroid_distance
        self.best_crops = OrderedDict()
        
    def register(self, centroid):
        """Register a new object with next available ID"""
        import time
        self.objects[self.next_id] = centroid
        self.disappeared[self.next_id] = 0
        self.detection_history[self.next_id] = deque(maxlen=self.persistence_window)
        self.detection_history[self.next_id].append(True)
        self.first_seen_time[self.next_id] = time.time()
        self.best_crops[self.next_id] = {'crop': None, 'quality': 0.0, 'bbox': None}
        self.next_id += 1
        
    def deregister(self, object_id):
        """Remove an object that has disappeared"""
        del self.objects[object_id]
        del self.disappeared[object_id]
        del self.detection_history[object_id]
        if object_id in self.first_seen_time:
            del self.first_seen_time[object_id]
        if object_id in self.best_crops:
            del self.best_crops[object_id]
        
    def update(self, bboxes):
        """Update tracked objects with new detections and return (tracked_objects, deregistered_info)"""
        deregistered_info = {}
        input_centroids = np.zeros((len(bboxes), 2), dtype="int")
        for i, (x, y, w, h) in enumerate(bboxes):
            cx, cy = int(x + w / 2), int(y + h / 2)
            input_centroids[i] = (cx, cy)
        
        # If no objects being tracked, register all
        if len(self.objects) == 0:
            for centroid in input_centroids:
                self.register(centroid)
        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())
            
            # If no new detections, mark all as disappeared
            if len(input_centroids) == 0:
                for object_id in object_ids:
                    self.disappeared[object_id] += 1
                    self.detection_history[object_id].append(False)
                    if self.disappeared[object_id] > self.max_disappeared:
                        # Save best crop info before deregistering
                        if object_id in self.best_crops:
                            deregistered_info[object_id] = {
                                'crop': self.best_crops[object_id]['crop'],
                                'quality': self.best_crops[object_id]['quality'],
                                'duration': self.get_tracking_duration(object_id)
                            }
                        self.deregister(object_id)
            else:
                # Compute distance between each pair of object centroids and input centroids
                D = dist.cdist(np.array(object_centroids), input_centroids)
                
                # Find the smallest value in each row and sort by these values
                rows = D.min(axis=1).argsort()
                cols = D.argmin(axis=1)[rows]
                
                used_rows = set()
                used_cols = set()
                
                # Match existing objects to new centroids
                for (row, col) in zip(rows, cols):
                    if row in used_rows or col in used_cols:
                        continue
                    
                    # Check if distance is reasonable (not too far)
                    if D[row, col] < self.max_centroid_distance:
                        object_id = object_ids[row]
                        self.objects[object_id] = input_centroids[col]
                        self.disappeared[object_id] = 0
                        self.detection_history[object_id].append(True)
                        used_rows.add(row)
                        used_cols.add(col)
                
                # Mark unused rows as disappeared
                unused_rows = set(range(D.shape[0])) - used_rows
                for row in unused_rows:
                    object_id = object_ids[row]
                    self.disappeared[object_id] += 1
                    self.detection_history[object_id].append(False)
                    if self.disappeared[object_id] > self.max_disappeared:
                        # Save best crop info before deregistering
                        if object_id in self.best_crops:
                            deregistered_info[object_id] = {
                                'crop': self.best_crops[object_id]['crop'],
                                'quality': self.best_crops[object_id]['quality'],
                                'duration': self.get_tracking_duration(object_id)
                            }
                        self.deregister(object_id)
                
                # Register new objects for unused columns
                unused_cols = set(range(D.shape[1])) - used_cols
                for col in unused_cols:
                    self.register(input_centroids[col])
        
        return self.objects, deregistered_info
    
    def is_persistent(self, object_id, min_ratio=0.9):
        """Check if an object has been detected in at least min_ratio of recent frames"""
        if object_id not in self.detection_history:
            return False
        history = self.detection_history[object_id]
        if len(history) == 0:
            return False
        detected_count = sum(history)
        ratio = detected_count / len(history)
        return ratio >= min_ratio
    
    def is_ready_to_save(self, object_id, min_ratio=0.7):
        """Check if object is stable enough to save (persistent + tracked for min time)"""
        import time
        if not self.is_persistent(object_id, min_ratio):
            return False
        if object_id not in self.first_seen_time:
            return False
        tracking_duration = time.time() - self.first_seen_time[object_id]
        return tracking_duration >= self.min_tracking_time
    
    def get_tracking_duration(self, object_id):
        """Get how long an object has been tracked in seconds"""
        if object_id not in self.first_seen_time:
            return 0.0
        return time.time() - self.first_seen_time[object_id]
    
    def calculate_quality_score(self, bbox, frame_width, frame_height):
        """Calculate quality score for PPE detection (0.0 to 1.0)
        
        Prioritizes:
        - Large bbox (close to camera, more detail for PPE)
        - Fully in frame (need full body for boots, vests, gloves)
        - Good aspect ratio (standing person pose)
        - Centered in frame (better lighting, less distortion)
        """
        x, y, w, h = bbox
        
        # 1. Size score (40%) - larger is better for PPE detail
        bbox_area = w * h
        frame_area = frame_width * frame_height
        size_ratio = bbox_area / frame_area
        # Normalize: 0.02 (2% of frame) = 0.0, 0.20 (20% of frame) = 1.0
        size_score = min(1.0, max(0.0, (size_ratio - 0.02) / 0.18))
        
        # 2. Fully in frame score (20%) - critical for full body PPE check
        margin = 10
        fully_in_frame = (
            x >= margin and y >= margin and
            x + w <= frame_width - margin and y + h <= frame_height - margin
        )
        in_frame_score = 1.0 if fully_in_frame else 0.0
        
        # 3. Aspect ratio score (20%) - standing person is ~1.5-2.5 H/W
        aspect_ratio = h / w if w > 0 else 0
        # Ideal range: 1.5 to 2.5, with peak at 2.0
        if 1.5 <= aspect_ratio <= 2.5:
            aspect_score = 1.0 - abs(aspect_ratio - 2.0) / 1.0
        else:
            aspect_score = max(0.0, 1.0 - abs(aspect_ratio - 2.0) / 2.0)
        
        # 4. Centering score (20%) - center of frame preferred
        cx = x + w / 2
        cy = y + h / 2
        center_x = frame_width / 2
        center_y = frame_height / 2
        dx = abs(cx - center_x) / (frame_width / 2)
        dy = abs(cy - center_y) / (frame_height / 2)
        centering_score = 1.0 - (dx * 0.6 + dy * 0.4)
        centering_score = max(0.0, centering_score)
        
        # Weighted combination
        total_score = (
            size_score * 0.2 +
            in_frame_score * 0.4 +
            aspect_score * 0.2 +
            centering_score * 0.2
        )
        
        return total_score
    
    def update_best_crop(self, object_id, crop, bbox, frame_width, frame_height):
        """Update best crop if current one is higher quality"""
        if object_id not in self.best_crops:
            return
        
        quality = self.calculate_quality_score(bbox, frame_width, frame_height)
        
        if quality > self.best_crops[object_id]['quality']:
            self.best_crops[object_id] = {
                'crop': crop.copy() if crop is not None else None,
                'quality': quality,
                'bbox': bbox
            }
    
    def get_best_crop(self, object_id):
        """Get the best quality crop for an object"""
        if object_id not in self.best_crops:
            return None
        return self.best_crops[object_id]['crop']



