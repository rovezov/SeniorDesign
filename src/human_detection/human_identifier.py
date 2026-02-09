

import cv2
import numpy as np
import onnxruntime as ort
import os
from scipy.spatial import distance as dist
from collections import OrderedDict, deque


class HumanTracker:
    """Tracks humans across frames and assigns persistent IDs"""
    def __init__(self, max_disappeared=30, persistence_window=10, min_tracking_time=2.0, max_centroid_distance=150):
        self.next_id = 0
        self.objects = OrderedDict()  # id -> centroid
        self.disappeared = OrderedDict()  # id -> count since last seen
        self.max_disappeared = max_disappeared
        self.persistence_window = persistence_window
        self.detection_history = OrderedDict()  # id -> deque of bool (detected in frame)
        self.first_seen_time = OrderedDict()  # id -> timestamp when first registered
        self.min_tracking_time = min_tracking_time  # Minimum seconds to track before considering "stable"
        self.max_centroid_distance = max_centroid_distance  # Max distance for centroid matching
        
    def register(self, centroid):
        """Register a new object with next available ID"""
        import time
        self.objects[self.next_id] = centroid
        self.disappeared[self.next_id] = 0
        self.detection_history[self.next_id] = deque(maxlen=self.persistence_window)
        self.detection_history[self.next_id].append(True)
        self.first_seen_time[self.next_id] = time.time()
        self.next_id += 1
        
    def deregister(self, object_id):
        """Remove an object that has disappeared"""
        del self.objects[object_id]
        del self.disappeared[object_id]
        del self.detection_history[object_id]
        if object_id in self.first_seen_time:
            del self.first_seen_time[object_id]
        
    def update(self, bboxes):
        """Update tracked objects with new detections"""
        # Convert bboxes to centroids
        input_centroids = np.zeros((len(bboxes), 2), dtype="int")
        for i, (x, y, w, h) in enumerate(bboxes):
            cx = int(x + w / 2)
            cy = int(y + h / 2)
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
                        self.deregister(object_id)
                
                # Register new objects for unused columns
                unused_cols = set(range(D.shape[1])) - used_cols
                for col in unused_cols:
                    self.register(input_centroids[col])
        
        return self.objects
    
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
        import time
        if object_id not in self.first_seen_time:
            return 0.0
        return time.time() - self.first_seen_time[object_id]


# Global YOLO session (initialized on first use)
_yolo_session = None
_model_path = None


def _get_yolo_session():
    """Get or initialize YOLO ONNX session"""
    global _yolo_session, _model_path
    if _yolo_session is None:
        # Find model path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        _model_path = os.path.join(current_dir, 'model', 'yolov8n.onnx')
        if not os.path.exists(_model_path):
            raise FileNotFoundError(f"YOLO model not found at {_model_path}")
        _yolo_session = ort.InferenceSession(_model_path, providers=['CPUExecutionProvider'])
    return _yolo_session


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    """Resize and pad image to meet new_shape while keeping aspect ratio"""
    shape = img.shape[:2]  # current shape [h, w]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[1], new_shape[1] / shape[0])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = new_shape[0] - new_unpad[0], new_shape[1] - new_unpad[1]
    dw /= 2
    dh /= 2
    img_resized = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img_padded = cv2.copyMakeBorder(img_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img_padded, r, (left, top)


def xywh2xyxy(xywh):
    """Convert box from center format to corner format"""
    x, y, w, h = xywh
    x1 = x - w / 2
    y1 = y - h / 2
    x2 = x + w / 2
    y2 = y + h / 2
    return [x1, y1, x2, y2]


def bbox_iou_np(box, other_boxes):
    """Calculate IoU between box and other_boxes"""
    x1 = np.maximum(box[0], other_boxes[:, 0])
    y1 = np.maximum(box[1], other_boxes[:, 1])
    x2 = np.minimum(box[2], other_boxes[:, 2])
    y2 = np.minimum(box[3], other_boxes[:, 3])
    inter_w = np.maximum(0.0, x2 - x1)
    inter_h = np.maximum(0.0, y2 - y1)
    inter = inter_w * inter_h
    area1 = (box[2] - box[0]) * (box[3] - box[1])
    area2 = (other_boxes[:, 2] - other_boxes[:, 0]) * (other_boxes[:, 3] - other_boxes[:, 1])
    union = area1 + area2 - inter
    iou = inter / np.maximum(union, 1e-8)
    return iou


def nms(boxes, scores, iou_threshold=0.45):
    """Non-maximum suppression"""
    idxs = np.argsort(-scores)
    keep = []
    while idxs.size > 0:
        i = idxs[0]
        keep.append(i)
        if idxs.size == 1:
            break
        ious = bbox_iou_np(boxes[i], boxes[idxs[1:]])
        idxs = idxs[1:][ious <= iou_threshold]
    return keep


def decode_yolo_output(pred, conf_thres=0.25, input_size=640):
    """Decode YOLO ONNX output to bounding boxes"""
    if pred.ndim == 3:
        pred = pred[0]
    
    # Handle transposed output
    if pred.ndim == 2 and pred.shape[0] <= 200 and pred.shape[1] > pred.shape[0]:
        pred = pred.T
    
    if pred.ndim != 2 or pred.shape[1] < 6:
        raise RuntimeError(f'Unexpected model output shape: {pred.shape}')
    
    xywh = pred[:, 0:4].astype(np.float32)
    num_classes = pred.shape[1] - 4
    
    # Handle models with or without objectness score
    if num_classes >= 20:
        # No objectness column - class scores start at index 4
        conf = np.ones((pred.shape[0],), dtype=np.float32)
        class_scores = pred[:, 4:]
    else:
        # Has objectness column
        conf = pred[:, 4].astype(np.float32)
        class_scores = pred[:, 5:]
    
    class_ids = np.argmax(class_scores, axis=1)
    class_conf = class_scores[np.arange(class_scores.shape[0]), class_ids]
    scores = conf * class_conf
    
    mask = scores > conf_thres
    if not np.any(mask):
        return []
    
    xywh = xywh[mask]
    scores = scores[mask]
    class_ids = class_ids[mask]
    
    # Scale coordinates if normalized
    if np.max(xywh) <= 1.01:
        xywh = xywh * float(input_size)
    
    boxes = []
    for x, s, cid in zip(xywh, scores, class_ids):
        box = xywh2xyxy(x.tolist())
        boxes.append((box, float(s), int(cid)))
    return boxes


def detect_humans(image, edge_margin=20, debug=False):
    """Detect humans in an image and return bounding boxes (x, y, w, h) using YOLO ONNX.
    
    Args:
        image: Input image
        edge_margin: Pixels from edge to filter out partial detections (0 to disable)
        debug: Print debug information
    
    Returns:
        List of bounding boxes (x, y, w, h) for detected persons fully in frame
    """
    sess = _get_yolo_session()
    input_name = sess.get_inputs()[0].name
    input_size = 640
    
    orig_h, orig_w = image.shape[:2]
    
    # Preprocess image
    img, ratio, (dw, dh) = letterbox(image, new_shape=(input_size, input_size))
    img = img[:, :, ::-1].astype(np.float32)  # BGR to RGB
    img = img / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, 0).astype(np.float32)
    
    # Run inference
    outputs = sess.run(None, {input_name: img})
    
    # Get prediction output
    pred = None
    if isinstance(outputs, list) and len(outputs) == 1:
        pred = outputs[0]
    else:
        for o in outputs:
            if isinstance(o, np.ndarray) and o.ndim == 3:
                pred = o
                break
    
    if pred is None:
        if debug:
            print("ERROR: No compatible prediction output from model")
        return []
    
    pred = np.asarray(pred)
    if debug:
        print(f"Raw prediction shape: {pred.shape}, min: {pred.min():.3f}, max: {pred.max():.3f}")
    
    # Lower confidence threshold to catch more detections
    detections = decode_yolo_output(pred, conf_thres=0.2, input_size=input_size)
    
    if debug:
        print(f"Total detections (all classes): {len(detections)}")
    
    # Filter for person class (COCO class 0) and convert to original image coordinates
    person_boxes = []
    for box, score, cid in detections:
        if cid == 0:  # Person class
            x1, y1, x2, y2 = box
            # Convert from letterbox coordinates to original image coordinates
            x1 = (x1 - dw) / ratio
            x2 = (x2 - dw) / ratio
            y1 = (y1 - dh) / ratio
            y2 = (y2 - dh) / ratio
            x1 = int(max(0, min(orig_w - 1, x1)))
            x2 = int(max(0, min(orig_w - 1, x2)))
            y1 = int(max(0, min(orig_h - 1, y1)))
            y2 = int(max(0, min(orig_h - 1, y2)))
            
            if x2 > x1 and y2 > y1:
                # Convert to (x, y, w, h) format
                w = x2 - x1
                h = y2 - y1
                person_boxes.append(([x1, y1, x2, y2], score))
    
    if debug:
        print(f"Person detections (class 0): {len(person_boxes)}")
    
    # Apply NMS
    if len(person_boxes) > 0:
        boxes_arr = np.array([p[0] for p in person_boxes])
        scores_arr = np.array([p[1] for p in person_boxes], dtype=float)
        keep = nms(boxes_arr, scores_arr, iou_threshold=0.45)
        final_boxes = boxes_arr[keep]
        
        if debug:
            print(f"After NMS: {len(final_boxes)} persons")
        
        # Convert to (x, y, w, h) format and filter edge detections
        result = []
        for box in final_boxes:
            x1, y1, x2, y2 = box
            x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
            
            # Filter out bboxes at the edge if margin is set
            if edge_margin > 0:
                if (x > edge_margin and 
                    y > edge_margin and 
                    x + w < orig_w - edge_margin and 
                    y + h < orig_h - edge_margin):
                    result.append((x, y, w, h))
                elif debug:
                    print(f"Filtered edge detection: bbox({x},{y},{w},{h}) margin={edge_margin} frame({orig_w}x{orig_h})")
            else:
                result.append((x, y, w, h))
        
        if debug:
            print(f"Final results after edge filter: {len(result)} persons")
        return result
    
    if debug:
        print("No person detections found")
    return []


def crop_person(image, bbox):
    """Crop person from image given bounding box (x, y, w, h)"""
    x, y, w, h = bbox
    x1, y1 = int(x), int(y)
    x2, y2 = int(x + w), int(y + h)
    
    # Ensure bounds are within image
    h_img, w_img = image.shape[:2]
    x1 = max(0, min(w_img - 1, x1))
    x2 = max(0, min(w_img, x2))
    y1 = max(0, min(h_img - 1, y1))
    y2 = max(0, min(h_img, y2))
    
    if x2 <= x1 or y2 <= y1:
        return None
    
    return image[y1:y2, x1:x2]


def identify_humans(frame, edge_margin=0, debug=False):
    """Detect humans and return bounding boxes"""
    return detect_humans(frame, edge_margin=edge_margin, debug=debug)

