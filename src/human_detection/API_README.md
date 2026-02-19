# Human Identifier API Documentation

## Overview

The `HumanIdentificationService` provides real-time person detection and tracking with persistent IDs across video frames. It automatically selects and saves the highest quality crop of each person when they leave the camera view, optimized for PPE detection applications.

## Installation Requirements

```python
import cv2
import numpy as np
from human_identifier import HumanIdentificationService
```

## API Reference

### `HumanIdentificationService`

#### Initialization

```python
HumanIdentificationService(
    model_path=None,
    edge_margin=0,
    min_tracking_time=2.0,
    max_centroid_distance=150,
    conf_threshold=0.15
)
```

**Parameters:**
- `model_path` (str, optional): Path to YOLO ONNX model file. Auto-detects `model/yolov8n.onnx` if None.
- `edge_margin` (int, optional): Pixels from frame edge to filter detections. Use to prevent duplicate IDs at boundaries. Default: 0 (disabled).
- `min_tracking_time` (float, optional): Minimum seconds a person must be tracked before saving. Default: 2.0.
- `max_centroid_distance` (int, optional): Maximum pixel distance for matching detections across frames. Default: 150.
- `conf_threshold` (float, optional): YOLO confidence threshold for detections. Default: 0.15.

**Raises:**
- `FileNotFoundError`: If model file not found at specified path.

---

#### `process_frame(frame)`

Process a single video frame to detect and track persons.

**Parameters:**
- `frame` (numpy.ndarray): BGR video frame from camera/video file.

**Returns:**
- `list[dict]`: List of detected/tracked persons. Each dict contains:
  - `id` (int): Persistent tracking ID
  - `bbox` (tuple): Bounding box as `(x, y, width, height)` or None if departed
  - `centroid` (tuple): Center point as `(cx, cy)`
  - `crop` (numpy.ndarray): Cropped person image (only present when `departed=True`)
  - `ready_to_save` (bool): True if person has been tracked for minimum time
  - `tracking_duration` (float): Seconds this person has been tracked
  - `is_new` (bool): True if this is a newly saved person (first time returned with crop)
  - `departed` (bool): True if person left frame and crop is available

**Notes:**
- Call this method once per frame in your video processing loop
- Persons with `departed=True` have left the frame and contain their best quality crop
- Quality selection prioritizes: close-up shots (40%), fully in-frame (20%), standing pose (20%), centered (20%)

---

#### `get_all_tracked_crops()`

Retrieve best quality crops for all currently tracked persons. Use when shutting down to save persons still in frame.

**Parameters:** None

**Returns:**
- `list[dict]`: List of tracked persons meeting save criteria. Each dict contains:
  - `id` (int): Tracking ID
  - `crop` (numpy.ndarray): Best quality cropped person image
  - `quality` (float): Quality score 0.0-1.0
  - `duration` (float): Seconds tracked

**Notes:**
- Only returns persons tracked for at least `min_tracking_time` seconds
- Only returns persons with quality score > 0.3
- Excludes persons already saved via `process_frame`

---

#### `reset()`

Reset all tracking state. Clears all tracked IDs and saved person records.

**Parameters:** None

**Returns:** None

---

## Example Usage

### Basic Video Processing

```python
import cv2
from human_identifier import HumanIdentificationService

# Initialize service
service = HumanIdentificationService(
    edge_margin=0,
    min_tracking_time=2.0,
    conf_threshold=0.15
)

# Open video source
cap = cv2.VideoCapture(0)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame
        results = service.process_frame(frame)
        
        # Handle results
        for person in results:
            # Save departed persons
            if person['departed'] and person['is_new'] and person['crop'] is not None:
                filename = f"person_{person['id']}.jpg"
                cv2.imwrite(filename, person['crop'])
                print(f"Saved {filename} (quality crop)")
            
            # Visualize active tracks
            elif person['bbox'] is not None:
                x, y, w, h = person['bbox']
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, f"ID: {person['id']}", (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        cv2.imshow('Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Save remaining tracked persons on exit
    remaining = service.get_all_tracked_crops()
    for person in remaining:
        filename = f"person_{person['id']}.jpg"
        cv2.imwrite(filename, person['crop'])
        print(f"Saved {filename} (quality: {person['quality']:.2f})")

finally:
    cap.release()
    cv2.destroyAllWindows()
```

### Factory Floor PPE Monitoring

```python
import cv2
import os
from human_identifier import HumanIdentificationService

# Initialize with edge filtering to prevent duplicate IDs at doorways
service = HumanIdentificationService(
    edge_margin=50,  # Ignore detections within 50px of frame edge
    min_tracking_time=3.0,  # Require 3 seconds for good PPE visibility
    conf_threshold=0.20  # Higher confidence for reliability
)

output_dir = "ppe_checks"
os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture("factory_camera_feed.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    results = service.process_frame(frame)
    
    for person in results:
        if person['departed'] and person['crop'] is not None:
            # Save for PPE compliance review
            filepath = os.path.join(output_dir, f"worker_{person['id']}.jpg")
            cv2.imwrite(filepath, person['crop'])
            
            # Send to PPE detection system
            # ppe_detector.check_compliance(person['crop'])

cap.release()
```

## Quality Scoring

The service automatically evaluates each frame and saves the **best quality crop** when a person departs. Quality is scored based on:

1. **Size (40%)**: Larger bounding box = closer to camera = more PPE detail visible
2. **Fully In-Frame (20%)**: Complete body visible (critical for boots, gloves, vest)
3. **Aspect Ratio (20%)**: Standing person pose (1.5-2.5 height/width ratio)
4. **Centering (20%)**: Centered in frame = better lighting and less distortion

Crops are only saved if quality score > 0.3 and tracking duration ≥ `min_tracking_time`.

## Performance Considerations

- **FPS**: Processes at ~10-20 FPS on CPU (depends on frame resolution and detections)
- **Memory**: Stores one crop per tracked ID (~100KB-500KB each)
- **Tracking Persistence**: Persons reappear within 10 frames (~0.67s at 15 FPS) maintain same ID
- **Edge Margin**: Recommended 30-50px for doorways/entrances to prevent ID switches

## Limitations

- Designed for person detection only (COCO class 0)
- Requires YOLO v8 ONNX model
- Best performance with stationary camera
- Quality scores optimized for standing persons (PPE use case)
