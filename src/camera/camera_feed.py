import cv2
import time
import logging
import threading

_CAPTURE_OPEN_TIMEOUT = 10

def _open_capture_with_timeout(pipeline, api, timeout=_CAPTURE_OPEN_TIMEOUT):
    result = {}
    def _open():
        try:
            cap = cv2.VideoCapture(pipeline, api)
            result['cap'] = cap
        except Exception as exc:
            result['error'] = exc

    t = threading.Thread(target=_open, daemon=True)
    t.start()
    t.join(timeout=timeout)
    if t.is_alive() or 'error' in result:
        raise RuntimeError(f"VideoCapture failed/timed out for: {pipeline}")
    
    cap = result.get('cap')
    if cap is None or not cap.isOpened():
        raise RuntimeError("VideoCapture opened but reported isOpened()=False")
    return cap

class Camera:
    def __init__(self, camera_id=0, fps=30, backend='webcam', stream_host=None, stream_port=9001, device_path=None):
        self.camera_id = camera_id
        self.fps = fps
        self.frame_interval = 1.0 / fps
        self.cap = None
        self.backend = backend
        self.stream_host = stream_host
        self.stream_port = stream_port
        
        # Stats for the Health Monitor
        self.total_read_failures = 0
        self.consecutive_read_failures = 0
        self.total_frames_read = 0

    def start_pipeline(self):
        # Build camera capture pipeline. Streaming is handled in main.py so
        # we can transmit frames after overlays are drawn.
        if self.backend == 'stream':
            pipeline = (
                f"qtiqmmfsrc camera={self.camera_id} ! "
                "image/jpeg,width=640,height=480,framerate=30/1 ! "
                "jpegdec ! videoconvert ! video/x-raw,format=BGR ! "
                "appsink drop=true max-buffers=1 sync=false"
            )
            logging.info(
                f"Starting capture pipeline for overlay streaming to {self.stream_host}:{self.stream_port}"
            )
            self.cap = _open_capture_with_timeout(pipeline, cv2.CAP_GSTREAMER)
            
        elif self.backend == 'pi':
            pipeline = (
                f"qtiqmmfsrc camera={self.camera_id} ! "
                "image/jpeg,width=640,height=480,framerate=30/1 ! "
                "jpegdec ! videoconvert ! video/x-raw,format=BGR ! "
                "appsink drop=true max-buffers=1 sync=false"
            )
            self.cap = _open_capture_with_timeout(pipeline, cv2.CAP_GSTREAMER)
        else:
            self.cap = _open_capture_with_timeout(self.camera_id, cv2.CAP_ANY)
        
        return self

    def start(self):
        return self.start_pipeline()

    def get_frame(self):
        if self.cap is None: return None
        ret, frame = self.cap.read()
        if not ret:
            self.total_read_failures += 1
            self.consecutive_read_failures += 1
            return None
        self.total_frames_read += 1
        self.consecutive_read_failures = 0
        return frame

    def get_health_stats(self):
        """Restored for main.py health monitor"""
        return {
            'read_failures': self.total_read_failures,
            'consecutive_read_failures': self.consecutive_read_failures,
            'frames_read': self.total_frames_read,
        }

    def release(self):
        if self.cap:
            self.cap.release()
            self.cap = None

    def get_frames(self, duration=None):
        if self.cap is None: self.start()
        start_time = time.time()
        last_frame_time = 0
        while True:
            curr = time.time()
            if duration and (curr - start_time) >= duration: break
            if (curr - last_frame_time) >= self.frame_interval:
                frame = self.get_frame()
                if frame is None: break
                yield frame
                last_frame_time = curr
            else:
                time.sleep(0.001)