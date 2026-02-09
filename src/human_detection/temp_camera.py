# Temporary Camera Module for Frame Capture

import cv2
import time

class TempCamera:
    """Captures frames from camera at a fixed rate"""
    def __init__(self, camera_id=0, fps=10):
        self.camera_id = camera_id
        self.fps = fps
        self.frame_interval = 1.0 / fps
        self.cap = None
        
    def start(self):
        """Initialize camera capture"""
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera {self.camera_id}")
        return self
        
    def get_frame(self):
        """Capture and return a single frame"""
        if self.cap is None:
            raise RuntimeError("Camera not started. Call start() first.")
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame
        
    def get_frames(self, duration=None):
        """
        Generator that yields frames at the specified FPS
        
        Args:
            duration: Optional duration in seconds. If None, runs indefinitely.
        """
        if self.cap is None:
            self.start()
            
        start_time = time.time()
        last_frame_time = 0
        
        while True:
            current_time = time.time()
            
            # Check duration limit if specified
            if duration is not None and (current_time - start_time) >= duration:
                break
                
            # Control FPS
            if (current_time - last_frame_time) >= self.frame_interval:
                frame = self.get_frame()
                if frame is None:
                    break
                last_frame_time = current_time
                yield frame
            else:
                # Small sleep to avoid busy waiting
                time.sleep(0.001)
                
    def release(self):
        """Release camera resources"""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        
