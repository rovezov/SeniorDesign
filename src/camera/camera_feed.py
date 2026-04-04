# Camera Feed Module
# Connects to camera and streams frames

import cv2
import time

class Camera:

    def __init__(self, camera_id=0, fps=30):
        self.camera_id = camera_id
        self.fps = fps
        self.frame_interval = 1.0 / fps
        self.cap = None
        
        
    def start_pipeline(self):
        """Initialize Raspberry Pi Camera pipeline"""
        
        pipeline = (
            f"qtiqmmfsrc camera={self.camera_id} ! "
            f"video/x-raw,format=NV12,width=1280,height=720,framerate={self.fps}/1 ! "
            "videoconvert ! "
            "appsink drop=true max-buffers=1 sync=false"
        )
        self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera {self.camera_id}")
        
        return self


    def get_frame(self):
        """Returns a single frame from the pipeline"""
        
        if self.cap is None:
            raise RuntimeError("Camera not started. Call start_pipeline() first.")
        
        ret, frame = self.cap.read()
        if not ret:
            return None

        return frame


    def get_frames(self, duration=None):
        """Continuously captures frames from the pipeline at a fixed fps
          
          Input: duration is an optional parameter measured in seconds. If not specified, it's default value is None, and frames will be captured indefinitely
          """
    
        if self.cap is None:
            self.start_pipeline()
        
        start_time = time.time()
        last_frame_time = 0
        
        while True:
            current_time = time.time()
                
            # Check duration limit if specified
            if duration is not None and (current_time - start_time) >= duration:
                break
                
            # Capture at specified fps
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