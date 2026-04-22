# Camera Feed Module
# Connects to camera and streams frames
# Supports both Rubik Pi 3 native camera and standard USB webcams

import cv2
import time
import logging


class Camera:

    def __init__(self, camera_id=0, fps=30, backend='webcam'):
        """
        Initialize camera with explicit backend selection.
        
        Args:
            camera_id: Camera device ID
            fps: Frames per second
            backend: 'pi' (Rubik Pi 3 camera) or 'webcam' (USB/standard camera). Default: 'webcam'
        """
        self.camera_id = camera_id
        self.fps = fps
        self.frame_interval = 1.0 / fps
        self.cap = None
        self.backend = backend
        self.total_read_failures = 0
        self.consecutive_read_failures = 0
        self.total_frames_read = 0
        
        if backend not in ['pi', 'webcam']:
            raise ValueError(f"Invalid backend '{backend}'. Must be 'pi' or 'webcam'.")
        
    def start_pipeline(self):
        """Initialize camera pipeline (Pi or webcam based on backend selection)"""
        
        if self.backend == 'pi':
            build_info = cv2.getBuildInformation()
            if 'GStreamer:                   YES' not in build_info:
                raise RuntimeError(
                    'OpenCV in this environment does not include GStreamer support. '
                    'The Rubik Pi camera backend requires a GStreamer-enabled build to open '
                    'the qtiqmmfsrc pipeline. Reinstall using install.sh and make sure no '
                    'conflicting opencv-python wheel is present.'
                )

            # Rubik Pi 3 optimized pipeline
            pipeline = (
                f"qtiqmmfsrc camera={self.camera_id} ! "
                f"video/x-raw,format=NV12,width=1280,height=720,framerate={self.fps}/1 ! "
                "videoconvert ! "
                "appsink drop=true max-buffers=1 sync=false"
            )
            logging.info(f"Camera backend=pi, cv2_version={cv2.__version__}, cv2_path={getattr(cv2, '__file__', 'unknown')}")
            logging.info(f"Camera pipeline: {pipeline}")
            self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        else:  # webcam
            # Standard USB/webcam pipeline (works on any OS)
            logging.info(f"Camera backend=webcam, camera_id={self.camera_id}")
            self.cap = cv2.VideoCapture(self.camera_id)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera {self.camera_id} using backend={self.backend}")
        
        return self

    def start(self):
        """Alias for start_pipeline() for compatibility with TempCamera"""
        return self.start_pipeline()

    def get_frame(self):
        """Returns a single frame from the pipeline"""
        
        if self.cap is None:
            raise RuntimeError("Camera not started. Call start_pipeline() or start() first.")
        
        ret, frame = self.cap.read()
        if not ret:
            self.total_read_failures += 1
            self.consecutive_read_failures += 1
            return None

        self.total_frames_read += 1
        self.consecutive_read_failures = 0

        return frame

    def get_health_stats(self):
        """Return lightweight camera-health counters for telemetry logging."""
        return {
            'read_failures': self.total_read_failures,
            'consecutive_read_failures': self.consecutive_read_failures,
            'frames_read': self.total_frames_read,
        }


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
