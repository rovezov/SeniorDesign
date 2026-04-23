# Camera Feed Module
# Connects to camera and streams frames
# Supports Rubik Pi 3 native camera, standard USB webcams, and H.265 UDP streaming
#
# === H.265 UDP Streaming Setup ===
# 
# To enable network streaming from Rubik Pi to Windows/Client:
# 
# 1. On Rubik Pi (Server Side):
#    - Use backend='stream' with the client's IP address:
#      camera = Camera(backend='stream', stream_host='192.168.1.178', stream_port=9001)
#    - This streams H.265-encoded video via UDP to the specified host:port
#    - Encoding uses UBWC compression for optimal performance
#    - Pipeline: qtiqmmfsrc → v4l2h265enc → rtph265pay → udpsink
#
# 2. On Windows (Client Side):
#    - Install VLC Media Player (https://www.videolan.org/vlc/)
#    - In VLC: Media → Open Network Stream
#    - Enter: udp://@:9001  (@ means listen on all interfaces)
#    - Note: Adjust port if using different stream_port
#    - For specific Rubik Pi IP: udp://<rubik-pi-ip>:9001
#
# Streaming requirements:
# - Both devices on same network (or port forwarding configured)
# - Firewall must allow UDP port 9001 (or configured stream_port)
# - GStreamer required on Rubik Pi (included in install.sh)
# - VLC Media Player required on Windows client

import cv2
import time
import logging
import subprocess


class Camera:

    def __init__(self, camera_id=0, fps=30, backend='webcam', stream_host=None, stream_port=9001):
        """
        Initialize camera with explicit backend selection.
        
        Args:
            camera_id: Camera device ID
            fps: Frames per second
            backend: 'pi' (Rubik Pi 3 camera), 'webcam' (USB/standard camera), 
                    or 'stream' (H.265 UDP streaming). Default: 'webcam'
            stream_host: Destination IP address for streaming (required if backend='stream')
            stream_port: UDP port for streaming (default: 9001)
        """
        self.camera_id = camera_id
        self.fps = fps
        self.frame_interval = 1.0 / fps
        self.cap = None
        self.backend = backend
        self.stream_host = stream_host
        self.stream_port = stream_port
        self.stream_process = None  # GStreamer process for streaming
        self.total_read_failures = 0
        self.consecutive_read_failures = 0
        self.total_frames_read = 0
        
        if backend not in ['pi', 'webcam', 'stream']:
            raise ValueError(f"Invalid backend '{backend}'. Must be 'pi', 'webcam', or 'stream'.")
        
        if backend == 'stream' and stream_host is None:
            raise ValueError("stream_host is required when backend='stream'")
        
    def start_pipeline(self):
        """Initialize camera pipeline (Pi, webcam, or H.265 UDP streaming)"""
        
        if self.backend == 'stream':
            build_info = cv2.getBuildInformation()
            if 'GStreamer:                   YES' not in build_info:
                raise RuntimeError(
                    'OpenCV in this environment does not include GStreamer support. '
                    'The streaming backend requires a GStreamer-enabled build.'
                )

            # Single-camera tee pipeline: one branch to local appsink (for detection),
            # one branch to UDP stream (for VLC client).
            pipeline_variants = [
                (
                    "x265enc_software",
                    (
                        f"qtiqmmfsrc camera={self.camera_id} ! "
                        f"video/x-raw,format=NV12,width=1280,height=720,framerate={self.fps}/1 ! "
                        f"tee name=t "
                        f"t. ! queue leaky=downstream max-size-buffers=1 ! videoconvert ! video/x-raw,format=BGR ! "
                        f"appsink drop=true max-buffers=1 sync=false "
                        f"t. ! queue leaky=downstream max-size-buffers=1 ! videoconvert ! video/x-raw,format=I420 ! "
                        f"x265enc speed-preset=ultrafast tune=zerolatency key-int-max=30 bitrate=2000 ! "
                        f"h265parse ! mpegtsmux ! "
                        f"udpsink host={self.stream_host} port={self.stream_port} sync=false async=false"
                    ),
                ),
                (
                    "v4l2h264_fallback",
                    (
                        f"qtiqmmfsrc camera={self.camera_id} ! "
                        f"video/x-raw,format=NV12,width=1280,height=720,framerate={self.fps}/1 ! "
                        f"tee name=t "
                        f"t. ! queue leaky=downstream max-size-buffers=1 ! videoconvert ! video/x-raw,format=BGR ! "
                        f"appsink drop=true max-buffers=1 sync=false "
                        f"t. ! queue leaky=downstream max-size-buffers=1 ! videoconvert ! video/x-raw,format=I420 ! "
                        f"v4l2h264enc capture-io-mode=5 output-io-mode=5 ! "
                        f"h264parse ! mpegtsmux ! "
                        f"udpsink host={self.stream_host} port={self.stream_port} sync=false async=false"
                    ),
                ),
            ]

            logging.info(f"Streaming backend initialized: {self.stream_host}:{self.stream_port}")
            startup_errors = []
            for variant_name, pipeline in pipeline_variants:
                logging.info(f"Trying stream capture variant={variant_name}")
                logging.info(f"Camera+stream pipeline: {pipeline}")
                cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
                if cap.isOpened():
                    self.cap = cap
                    logging.info(
                        f"Streaming+capture initialized successfully (variant={variant_name})"
                    )
                    return self
                startup_errors.append(f"{variant_name}: OpenCV could not open pipeline")
                cap.release()

            detail = "\n".join(startup_errors) if startup_errors else "Unknown streaming startup failure"
            raise RuntimeError(
                f"Failed to initialize camera streaming backend to {self.stream_host}:{self.stream_port}. "
                f"Details: {detail}"
            )
        
        if self.backend == 'pi':
            build_info = cv2.getBuildInformation()
            if 'GStreamer:                   YES' not in build_info:
                raise RuntimeError(
                    'OpenCV in this environment does not include GStreamer support. '
                    'The Rubik Pi camera backend requires a GStreamer-enabled build to open '
                    'the qtiqmmfsrc pipeline. Reinstall using install.sh and make sure no '
                    'conflicting opencv-python wheel is present.'
                )

            # Rubik Pi 3 optimized pipeline (for local frame capture)
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
        """Returns a single frame from the pipeline (or None if streaming)"""
        
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
          
          Note: In streaming mode ('stream' backend), frames are transmitted over UDP and 
          this method will yield None until the specified duration expires.
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
        """Release camera resources and stop streaming process if active"""
        if self.stream_process is not None:
            try:
                self.stream_process.terminate()
                self.stream_process.wait(timeout=2)
            except Exception as e:
                logging.warning(f"Failed to terminate streaming process: {e}")
                try:
                    self.stream_process.kill()
                except Exception:
                    pass
            self.stream_process = None
        
        if self.cap is not None:
            self.cap.release()
            self.cap = None
