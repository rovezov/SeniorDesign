"""
Master Script - Run Detection Pipeline with Optional Display

Provides flexible execution modes:
- Pipeline only (no display) - for headless/embedded systems
- Display mode (with pipeline) - for development and monitoring

Supports multiple camera backends:
- webcam (default) - USB/standard camera (works on any OS)
- pi - Rubik Pi 3 native camera (GStreamer optimized)

Usage:
    python main.py --mode pipeline [--camera-backend webcam|pi] [--camera ID] [--fps FPS] [--verbose]
    python main.py --mode display [--camera-backend webcam|pi] [--camera ID] [--fps FPS] [--verbose]
    python main.py --mode pipeline --stream-host 192.168.1.178 --stream-port 9001 [--fps FPS]

Stream to Windows via H.265 UDP:
    python main.py --mode pipeline --camera-backend pi --stream-host <WINDOWS_IP> --stream-port 9001

Controls (Display Mode):
    Press 'q' or ESC to quit
"""

import argparse
import time
import os
import glob
import tempfile
import queue
import cv2
import logging
import traceback
import threading
from datetime import datetime
from logging.handlers import QueueHandler, QueueListener
from camera.camera_feed import Camera
from detection_pipeline import DetectionPipeline
from display import DisplayRenderer

SILENCEABLE_MODELS = ['ppe', 'face', 'human', 'matching']


def _ts() -> str:
    """Current wall-clock time as HH:MM:SS.mmm for log prefixes."""
    t = time.localtime()
    ms = int((time.time() % 1) * 1000)
    return f"{t.tm_hour:02d}:{t.tm_min:02d}:{t.tm_sec:02d}.{ms:03d}"


def configure_runtime_threads(onnx_threads: int = 1):
    """Set conservative runtime thread limits for edge stability."""
    t = str(max(1, int(onnx_threads)))
    os.environ.setdefault("OMP_NUM_THREADS", t)
    os.environ.setdefault("OPENBLAS_NUM_THREADS", t)
    os.environ.setdefault("MKL_NUM_THREADS", t)
    os.environ.setdefault("NUMEXPR_NUM_THREADS", t)
    os.environ.setdefault("ORT_INTRA_OP_NUM_THREADS", t)
    os.environ.setdefault("ORT_INTER_OP_NUM_THREADS", "1")


def setup_logging(output_dir):
    """
    Set up logging to both console and file.
    
    Args:
        output_dir: Directory to save log files (will be created if needed)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create log filename with timestamp
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # log_file = os.path.join(output_dir, f"session_{timestamp}.txt")
    log_file = os.path.join(output_dir, f"session_log.txt")
    
    # Get root logger and clear existing handlers
    logger = logging.getLogger()
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Configure logging with fresh handlers
    logger.setLevel(logging.INFO)

    # File handler
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))

    log_queue = queue.Queue()
    queue_handler = QueueHandler(log_queue)
    logger.addHandler(queue_handler)

    listener = QueueListener(log_queue, file_handler, console_handler, respect_handler_level=True)
    listener.start()

    return log_file, listener


def log_runtime_diagnostics(camera_backend, fps):
    """Log environment details that help diagnose board-side crashes."""
    try:
        cv2_path = getattr(cv2, '__file__', 'unknown')
        build = cv2.getBuildInformation()
        gst_enabled = 'GStreamer:                   YES' in build
        ffmpeg_enabled = 'FFMPEG:                      YES' in build
        logging.info(f"Runtime diagnostics: cv2_version={cv2.__version__}, cv2_path={cv2_path}")
        logging.info(f"Runtime diagnostics: backend={camera_backend}, requested_fps={fps}, gst_enabled={gst_enabled}, ffmpeg_enabled={ffmpeg_enabled}")
    except Exception as e:
        logging.info(f"Runtime diagnostics unavailable: {e}")


class RuntimeHealthMonitor:
    """Periodic runtime telemetry logger for edge diagnostics."""

    def __init__(self, camera, frame_count_getter, interval_sec: float = 1.0):
        self._camera = camera
        self._frame_count_getter = frame_count_getter
        self._interval_sec = max(0.5, float(interval_sec))
        self._stop_event = threading.Event()
        self._thread = None
        self._temp_path = self._discover_temp_path()

    def _discover_temp_path(self):
        base = "/sys/class/thermal"
        try:
            zones = [z for z in os.listdir(base) if z.startswith("thermal_zone")]
        except Exception:
            return None

        fallback = None
        for zone in zones:
            zone_dir = os.path.join(base, zone)
            type_path = os.path.join(zone_dir, "type")
            temp_path = os.path.join(zone_dir, "temp")
            if not os.path.exists(temp_path):
                continue
            if fallback is None:
                fallback = temp_path
            try:
                with open(type_path, "r") as f:
                    zone_type = f.read().strip().lower()
                if "cpu" in zone_type:
                    return temp_path
            except Exception:
                continue
        return fallback

    def _read_cpu_temp_c(self):
        if not self._temp_path:
            return None
        try:
            with open(self._temp_path, "r") as f:
                raw = f.read().strip()
            value = float(raw)
            return value / 1000.0 if value > 1000 else value
        except Exception:
            return None

    def _read_mem_available_mb(self):
        try:
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    if line.startswith("MemAvailable:"):
                        parts = line.split()
                        kb = int(parts[1])
                        return kb // 1024
        except Exception:
            pass
        return None

    def _log_once(self):
        frame_count = int(self._frame_count_getter())
        mem_mb = self._read_mem_available_mb()
        temp_c = self._read_cpu_temp_c()
        camera_stats = self._camera.get_health_stats() if self._camera else {}

        mem_str = "NA" if mem_mb is None else str(mem_mb)
        temp_str = "NA" if temp_c is None else f"{temp_c:.1f}"
        read_failures = camera_stats.get("read_failures", "NA")
        consecutive_failures = camera_stats.get("consecutive_read_failures", "NA")

        logging.info(
            "Health heartbeat: "
            f"frame={frame_count}, "
            f"mem_available_mb={mem_str}, "
            f"cpu_temp_c={temp_str}, "
            f"camera_read_failures={read_failures}, "
            f"camera_consecutive_failures={consecutive_failures}"
        )

    def _run(self):
        while not self._stop_event.wait(self._interval_sec):
            self._log_once()

    def start(self):
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, daemon=True, name="RuntimeHealthMonitor")
        self._thread.start()
        logging.info("Health monitor started (1s interval)")

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        logging.info("Health monitor stopped")


class OverlayStreamer:
    """Pushes processed BGR frames to UDP via GStreamer at fixed output FPS."""

    def __init__(self, host: str, port: int, fps: int):
        self.host = host
        self.port = int(port)
        self.fps = max(1, int(fps))
        self._writer = None
        self._size = None
        self._latest_frame = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = None

    def _build_pipeline(self, width: int, height: int) -> str:
        return (
            "appsrc is-live=true block=false format=time do-timestamp=true ! "
            f"video/x-raw,format=BGR,width={width},height={height},framerate={self.fps}/1 ! "
            "queue leaky=downstream max-size-buffers=1 ! "
            "videoconvert ! video/x-raw,format=NV12 ! "
            "v4l2h264enc ! h264parse config-interval=-1 ! mpegtsmux ! "
            f"udpsink host={self.host} port={self.port} sync=false async=false"
        )

    def _open_writer(self, width: int, height: int):
        size = (width, height)
        pipeline = self._build_pipeline(width, height)
        writer = cv2.VideoWriter(pipeline, cv2.CAP_GSTREAMER, 0, float(self.fps), size, True)
        if not writer.isOpened():
            writer.release()
            raise RuntimeError(
                f"Failed to open overlay stream writer for {self.host}:{self.port} at {width}x{height}"
            )
        self._writer = writer
        self._size = size
        logging.info(f"Overlay streaming started: udp://{self.host}:{self.port} ({width}x{height}@{self.fps})")

    def _run(self):
        frame_interval = 1.0 / float(self.fps)
        while not self._stop_event.is_set():
            t0 = time.time()

            frame = None
            with self._lock:
                if self._latest_frame is not None:
                    frame = self._latest_frame

            if frame is not None:
                h, w = frame.shape[:2]
                target_size = (w, h)
                try:
                    if self._writer is None:
                        self._open_writer(w, h)
                    elif self._size != target_size:
                        self._writer.release()
                        self._writer = None
                        self._size = None
                        self._open_writer(w, h)
                    self._writer.write(frame)
                except Exception as e:
                    logging.info(f"Overlay streaming write error: {e}")

            elapsed = time.time() - t0
            delay = frame_interval - elapsed
            if delay > 0:
                time.sleep(delay)

    def write(self, frame):
        if frame is None or getattr(frame, "shape", None) is None:
            return

        # Keep only the latest processed frame. The sender thread transmits this
        # at fixed FPS, which keeps network output smooth even under heavy inference load.
        with self._lock:
            self._latest_frame = frame

        if self._thread is None:
            self._thread = threading.Thread(target=self._run, daemon=True, name="OverlayStreamer")
            self._thread.start()

    def release(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

        with self._lock:
            self._latest_frame = None

        if self._writer is not None:
            self._writer.release()
            self._writer = None
            self._size = None
            logging.info("Overlay streaming stopped")


def save_person_info(person_id, name, crop_image, output_dir, ppe_status=None):
    """
    Save departed person's information
    
    Args:
        person_id: Unique tracking ID
        name: Identified name or "Unknown"
        crop_image: Cropped person image
        output_dir: Directory to save to
        ppe_status: Dict with PPE compliance info
    """
    if ppe_status is None:
        # Update default dictionary fallback to use the new keys
        ppe_status = {
            'current_missing': [], 'current_compliant': True,
            'cumulative_missing': [], 'cumulative_compliant': True
        }
    
    # Use name or "unknown_person" if name is Unknown
    name_part = name if name.lower() != "unknown" else "unknown_person"
    
    # Build filename based on CUMULATIVE compliance status (majority vote over time)
    if not ppe_status.get('cumulative_compliant', True):
        missing_str = "_".join(ppe_status.get('cumulative_missing', []))
        filename = os.path.join(output_dir, f"{name_part}_{person_id}_missing_{missing_str}.jpg")
        
        missing_items = ppe_status.get('cumulative_missing', [])
        status_str = f"MISSING: {', '.join(missing_items).upper()}"
    else:
        filename = os.path.join(output_dir, f"{name_part}_{person_id}_compliant.jpg")
        status_str = "COMPLIANT"
    
    if crop_image is None or getattr(crop_image, "size", 0) == 0:
        logging.info(f"[SAVE-ERROR] person_id_{person_id} - Empty crop, skipping save")
        return

    try:
        saved = _atomic_save_jpeg(filename, crop_image)
    except Exception as e:
        logging.info(f"[SAVE-ERROR] person_id_{person_id} - Exception while saving image: {e}")
        logging.info(traceback.format_exc())
        return

    if not saved:
        logging.info(f"[SAVE-ERROR] person_id_{person_id} - Failed to encode/write image")
        return

    file_size = os.path.getsize(filename) if os.path.exists(filename) else -1
    if file_size <= 0:
        logging.info(f"[SAVE-ERROR] person_id_{person_id} - Saved file is empty: {filename}")
        return

    log_msg = f"[SAVE] person_id_{person_id} - Name: {name} - PPE Status: {status_str} - bytes={file_size}"
    logging.info(log_msg)


def _atomic_save_jpeg(path: str, image, quality: int = 95) -> bool:
    """Write JPEG via temp file + fsync + atomic rename to avoid zero-byte outputs."""
    ok, encoded = cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        return False

    out_dir = os.path.dirname(path) or '.'
    os.makedirs(out_dir, exist_ok=True)

    fd, tmp_path = tempfile.mkstemp(prefix='.tmp_img_', suffix='.jpg', dir=out_dir)
    try:
        with os.fdopen(fd, 'wb') as f:
            f.write(encoded.tobytes())
            f.flush()
            os.fsync(f.fileno())

        os.replace(tmp_path, path)

        # Best-effort directory fsync so rename is durable after sudden power loss.
        try:
            dir_fd = os.open(out_dir, os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except Exception:
            pass

        return True
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


def run_pipeline_only(camera_id=0, fps=60, duration=None, edge_margin=0, verbose=False, ppe_requirements=None, camera_backend='auto', debug_camera=False, face_workers=1, stream_host=None, stream_port=9001, silenced_models=None):
    """
    Run detection pipeline without display (ideal for embedded/headless systems)
    
    Args:
        camera_id: Camera device ID
        fps: Frames per second
        duration: Duration in seconds (None for indefinite)
        edge_margin: Pixels from edge to filter detections
        verbose: Enable verbose logging
        ppe_requirements: List of required PPE items
        camera_backend: 'auto', 'pi', or 'webcam'
        debug_camera: Enable periodic camera heartbeat logging
        face_workers: Number of face matching worker threads
        stream_host: Destination IP for H.265 UDP streaming (enables streaming mode if provided)
        stream_port: UDP port for streaming (default: 9001)
    """
    def vprint(*args, **kwargs):
        if verbose:
            logging.info(' '.join(str(arg) for arg in args))
    
    # Output directory
    output_dir = os.path.join(os.path.dirname(__file__), 'Output')
    _, log_listener = setup_logging(output_dir)
    log_runtime_diagnostics(camera_backend, fps)
    
    # Determine actual backend
    if stream_host:
        actual_backend = 'stream'
        logging.info(f"Starting pipeline-only mode with overlay streaming to {stream_host}:{stream_port}")
        vprint(f"Starting pipeline-only mode with overlay streaming to {stream_host}:{stream_port}")
    else:
        actual_backend = camera_backend
        logging.info(f"Starting pipeline-only mode (Camera {camera_id}, {fps} FPS, backend={camera_backend})")
        vprint(f"Starting pipeline-only mode (Camera {camera_id}, {fps} FPS, backend={camera_backend})")
    
    # Initialize camera with streaming if requested
    camera = Camera(
        camera_id=camera_id,
        fps=fps,
        backend=actual_backend,
        stream_host=stream_host,
        stream_port=stream_port
    )
    
    # Initialize detection pipeline
    ppe_requirements = ppe_requirements or ['vest']
    pipeline = DetectionPipeline(
        ppe_requirements=ppe_requirements,
        min_tracking_time=0.5,
        edge_margin=edge_margin,
        verbose=verbose,
        face_workers=face_workers,
        silenced_models=silenced_models,
    )
    
    # Clear previous output images
    for img_path in glob.glob(os.path.join(output_dir, '*.jpg')) + glob.glob(os.path.join(output_dir, '*.png')):
        try:
            os.remove(img_path)
        except Exception:
            pass
    
    frame_count = 0
    fps_history = []
    last_cleanup = time.time()
    cleanup_interval = 3600  # 1 hour
    health_monitor = RuntimeHealthMonitor(camera, lambda: frame_count, interval_sec=1.0)
    streamer = OverlayStreamer(stream_host, stream_port, fps) if stream_host else None
    
    logging.info("Pipeline running... (Press Ctrl+C to stop)")
    vprint("Pipeline running... (Press Ctrl+C to stop)")
    
    try:
        logging.info("Stage: camera.start begin")
        camera.start()
        logging.info("Stage: camera.start complete")
        health_monitor.start()
        
        for frame in camera.get_frames(duration=duration):
            t0 = time.time()
            frame_count += 1
            # print(f"\rProcessing frame {frame_count}...", end='', flush=True)
            # continue
            if debug_camera and frame_count % 30 == 0:
                shape = None if frame is None else tuple(frame.shape)
                logging.info(f"Debug heartbeat: frame={frame_count}, shape={shape}")
            
            # ------------------------------------------------------------------
            # Process frame through detection pipeline
            # ------------------------------------------------------------------
            try:
                results = pipeline.process_frame(frame)
            except Exception as e:
                logging.info(f"Stage: pipeline.process_frame failed on frame={frame_count}: {e}")
                logging.info(traceback.format_exc())
                raise
            
            # Track if we had any departures in this frame
            had_departures = False
            
            for person in results:
                person_id = person['id']
                
                # Handle departed persons
                if person.get('departed', False):
                    if person['is_new'] and person['crop'] is not None:
                        # Save person information
                        name = pipeline.person_tracker.get_name(person_id)
                        ppe_status = pipeline.person_tracker.get_ppe_status(person_id)
                        save_person_info(person_id, name, person['crop'], output_dir, ppe_status=ppe_status)
                        
                        # Remove from tracking
                        pipeline.person_tracker.remove_person(person_id)
                        pipeline.in_flight.pop(person_id, None)
                        
                        vprint(f"[{frame_count}|{_ts()}] Person {person_id} departed - Name: {name}")
                        had_departures = True

                if person.get('departed', False):
                    continue

                # Draw overlays even in pipeline-only mode when streaming.
                if streamer is not None:
                    DisplayRenderer.draw_person_overlays(frame, person, pipeline)

            elapsed = max(1e-6, time.time() - t0)
            current_fps = 1.0 / elapsed
            fps_history.append(current_fps)
            if len(fps_history) > 15:
                fps_history.pop(0)
            avg_fps = sum(fps_history) / len(fps_history)

            # Draw FPS/statistics on streamed output frames.
            if streamer is not None:
                DisplayRenderer.draw_statistics_overlay(
                    frame,
                    results,
                    pipeline,
                    current_fps=current_fps,
                    avg_fps=avg_fps,
                )
            
            # Clean up departed IDs immediately after processing
            if had_departures:
                cleared = pipeline.cleanup_departed()
                if cleared > 0:
                    vprint(f"[{frame_count}|{_ts()}] Cleaned up {cleared} departed IDs from memory")
            
            # Periodic memory cleanup
            if time.time() - last_cleanup > cleanup_interval:
                cleared = pipeline.cleanup_departed()
                vprint(f"[CLEANUP] Cleared {cleared} departed person IDs from memory")
                last_cleanup = time.time()

            if streamer is not None:
                streamer.write(frame)
        
        # Handle remaining tracked persons on exit
        vprint(f"\nProcessed {frame_count} frames")
        remaining_crops = pipeline.get_remaining_crops()
        for person_data in remaining_crops:
            person_id = person_data['id']
            crop = person_data['crop']
            
            name = pipeline.person_tracker.get_name(person_id)
            ppe_status = pipeline.person_tracker.get_ppe_status(person_id)
            save_person_info(person_id, name, crop, output_dir, ppe_status=ppe_status)
            
            vprint(f"[EXIT] Saved person_id_{person_id} - Name: {name} (quality {person_data['quality']:.2f})")
        
        vprint(f"Total persons saved: {len(pipeline.identification_service.saved_ids)}")
        
    except KeyboardInterrupt:
        vprint(f"\n\nInterrupted. Processed {frame_count} frames")
        # Save remaining on interrupt
        remaining_crops = pipeline.get_remaining_crops()
        for person_data in remaining_crops:
            person_id = person_data['id']
            name = pipeline.person_tracker.get_name(person_id)
            ppe_status = pipeline.person_tracker.get_ppe_status(person_id)
            save_person_info(person_id, name, person_data['crop'], output_dir, ppe_status=ppe_status)
        vprint(f"Total persons saved: {len(pipeline.identification_service.saved_ids)}")
    except Exception as e:
        logging.info(f"Pipeline exception: {e}")
        logging.info(traceback.format_exc())
        vprint(f"Error: {e}")
        if verbose:
            traceback.print_exc()
    finally:
        logging.info("Stage: cleanup begin")
        health_monitor.stop()
        pipeline.stop()
        camera.release()
        if streamer is not None:
            streamer.release()
        logging.info("Stage: cleanup complete")
        if log_listener is not None:
            log_listener.stop()


def run_display_mode(camera_id=0, fps=60, duration=None, edge_margin=0, verbose=False, ppe_requirements=None, camera_backend='auto', stream_host=None, stream_port=9001, silenced_models=None):
    """
    Run detection pipeline with real-time video display and overlays
    
    Args:
        camera_id: Camera device ID
        fps: Frames per second
        duration: Duration in seconds (None for indefinite)
        edge_margin: Pixels from edge to filter detections
        verbose: Enable verbose logging
        ppe_requirements: List of required PPE items
        camera_backend: 'auto', 'pi', or 'webcam'
        stream_host: Destination IP for H.265 UDP streaming (enables streaming mode if provided)
        stream_port: UDP port for streaming (default: 9001)
    """
    def vprint(*args, **kwargs):
        if verbose:
            logging.info(' '.join(str(arg) for arg in args))
    
    # Output directory
    output_dir = os.path.join(os.path.dirname(__file__), 'Output')
    _, log_listener = setup_logging(output_dir)
    
    # Determine actual backend
    if stream_host:
        actual_backend = 'stream'
        logging.info(f"Starting display mode with overlay streaming to {stream_host}:{stream_port}")
        vprint(f"Starting display mode with overlay streaming to {stream_host}:{stream_port}")
    else:
        actual_backend = camera_backend
        logging.info(f"Starting display mode (Camera {camera_id}, {fps} FPS, backend={camera_backend})")
        vprint(f"Starting display mode (Camera {camera_id}, {fps} FPS, backend={camera_backend})")
    
    # Initialize camera with streaming if requested
    camera = Camera(
        camera_id=camera_id,
        fps=fps,
        backend=actual_backend,
        stream_host=stream_host,
        stream_port=stream_port
    )
    
    # Initialize detection pipeline
    ppe_requirements = ppe_requirements or ['vest']
    pipeline = DetectionPipeline(
        ppe_requirements=ppe_requirements,
        min_tracking_time=0.5,
        edge_margin=edge_margin,
        verbose=verbose,
        silenced_models=silenced_models,
    )
    
    # Clear previous output images
    for img_path in glob.glob(os.path.join(output_dir, '*.jpg')) + glob.glob(os.path.join(output_dir, '*.png')):
        try:
            os.remove(img_path)
        except Exception:
            pass
    
    frame_count = 0
    fps_history = []
    last_cleanup = time.time()
    cleanup_interval = 3600  # 1 hour
    streamer = OverlayStreamer(stream_host, stream_port, fps) if stream_host else None
    
    logging.info("Press 'q' or ESC to quit")
    vprint("Press 'q' or ESC to quit")
    
    try:
        camera.start()
        
        for frame in camera.get_frames(duration=duration):
            t0 = time.time()
            frame_count += 1
            
            # ------------------------------------------------------------------
            # Process frame through detection pipeline
            # ------------------------------------------------------------------
            results = pipeline.process_frame(frame)
            
            # Track if we had any departures in this frame
            had_departures = False
            
            for person in results:
                person_id = person['id']
                
                # Handle departed persons
                if person.get('departed', False):
                    if person['is_new'] and person['crop'] is not None:
                        # Save person information
                        name = pipeline.person_tracker.get_name(person_id)
                        ppe_status = pipeline.person_tracker.get_ppe_status(person_id)
                        save_person_info(person_id, name, person['crop'], output_dir, ppe_status=ppe_status)
                        
                        # Remove from tracking
                        pipeline.person_tracker.remove_person(person_id)
                        pipeline.in_flight.discard(person_id)
                        
                        vprint(f"[{frame_count}|{_ts()}] Person {person_id} departed - Name: {name}")
                        had_departures = True
                    continue
                
                # ------------------------------------------------------------------
                # Draw overlays on frame
                # ------------------------------------------------------------------
                DisplayRenderer.draw_person_overlays(frame, person, pipeline)

            elapsed = max(1e-6, time.time() - t0)
            current_fps = 1.0 / elapsed
            fps_history.append(current_fps)
            if len(fps_history) > 15:
                fps_history.pop(0)
            avg_fps = sum(fps_history) / len(fps_history)
            DisplayRenderer.draw_statistics_overlay(
                frame,
                results,
                pipeline,
                current_fps=current_fps,
                avg_fps=avg_fps,
            )
            
            # Clean up departed IDs immediately after processing
            if had_departures:
                cleared = pipeline.cleanup_departed()
                if cleared > 0:
                    vprint(f"[{frame_count}|{_ts()}] Cleaned up {cleared} departed IDs from memory")
            
            if streamer is not None:
                streamer.write(frame)
            
            # Display frame
            cv2.imshow('Integrated Human Detection & Identification', frame)
            
            # Check for quit
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                logging.info("Exiting...")
                print("\nExiting...")
                break
            
            # Periodic memory cleanup
            if time.time() - last_cleanup > cleanup_interval:
                cleared = pipeline.cleanup_departed()
                vprint(f"[CLEANUP] Cleared {cleared} departed person IDs from memory")
                last_cleanup = time.time()
        
        # Handle remaining tracked persons on exit
        vprint(f"\nProcessed {frame_count} frames")
        remaining_crops = pipeline.get_remaining_crops()
        for person_data in remaining_crops:
            person_id = person_data['id']
            crop = person_data['crop']
            
            name = pipeline.person_tracker.get_name(person_id)
            ppe_status = pipeline.person_tracker.get_ppe_status(person_id)
            save_person_info(person_id, name, crop, output_dir, ppe_status=ppe_status)
            
            vprint(f"[EXIT] Saved person_id_{person_id} - Name: {name} (quality {person_data['quality']:.2f})")
        
        vprint(f"Total persons saved: {len(pipeline.identification_service.saved_ids)}")
        
    except KeyboardInterrupt:
        vprint(f"\n\nInterrupted. Processed {frame_count} frames")
        # Save remaining on interrupt
        remaining_crops = pipeline.get_remaining_crops()
        for person_data in remaining_crops:
            person_id = person_data['id']
            name = pipeline.person_tracker.get_name(person_id)
            ppe_status = pipeline.person_tracker.get_ppe_status(person_id)
            save_person_info(person_id, name, person_data['crop'], output_dir, ppe_status=ppe_status)
        vprint(f"Total persons saved: {len(pipeline.identification_service.saved_ids)}")
    except Exception as e:
        vprint(f"Error: {e}")
        import traceback
        if verbose:
            traceback.print_exc()
    finally:
        pipeline.stop()
        camera.release()
        if streamer is not None:
            streamer.release()
        logging.info("Stage: cleanup complete")
        if log_listener is not None:
            log_listener.stop()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description='Master Script: Detection Pipeline with Optional Display',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run pipeline only (no display) - default webcam
  python main.py --mode pipeline --verbose
  
  # Run with display (development/monitoring)
  python main.py --mode display --camera 0 --fps 30
  
  # Use Rubik Pi 3 camera backend
  python main.py --mode pipeline --camera-backend pi --verbose
  
  # Use USB/webcam backend (explicit)
  python main.py --mode display --camera-backend webcam --camera 0
  
  # Custom PPE requirements
  python main.py --mode pipeline --ppe helmet vest --camera-backend pi
        """
    )
    
    parser.add_argument('--mode', '-m', choices=['pipeline', 'display'], default='display',
                       help='Execution mode: pipeline (no display) or display (with visualization). Default: display')
    parser.add_argument('--camera-backend', '-b', choices=['webcam', 'pi'], default='webcam',
                       help='Camera backend: webcam (USB/standard), pi (Rubik Pi 3). Default: webcam')
    parser.add_argument('--camera', '-c', type=int, default=0,
                       help='Camera device ID (default: 0)')
    parser.add_argument('--fps', '-f', type=int, default=60,
                       help='Frames per second (default: 60)')
    parser.add_argument('--duration', '-d', type=float, default=None,
                       help='Duration in seconds (default: indefinite)')
    parser.add_argument('--edge-margin', '-e', type=int, default=0,
                       help='Pixels from edge to filter detections (default: 0)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output (default: False)')
    parser.add_argument('--ppe', type=str, nargs='+', default=['vest'],
                       help='PPE requirements (default: vest)')
    parser.add_argument('--debug-camera', action='store_true',
                       help='Enable periodic camera/pipeline heartbeat logging (default: False)')
    parser.add_argument('--face-workers', type=int, default=1,
                       help='Background face/PPE worker threads for edge mode (default: 1)')
    parser.add_argument('--stream-host', type=str, default=None,
                       help='Destination IP address for H.265 UDP streaming (enables streaming if provided)')
    parser.add_argument('--stream-port', type=int, default=9001,
                       help='UDP port for streaming (default: 9001)')
    parser.add_argument('--silence-models', type=str, nargs='+', choices=SILENCEABLE_MODELS, default=[],
                       help='Silence selected models and dry-run them. Choices: ppe, face, human, matching')
    
    args = parser.parse_args()
    configure_runtime_threads(onnx_threads=1)
    silenced_models = set(args.silence_models or [])
    if silenced_models:
        logging.info(f"Silenced models requested: {', '.join(sorted(silenced_models))}")
        print(f"Silenced models requested: {', '.join(sorted(silenced_models))}")
    
    if args.mode == 'pipeline':
        logging.info(f"Running in PIPELINE mode (no display) with camera backend={args.camera_backend}...")
        print(f"Running in PIPELINE mode (no display) with camera backend={args.camera_backend}...")
        run_pipeline_only(
            camera_id=args.camera,
            fps=args.fps,
            duration=args.duration,
            edge_margin=args.edge_margin,
            verbose=args.verbose,
            ppe_requirements=args.ppe,
            camera_backend=args.camera_backend,
            debug_camera=args.debug_camera,
            face_workers=args.face_workers,
            stream_host=args.stream_host,
            stream_port=args.stream_port,
            silenced_models=silenced_models,
        )
    else:  # display mode
        logging.info(f"Running in DISPLAY mode with camera backend={args.camera_backend}...")
        print(f"Running in DISPLAY mode with camera backend={args.camera_backend}...")
        run_display_mode(
            camera_id=args.camera,
            fps=args.fps,
            duration=args.duration,
            edge_margin=args.edge_margin,
            verbose=args.verbose,
            ppe_requirements=args.ppe,
            camera_backend=args.camera_backend,
            stream_host=args.stream_host,
            stream_port=args.stream_port,
            silenced_models=silenced_models,
        )


if __name__ == '__main__':
    main()
