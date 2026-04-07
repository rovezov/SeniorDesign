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

Controls (Display Mode):
    Press 'q' or ESC to quit
"""

import argparse
import time
import os
import glob
import cv2
import logging
from datetime import datetime
from camera.camera_feed import Camera
from detection_pipeline import DetectionPipeline
from display import DisplayRenderer


def _ts() -> str:
    """Current wall-clock time as HH:MM:SS.mmm for log prefixes."""
    t = time.localtime()
    ms = int((time.time() % 1) * 1000)
    return f"{t.tm_hour:02d}:{t.tm_min:02d}:{t.tm_sec:02d}.{ms:03d}"


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
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return log_file


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
        ppe_status = {'missing': [], 'compliant': True, 'is_non_compliant': False}
    
    # Use name or "unknown_person" if name is Unknown
    name_part = name if name.lower() != "unknown" else "unknown_person"
    
    # Build filename based on compliance status (includes person_id to prevent overwrites)
    if ppe_status['is_non_compliant']:
        missing_str = "_".join(ppe_status['missing'])
        filename = os.path.join(output_dir, f"{name_part}_{person_id}_missing_{missing_str}.jpg")
        status_str = f"MISSING: {', '.join(ppe_status['missing']).upper()}"
    else:
        filename = os.path.join(output_dir, f"{name_part}_{person_id}_compliant.jpg")
        status_str = "COMPLIANT"
    
    cv2.imwrite(filename, crop_image)
    log_msg = f"[SAVE] person_id_{person_id} - Name: {name} - PPE Status: {status_str}"
    logging.info(log_msg)


def run_pipeline_only(camera_id=0, fps=60, duration=None, edge_margin=0, verbose=False, ppe_requirements=None, camera_backend='auto'):
    """
    Run detection pipeline without display (ideal for embedded/headless systems)
    
    Args:
        camera_id: Camera device ID
        fps: Frames per second
        duration: Duration in seconds (None for indefinite)
        edge_margin: Pixels from edge to filter detections
        verbose: Enable verbose logging
        ppe_requirements: List of required PPE items
        camera_backend: 'auto', 'pi', or 'usb'
    """
    def vprint(*args, **kwargs):
        if verbose:
            logging.info(' '.join(str(arg) for arg in args))
    
    # Output directory
    output_dir = os.path.join(os.path.dirname(__file__), 'Output')
    setup_logging(output_dir)
    
    logging.info(f"Starting pipeline-only mode (Camera {camera_id}, {fps} FPS, backend={camera_backend})")
    vprint(f"Starting pipeline-only mode (Camera {camera_id}, {fps} FPS, backend={camera_backend})")
    
    # Initialize camera
    camera = Camera(camera_id=camera_id, fps=fps, backend=camera_backend)
    
    # Initialize detection pipeline
    ppe_requirements = ppe_requirements or ['vest']
    pipeline = DetectionPipeline(
        ppe_requirements=ppe_requirements,
        min_tracking_time=0.5,
        edge_margin=edge_margin,
        verbose=verbose
    )
    
    # Clear previous output images
    for img_path in glob.glob(os.path.join(output_dir, '*.jpg')) + glob.glob(os.path.join(output_dir, '*.png')):
        try:
            os.remove(img_path)
        except Exception:
            pass
    
    frame_count = 0
    last_cleanup = time.time()
    cleanup_interval = 3600  # 1 hour
    
    logging.info("Pipeline running... (Press Ctrl+C to stop)")
    vprint("Pipeline running... (Press Ctrl+C to stop)")
    
    try:
        camera.start()
        
        for frame in camera.get_frames(duration=duration):
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


def run_display_mode(camera_id=0, fps=60, duration=None, edge_margin=0, verbose=False, ppe_requirements=None, camera_backend='auto'):
    """
    Run detection pipeline with real-time video display and overlays
    
    Args:
        camera_id: Camera device ID
        fps: Frames per second
        duration: Duration in seconds (None for indefinite)
        edge_margin: Pixels from edge to filter detections
        verbose: Enable verbose logging
        ppe_requirements: List of required PPE items
        camera_backend: 'auto', 'pi', or 'usb'
    """
    def vprint(*args, **kwargs):
        if verbose:
            logging.info(' '.join(str(arg) for arg in args))
    
    # Output directory
    output_dir = os.path.join(os.path.dirname(__file__), 'Output')
    setup_logging(output_dir)
    
    logging.info(f"Starting display mode (Camera {camera_id}, {fps} FPS, backend={camera_backend})")
    vprint(f"Starting display mode (Camera {camera_id}, {fps} FPS, backend={camera_backend})")
    
    # Initialize camera
    camera = Camera(camera_id=camera_id, fps=fps, backend=camera_backend)
    
    # Initialize detection pipeline
    ppe_requirements = ppe_requirements or ['vest']
    pipeline = DetectionPipeline(
        ppe_requirements=ppe_requirements,
        min_tracking_time=0.5,
        edge_margin=edge_margin,
        verbose=verbose
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
            
            # Clean up departed IDs immediately after processing
            if had_departures:
                cleared = pipeline.cleanup_departed()
                if cleared > 0:
                    vprint(f"[{frame_count}|{_ts()}] Cleaned up {cleared} departed IDs from memory")
            
            # Calculate FPS
            elapsed = time.time() - t0 or 1e-6
            fps_history.append(1.0 / elapsed)
            if len(fps_history) > 30:
                fps_history.pop(0)
            avg_fps = sum(fps_history) / len(fps_history)
            
            # ------------------------------------------------------------------
            # Draw statistics overlay
            # ------------------------------------------------------------------
            DisplayRenderer.draw_statistics_overlay(frame, results, pipeline, avg_fps)
            
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
    
    args = parser.parse_args()
    
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
            camera_backend=args.camera_backend
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
            camera_backend=args.camera_backend
        )


if __name__ == '__main__':
    main()
