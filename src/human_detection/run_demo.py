"""
Human Detection and Tracking Demo

This script demonstrates person detection and tracking using YOLO ONNX model
with centroid-based tracking to assign persistent IDs. Automatically saves one
cropped image per tracked person ID to the Output_images folder.

Usage:
    python run_demo.py [--camera CAMERA_ID] [--fps FPS] [--duration DURATION] [--edge-margin PIXELS]

Controls:
    - Press 'q' or ESC to quit

Note:
    - The Output_images folder is cleared when the script starts
    - One cropped image per person ID is saved after 2 seconds of stable tracking
    - Use --edge-margin to filter detections near frame edges (helps prevent duplicate IDs)
"""

import cv2
import argparse
import time
import os
import glob
from temp_camera import TempCamera
from human_identifier import HumanTracker, detect_humans, crop_person


def run_demo(camera_id=0, fps=15, duration=None, edge_margin=0, debug=False):
    """
    Run the human detection and tracking demo
    
    Args:
        camera_id: Camera device ID (default: 0)
        fps: Frames per second to process (default: 15)
        duration: Duration in seconds to run (None = indefinite)
        edge_margin: Pixels from edge to filter detections (0 to disable, default: 0)
        debug: Enable debug output (default: False)
    """
    print(f"Starting human detection demo...")
    print(f"Camera: {camera_id}, FPS: {fps}, Edge margin: {edge_margin}px, Debug: {debug}")
    print(f"Press 'q' or ESC to quit")
    
    # Initialize camera and tracker
    camera = TempCamera(camera_id=camera_id, fps=fps)
    # Tracker with improved parameters for walking pace detection
    tracker = HumanTracker(max_disappeared=30, persistence_window=10, 
                          min_tracking_time=2.0, max_centroid_distance=150)
    
    # Create output directory and clear existing images
    output_dir = os.path.join(os.path.dirname(__file__), 'Output_images')
    os.makedirs(output_dir, exist_ok=True)
    
    # Clear all existing images in the folder
    existing_images = glob.glob(os.path.join(output_dir, '*.jpg'))
    existing_images += glob.glob(os.path.join(output_dir, '*.png'))
    for img_path in existing_images:
        try:
            os.remove(img_path)
        except Exception as e:
            print(f"Warning: Could not remove {img_path}: {e}")
    print(f"Cleared {len(existing_images)} existing images from output folder")
    
    # Track which person IDs have been saved
    saved_ids = set()
    
    fps_history = []
    frame_count = 0
    
    try:
        camera.start()
        
        for frame in camera.get_frames(duration=duration):
            t0 = time.time()
            frame_count += 1
            
            # Print debug info every 30 frames (about every 2 seconds at 15fps)
            debug_this_frame = debug and (frame_count % 30 == 1)
            
            if debug_this_frame:
                print(f"\n=== Frame {frame_count} ===")
            
            # Detect humans in frame (with edge filtering built-in)
            bboxes = detect_humans(frame, edge_margin=edge_margin, debug=debug_this_frame)
            
            # Update tracker with detections
            objects = tracker.update(bboxes)
            
            # Draw detections and IDs
            for (x, y, w, h) in bboxes:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Draw tracked objects with IDs and save cropped images for new IDs
            for object_id, centroid in objects.items():
                # Draw centroid
                cv2.circle(frame, tuple(centroid), 4, (0, 0, 255), -1)
                
                # Draw ID
                text = f"ID {object_id}"
                cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                # Check if ready to save (persistent tracking + min time)
                if tracker.is_ready_to_save(object_id):
                    tracking_duration = tracker.get_tracking_duration(object_id)
                    cv2.putText(frame, "TRACKED", (centroid[0] - 10, centroid[1] + 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    # Save cropped person image if not already saved
                    if object_id not in saved_ids:
                        # Find the bbox that contains this centroid
                        for (x, y, w, h) in bboxes:
                            cx = x + w // 2
                            cy = y + h // 2
                            # Check if this bbox's centroid matches the tracked centroid
                            if abs(cx - centroid[0]) < 20 and abs(cy - centroid[1]) < 20:
                                # Crop the person
                                person_crop = crop_person(frame, (x, y, w, h))
                                if person_crop is not None and person_crop.size > 0:
                                    # Save the cropped image
                                    filename = os.path.join(output_dir, f'person_id_{object_id}.jpg')
                                    cv2.imwrite(filename, person_crop)
                                    saved_ids.add(object_id)
                                    print(f"Saved cropped person: {filename} (tracked for {tracking_duration:.1f}s)")
                                break
            
            # Calculate and display FPS
            dt = time.time() - t0
            current_fps = 1.0 / dt if dt > 0 else 0.0
            fps_history.append(current_fps)
            if len(fps_history) > 30:
                fps_history.pop(0)
            avg_fps = sum(fps_history) / len(fps_history)
            
            # Display info
            cv2.putText(frame, f"FPS: {avg_fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            cv2.putText(frame, f"Detected: {len(bboxes)}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Tracked: {len(objects)}", (10, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Saved: {len(saved_ids)}", (10, 130),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Display frame
            cv2.imshow('Human Detection and Tracking Demo', frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):  # ESC or 'q'
                print("Exiting...")
                break
        
        print("Demo completed.")
        
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        camera.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='Human Detection and Tracking Demo')
    parser.add_argument('--camera', '-c', type=int, default=0,
                       help='Camera device ID (default: 0)')
    parser.add_argument('--fps', '-f', type=int, default=15,
                       help='Frames per second to process (default: 15)')
    parser.add_argument('--duration', '-d', type=float, default=None,
                       help='Duration in seconds to run (default: indefinite)')
    parser.add_argument('--edge-margin', '-e', type=int, default=0,
                       help='Pixels from edge to filter detections (0=disabled, default: 0)')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug output to diagnose detection issues')
    
    args = parser.parse_args()
    
    run_demo(camera_id=args.camera, fps=args.fps, duration=args.duration, 
             edge_margin=args.edge_margin, debug=args.debug)


if __name__ == '__main__':
    main()