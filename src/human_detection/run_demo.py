"""
Human Detection and Tracking Demo

Detects and tracks persons with persistent IDs, saving high-quality cropped images 
optimized for PPE detection when persons leave the camera view.

Usage:
    python run_demo.py [--camera ID] [--fps FPS] [--duration SEC] [--edge-margin PX]

Controls:
    Press 'q' or ESC to quit

Features:
    - Quality-based crop selection (saves best frame when person departs)
    - Persistent ID tracking across frames
    - Minimum 2 seconds tracking before saving
    - Output saved to Output_images/ folder
"""

import cv2
import argparse
import time
import os
import glob
from temp_camera import TempCamera
from human_identifier import HumanIdentificationService


def save_remaining_crops(identification_service, output_dir):
    """Save crops for persons still being tracked"""
    remaining_crops = identification_service.get_all_tracked_crops()
    for person_data in remaining_crops:
        filename = os.path.join(output_dir, f"person_id_{person_data['id']}.jpg")
        cv2.imwrite(filename, person_data['crop'])
        identification_service.saved_ids.add(person_data['id'])
        print(f"[EXIT] Saved person_id_{person_data['id']}.jpg (quality {person_data['quality']:.2f})")
    return len(remaining_crops)


def run_demo(camera_id=0, fps=15, duration=None, edge_margin=0):
    """Run human detection and tracking demo"""
    print(f"Starting detection service (Camera {camera_id}, {fps} FPS, Edge margin {edge_margin}px)")
    print("Press 'q' or ESC to quit")
    
    camera = TempCamera(camera_id=camera_id, fps=fps)
    identification_service = HumanIdentificationService(
        edge_margin=edge_margin,
        min_tracking_time=2.0,
        max_centroid_distance=150,
        conf_threshold=0.15
    )
    
    output_dir = os.path.join(os.path.dirname(__file__), 'Output_images')
    os.makedirs(output_dir, exist_ok=True)
    
    for img_path in glob.glob(os.path.join(output_dir, '*.jpg')) + glob.glob(os.path.join(output_dir, '*.png')):
        try:
            os.remove(img_path)
        except Exception:
            pass
    
    frame_count = 0
    fps_history = []
    
    try:
        camera.start()
        
        for frame in camera.get_frames(duration=duration):
            t0 = time.time()
            frame_count += 1
            results = identification_service.process_frame(frame)
            
            for person in results:
                if person.get('departed', False) and person['is_new'] and person['crop'] is not None:
                    filename = os.path.join(output_dir, f"person_id_{person['id']}.jpg")
                    cv2.imwrite(filename, person['crop'])
                    print(f"[{frame_count}] Saved person_id_{person['id']}.jpg (departed)")
                    continue
                
                if person['bbox'] is not None:
                    x, y, w, h = person['bbox']
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                cx, cy = person['centroid']
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                cv2.putText(frame, f"ID {person['id']}", (cx - 10, cy - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                if person['ready_to_save']:
                    cv2.putText(frame, "TRACKED", (cx - 10, cy + 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            fps_history.append(1.0 / (time.time() - t0 or 1e-6))
            if len(fps_history) > 30:
                fps_history.pop(0)
            avg_fps = sum(fps_history) / len(fps_history)
            
            detected_count = sum(1 for p in results if p['bbox'] is not None)
            
            cv2.putText(frame, f"FPS: {avg_fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            cv2.putText(frame, f"Detected: {detected_count}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Tracked: {len(results)}", (10, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Saved: {len(identification_service.saved_ids)}", (10, 130),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow('Human Detection and Tracking', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                print("\nExiting...")
                break
        
        print(f"\nProcessed {frame_count} frames")
        saved = save_remaining_crops(identification_service, output_dir)
        print(f"Total persons saved: {len(identification_service.saved_ids)}")
        
    except KeyboardInterrupt:
        print(f"\n\nInterrupted. Processed {frame_count} frames")
        save_remaining_crops(identification_service, output_dir)
        print(f"Total persons saved: {len(identification_service.saved_ids)}")
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
                       help='Frames per second (default: 15)')
    parser.add_argument('--duration', '-d', type=float, default=None,
                       help='Duration in seconds (default: indefinite)')
    parser.add_argument('--edge-margin', '-e', type=int, default=0,
                       help='Pixels from edge to filter detections (default: 0)')
    
    args = parser.parse_args()
    run_demo(camera_id=args.camera, fps=args.fps, duration=args.duration, 
             edge_margin=args.edge_margin)


if __name__ == '__main__':
    main()