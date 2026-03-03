"""
Integrated Detection Demo

Combines human detection, face detection, and face matching to track and identify
persons in real-time video. Maintains persistent IDs and person names across frames.

Usage:
    python integrated_demo.py [--camera ID] [--fps FPS] [--duration SEC] [--edge-margin PX]

Controls:
    Press 'q' or ESC to quit

Features:
    - Human detection and tracking with persistent IDs
    - Face detection and matching for identification (runs in background thread)
    - Name persistence (stops matching once identified)
    - Automatic cleanup when persons depart
    - Output saved to human_detection/Output_images/ folder
"""

import cv2
import argparse
import time
import os
import glob
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from human_detection.temp_camera import TempCamera
from human_detection.human_identifier import HumanIdentificationService
from face_detection.face_detector import detect_face
from face_matching.face_matcher import FaceMatcher


def _ts() -> str:
    """Current wall-clock time as HH:MM:SS.mmm for log prefixes."""
    t = time.localtime()
    ms = int((time.time() % 1) * 1000)
    return f"{t.tm_hour:02d}:{t.tm_min:02d}:{t.tm_sec:02d}.{ms:03d}"


# ---------------------------------------------------------------------------
# Background face-recognition worker
# ---------------------------------------------------------------------------
# All SCRFD + ArcFace inference is dispatched here so the main display loop
# is never blocked by heavy ONNX calls.

class FaceWorker:
    """Runs face detection + matching in a single background thread.

    Main thread submits (person_id, crop_image) via ``submit()``.
    Results are stored in ``results`` dict keyed by person_id and can be
    polled with ``drain_results()``.
    """

    def __init__(self, face_matcher: FaceMatcher, num_workers: int = 2, maxsize: int = 8):
        self._matcher = face_matcher
        # Bounded queue: if full, main thread skips instead of stacking up work
        self._q: queue.Queue = queue.Queue(maxsize=maxsize)
        self._results: dict = {}   # person_id -> (name, confidence)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._pool = ThreadPoolExecutor(max_workers=num_workers, thread_name_prefix="FaceWorker")
        # Dispatch loop thread feeds jobs from queue into the thread pool
        self._dispatcher = threading.Thread(target=self._dispatch, daemon=True, name="FaceDispatcher")
        self._dispatcher.start()

    def submit(self, person_id, crop) -> bool:
        """Queue a job. Returns False (dropped) if queue is full."""
        try:
            self._q.put_nowait((person_id, crop))
            return True
        except queue.Full:
            return False

    def drain_results(self) -> dict:
        """Return and clear all completed results."""
        with self._lock:
            out = dict(self._results)
            self._results.clear()
        return out

    def stop(self):
        self._stop.set()
        try:
            self._q.put_nowait(None)  # unblock dispatcher
        except queue.Full:
            pass
        self._dispatcher.join(timeout=3)
        self._pool.shutdown(wait=False)

    def _process(self, person_id, crop):
        """Runs in a thread-pool worker: face detect + match."""
        try:
            face_crop = detect_face(crop)
            if face_crop is not None:
                name, confidence = self._matcher.match_face_image(face_crop)
                result = (name, confidence)
            else:
                result = (None, None)
        except Exception:
            result = (None, None)
        with self._lock:
            self._results[person_id] = result

    def _dispatch(self):
        """Pulls jobs from the queue and submits them to the thread pool."""
        while not self._stop.is_set():
            try:
                item = self._q.get(timeout=0.5)
            except queue.Empty:
                continue
            if item is None:
                break
            person_id, crop = item
            self._pool.submit(self._process, person_id, crop)
            self._q.task_done()


class PersonTracker:
    """Tracks person IDs with their identified names using voting"""
    
    def __init__(self, confidence_threshold=0.25, min_votes=2):
        self.person_votes = {}  # person_id -> {name: [confidences]}
        self.person_names = {}  # person_id -> final decided name
        self.face_attempts = {}  # person_id -> number of face matching attempts
        self.last_attempt_time = {}  # person_id -> timestamp of last face attempt
        # Cosine-similarity threshold (0–1, higher = better).  Only votes
        # at or above this value are counted (replaces old LBPH "below" logic).
        self.confidence_threshold = confidence_threshold
        self.min_votes = min_votes  # Minimum votes needed to confirm identity
    
    def is_identified(self, person_id):
        """Check if person has been identified with a valid name (not Unknown)"""
        name = self.person_names.get(person_id, "Unknown")
        return name and name.lower() != "unknown"
    
    def add_match(self, person_id, name, confidence):
        """Add a face match result and update identity based on voting"""
        if person_id not in self.person_votes:
            self.person_votes[person_id] = {}
        
        # Only count high-quality matches (cosine similarity above threshold)
        if confidence is not None and confidence >= self.confidence_threshold:
            if name not in self.person_votes[person_id]:
                self.person_votes[person_id][name] = []
            self.person_votes[person_id][name].append(confidence)
        
        # Recalculate best identity based on votes
        votes = self.person_votes[person_id]
        if not votes:
            return False
        
        # Calculate average confidence and vote count for each name
        name_scores = {}
        for candidate_name, confidences in votes.items():
            avg_conf = sum(confidences) / len(confidences)
            vote_count = len(confidences)
            # Score: most votes first, then highest avg cosine similarity as tiebreaker
            name_scores[candidate_name] = (vote_count, avg_conf)
        
        # Find best candidate (most votes, then best average confidence)
        best_name = max(name_scores.keys(), key=lambda n: name_scores[n])
        best_vote_count, _ = name_scores[best_name]
        
        # Only confirm identity if we have enough votes
        if best_vote_count >= self.min_votes:
            old_name = self.person_names.get(person_id, "Unknown")
            if old_name != best_name:
                self.person_names[person_id] = best_name
                return True  # Name was updated
        
        return False  # No update
    
    def get_name(self, person_id):
        """Get name for a person ID"""
        return self.person_names.get(person_id, "Unknown")
    
    def increment_attempts(self, person_id):
        """Increment face matching attempts for a person and record timestamp"""
        self.face_attempts[person_id] = self.face_attempts.get(person_id, 0) + 1
        self.last_attempt_time[person_id] = time.time()
        return self.face_attempts[person_id]

    def can_attempt(self, person_id, cooldown_seconds: float = 1.5) -> bool:
        """Return True if enough time has passed since the last attempt."""
        last = self.last_attempt_time.get(person_id)
        if last is None:
            return True
        return (time.time() - last) >= cooldown_seconds
    
    def get_vote_stats(self, person_id):
        """Get voting statistics for a person"""
        if person_id not in self.person_votes:
            return {}
        votes = self.person_votes[person_id]
        stats = {}
        for name, confidences in votes.items():
            stats[name] = {
                'count': len(confidences),
                'avg_confidence': sum(confidences) / len(confidences),
                'best_confidence': max(confidences)  # higher cosine sim = better
            }
        return stats
    
    def remove_person(self, person_id):
        """Remove person from tracking"""
        if person_id in self.person_names:
            del self.person_names[person_id]
        if person_id in self.person_votes:
            del self.person_votes[person_id]
        if person_id in self.face_attempts:
            del self.face_attempts[person_id]
        if person_id in self.last_attempt_time:
            del self.last_attempt_time[person_id]


def save_person_info(person_id, name, crop_image, output_dir):
    """
    Save departed person's information
    
    Args:
        person_id: Unique tracking ID
        name: Identified name or "Unknown"
        crop_image: Cropped person image
        output_dir: Directory to save to
    
    TODO: Implement actual storage logic (database, CSV, etc.)
    """
    filename = os.path.join(output_dir, f"person_id_{person_id}_{name}.jpg")
    cv2.imwrite(filename, crop_image)
    print(f"[SAVE] person_id_{person_id} - Name: {name}")


def run_integrated_demo(camera_id=0, fps=60, duration=None, edge_margin=0):
    """Run integrated human detection, face detection, and face matching demo"""
    print(f"Starting integrated detection service (Camera {camera_id}, {fps} FPS, Edge margin {edge_margin}px)")
    print("Initializing face matcher...")
    
    # Initialize services
    camera = TempCamera(camera_id=camera_id, fps=fps)
    identification_service = HumanIdentificationService(
        edge_margin=edge_margin,
        min_tracking_time=2.0,
        max_centroid_distance=150,
        conf_threshold=0.15
    )
    
    face_worker = None
    try:
        face_matcher = FaceMatcher()
        face_worker = FaceWorker(face_matcher)
        print("Face matcher initialized successfully")
    except Exception as e:
        print(f"Warning: Face matcher initialization failed: {e}")
        print("Continuing without face matching...")
        face_matcher = None
    
    person_tracker = PersonTracker()
    
    # Track which persons have a face job currently in-flight so we don't
    # queue duplicate jobs before the result comes back.
    in_flight: set = set()

    # Output to human_detection folder
    output_dir = os.path.join(os.path.dirname(__file__), 'human_detection', 'Output_images')
    os.makedirs(output_dir, exist_ok=True)
    
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

    # How long to wait between face-attempt submissions per person (seconds).
    FACE_COOLDOWN = 0.01
    
    print("Press 'q' or ESC to quit")
    
    try:
        camera.start()
        
        for frame in camera.get_frames(duration=duration):
            t0 = time.time()
            frame_count += 1

            # ------------------------------------------------------------------
            # 1. Drain completed face-recognition results from the worker thread
            # ------------------------------------------------------------------
            if face_worker is not None:
                for pid, (name, confidence) in face_worker.drain_results().items():
                    in_flight.discard(pid)
                    if name is not None:
                        was_updated = person_tracker.add_match(pid, name, confidence)
                        if was_updated:
                            final_name = person_tracker.get_name(pid)
                            stats = person_tracker.get_vote_stats(pid)
                            print(f"[{frame_count}|{_ts()}] ✓ CONFIRMED person {pid} as {final_name}")
                            print(f"[{frame_count}|{_ts()}]   Vote stats: {stats}")
                    person_tracker.increment_attempts(pid)
            
            # Process frame through human identification
            results = identification_service.process_frame(frame)
            
            # Track if we had any departures in this frame
            had_departures = False
            
            for person in results:
                person_id = person['id']
                
                # Handle departed persons
                if person.get('departed', False):
                    if person['is_new'] and person['crop'] is not None:
                        # Save person information
                        name = person_tracker.get_name(person_id)
                        save_person_info(person_id, name, person['crop'], output_dir)
                        
                        # Remove from our tracking
                        person_tracker.remove_person(person_id)
                        in_flight.discard(person_id)
                        
                        print(f"[{frame_count}|{_ts()}] Person {person_id} departed - Name: {name}")
                        had_departures = True
                    continue
                
                # Handle currently tracked persons
                if person['bbox'] is not None:
                    # Draw bounding box
                    x, y, w, h = person['bbox']
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    
                    # ----------------------------------------------------------
                    # 2. Submit face job to background worker (non-blocking)
                    #    Conditions: ready_to_save, not already identified,
                    #    cooldown elapsed, not already in-flight, worker available
                    # ----------------------------------------------------------
                    if (person['ready_to_save']
                            and face_worker is not None
                            and not person_tracker.is_identified(person_id)
                            and person_id not in in_flight
                            and person_tracker.can_attempt(person_id, FACE_COOLDOWN)):
                        crop = person.get('crop')
                        if crop is not None:
                            submitted = face_worker.submit(person_id, crop.copy())
                            if submitted:
                                in_flight.add(person_id)
                                attempts = person_tracker.face_attempts.get(person_id, 0) + 1
                                print(f"[{frame_count}|{_ts()}] Queued face job for person {person_id} (attempt {attempts})")
                
                # Draw centroid
                cx, cy = person['centroid']
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                
                # Draw ID label
                cv2.putText(frame, f"ID {person_id}", (cx - 10, cy - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                # Draw name label
                name = person_tracker.get_name(person_id)
                cv2.putText(frame, name, (cx - 10, cy + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
                # Draw "TRACKED" status if ready to save
                if person['ready_to_save']:
                    cv2.putText(frame, "TRACKED", (cx - 10, cy + 45),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Clean up departed IDs immediately after processing
            if had_departures:
                cleared = identification_service.cleanup_departed()
                if cleared > 0:
                    print(f"[{frame_count}|{_ts()}] Cleaned up {cleared} departed IDs from memory")
            
            # Calculate FPS
            elapsed = time.time() - t0 or 1e-6
            fps_history.append(1.0 / elapsed)
            if len(fps_history) > 30:
                fps_history.pop(0)
            avg_fps = sum(fps_history) / len(fps_history)
            
            # Count statistics
            detected_count = sum(1 for p in results if p['bbox'] is not None)
            identified_count = sum(1 for p in results if person_tracker.is_identified(p['id']))
            
            # Draw statistics overlay
            cv2.putText(frame, f"FPS: {avg_fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            cv2.putText(frame, f"Detected: {detected_count}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Tracked: {len(results)}", (10, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Identified: {identified_count}", (10, 130),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Saved: {len(identification_service.saved_ids)}", (10, 160),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Display frame
            cv2.imshow('Integrated Human Detection & Identification', frame)
            
            # Check for quit
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                print("\nExiting...")
                break
            
            # Periodic memory cleanup
            if time.time() - last_cleanup > cleanup_interval:
                cleared = identification_service.cleanup_departed()
                print(f"[CLEANUP] Cleared {cleared} departed person IDs from memory")
                last_cleanup = time.time()
        
        # Handle remaining tracked persons on exit
        print(f"\nProcessed {frame_count} frames")
        remaining_crops = identification_service.get_all_tracked_crops()
        for person_data in remaining_crops:
            person_id = person_data['id']
            crop = person_data['crop']
            
            name = person_tracker.get_name(person_id)
            save_person_info(person_id, name, crop, output_dir)
            
            print(f"[EXIT] Saved person_id_{person_id} - Name: {name} (quality {person_data['quality']:.2f})")
        
        print(f"Total persons saved: {len(identification_service.saved_ids)}")
        
    except KeyboardInterrupt:
        print(f"\n\nInterrupted. Processed {frame_count} frames")
        # Save remaining on interrupt
        remaining_crops = identification_service.get_all_tracked_crops()
        for person_data in remaining_crops:
            person_id = person_data['id']
            name = person_tracker.get_name(person_id)
            save_person_info(person_id, name, person_data['crop'], output_dir)
        print(f"Total persons saved: {len(identification_service.saved_ids)}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if face_worker is not None:
            face_worker.stop()
        camera.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='Integrated Human Detection and Identification Demo')
    parser.add_argument('--camera', '-c', type=int, default=0,
                       help='Camera device ID (default: 0)')
    parser.add_argument('--fps', '-f', type=int, default=60,
                       help='Frames per second (default: 60)')
    parser.add_argument('--duration', '-d', type=float, default=None,
                       help='Duration in seconds (default: indefinite)')
    parser.add_argument('--edge-margin', '-e', type=int, default=0,
                       help='Pixels from edge to filter detections (default: 0)')
    
    args = parser.parse_args()
    run_integrated_demo(camera_id=args.camera, fps=args.fps, duration=args.duration, 
                       edge_margin=args.edge_margin)


if __name__ == '__main__':
    main()
