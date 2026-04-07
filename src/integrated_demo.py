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
from ppe_detection.ppe_detector import PPEDetector

MIN_TIME_BEFORE_TRACKING = 0.5
PPE_REQUIREMENTS = ['vest']

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
    """Runs face detection + matching + PPE detection in a single background thread.

    Main thread submits (person_id, crop_image) via ``submit()``.
    Results are stored in ``results`` dict keyed by person_id and can be
    polled with ``drain_results()``.
    """

    def __init__(self, face_matcher: FaceMatcher, ppe_detector: PPEDetector = None, num_workers: int = 2, maxsize: int = 8):
        self._matcher = face_matcher
        self._ppe_detector = ppe_detector
        # Bounded queue: if full, main thread skips instead of stacking up work
        self._q: queue.Queue = queue.Queue(maxsize=maxsize)
        self._results: dict = {}   # person_id -> (name, confidence, missing_items, all_present)
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
        """Runs in a thread-pool worker: face detect + match + PPE detect."""
        face_result = (None, None)
        ppe_result = ([], True)  # (missing_items, all_present)
        
        # Downsample crop 2x for faster inference (improves speed ~3-4x)
        h, w = crop.shape[:2]
        small_crop = cv2.resize(crop, (w // 2, h // 2)) if h > 50 and w > 50 else crop
        
        try:
            # Face detection and matching
            face_crop = detect_face(small_crop)
            if face_crop is not None:
                name, confidence = self._matcher.match_face_image(face_crop)
                face_result = (name, confidence)
        except Exception:
            pass
        
        try:
            # PPE detection
            if self._ppe_detector is not None:
                ppe_detection_result = self._ppe_detector.detect(small_crop)
                ppe_result = (ppe_detection_result['missing'], ppe_detection_result['all_present'])
        except Exception:
            pass
        
        result = (face_result[0], face_result[1], ppe_result[0], ppe_result[1])
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
    """Tracks person IDs with their identified names and PPE compliance using voting"""
    
    def __init__(self, confidence_threshold=0.25, min_votes=2, ppe_min_votes=2):
        self.person_votes = {}  # person_id -> {name: [confidences]}
        self.person_names = {}  # person_id -> final decided name
        self.face_attempts = {}  # person_id -> number of face matching attempts
        self.last_attempt_time = {}  # person_id -> timestamp of last face attempt
        # Cosine-similarity threshold (0–1, higher = better).  Only votes
        # at or above this value are counted (replaces old LBPH "below" logic).
        self.confidence_threshold = confidence_threshold
        self.min_votes = min_votes  # Minimum votes needed to confirm identity
        
        # PPE tracking
        self.ppe_votes = {}  # person_id -> {frozenset(detected_equipment): count}
        self.ppe_status = {}  # person_id -> {'missing': [items], 'compliant': bool, 'is_non_compliant': bool}
        self.ppe_min_votes = ppe_min_votes  # Minimum votes for PPE determination
    
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
    
    def add_ppe_result(self, person_id, missing_items, all_present):
        """Add a PPE detection result and update compliance based on voting.
        
        Args:
            person_id: Person tracking ID
            missing_items: List of missing PPE items (e.g., ['helmet', 'vest'])
            all_present: Boolean - True if all required PPE is present
        """
        if person_id not in self.ppe_votes:
            self.ppe_votes[person_id] = {}
        if person_id not in self.ppe_status:
            self.ppe_status[person_id] = {'missing': [], 'compliant': True, 'is_non_compliant': False}
        
        # Store vote as frozenset of missing items for counting
        missing_key = frozenset(missing_items)
        self.ppe_votes[person_id][missing_key] = self.ppe_votes[person_id].get(missing_key, 0) + 1
        
        # Recalculate best PPE status based on votes
        votes = self.ppe_votes[person_id]
        if not votes:
            return
        
        # Find the missing items set with the most votes
        best_missing_key = max(votes.keys(), key=lambda k: votes[k])
        best_vote_count = votes[best_missing_key]
        
        # Only update if we have enough votes
        if best_vote_count >= self.ppe_min_votes:
            best_missing = sorted(list(best_missing_key))
            is_now_compliant = (len(best_missing) == 0)
            
            # Once marked non-compliant, stay non-compliant
            if not is_now_compliant:
                self.ppe_status[person_id]['is_non_compliant'] = True
            
            self.ppe_status[person_id]['missing'] = best_missing
            self.ppe_status[person_id]['compliant'] = is_now_compliant
    
    def get_ppe_status(self, person_id):
        """Get PPE compliance status for a person.
        
        Returns:
            Dict with keys: 'missing' (list), 'compliant' (bool), 'is_non_compliant' (bool)
        """
        return self.ppe_status.get(person_id, {'missing': [], 'compliant': True, 'is_non_compliant': False})
    
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
        if person_id in self.ppe_votes:
            del self.ppe_votes[person_id]
        if person_id in self.ppe_status:
            del self.ppe_status[person_id]


def save_person_info(person_id, name, crop_image, output_dir, ppe_status=None):
    """
    Save departed person's information
    
    Args:
        person_id: Unique tracking ID
        name: Identified name or "Unknown"
        crop_image: Cropped person image
        output_dir: Directory to save to
        ppe_status: Dict with PPE compliance info {'missing': [...], 'compliant': bool, 'is_non_compliant': bool}
    
    TODO: Implement actual storage logic (database, CSV, etc.)
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
    print(f"[SAVE] person_id_{person_id} - Name: {name} - PPE Status: {status_str}")


def run_integrated_demo(camera_id=0, fps=60, duration=None, edge_margin=0, verbose=False):
    """Run integrated human detection, face detection, face matching, and PPE detection demo"""
    def vprint(*args, **kwargs):
        """Conditional print based on verbose flag"""
        if verbose:
            print(*args, **kwargs)
    
    vprint(f"Starting integrated detection service (Camera {camera_id}, {fps} FPS, Edge margin {edge_margin}px)")
    vprint("Initializing face matcher and PPE detector...")
    
    # Initialize services
    camera = TempCamera(camera_id=camera_id, fps=fps)
    identification_service = HumanIdentificationService(
        edge_margin=edge_margin,
        min_tracking_time=MIN_TIME_BEFORE_TRACKING,
        max_centroid_distance=150,
        conf_threshold=0.15
    )
    
    face_worker = None
    ppe_detector = None
    try:
        ppe_detector = PPEDetector(requirements=PPE_REQUIREMENTS)
        vprint("PPE detector initialized successfully")
    except Exception as e:
        vprint(f"Warning: PPE detector initialization failed: {e}")
        vprint("Continuing without PPE detection...")
        ppe_detector = None
    
    try:
        face_matcher = FaceMatcher()
        face_worker = FaceWorker(face_matcher, ppe_detector=ppe_detector, num_workers=4, maxsize=5)
        vprint("Face matcher initialized successfully")
    except Exception as e:
        vprint(f"Warning: Face matcher initialization failed: {e}")
        vprint("Continuing without face matching...")
        face_matcher = None
    
    person_tracker = PersonTracker(confidence_threshold=0.25, min_votes=1, ppe_min_votes=1)
    
    # Track which persons have a face job currently in-flight so we don't
    # queue duplicate jobs before the result comes back.
    in_flight: set = set()

    # Output to src/Output folder
    output_dir = os.path.join(os.path.dirname(__file__), 'Output')
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
    FACE_COOLDOWN = 0.75
    
    vprint("Press 'q' or ESC to quit")
    
    try:
        camera.start()
        
        for frame in camera.get_frames(duration=duration):
            t0 = time.time()
            frame_count += 1

            # ------------------------------------------------------------------
            # 1. Drain completed face-recognition and PPE detection results
            # ------------------------------------------------------------------
            if face_worker is not None:
                for pid, result in face_worker.drain_results().items():
                    in_flight.discard(pid)
                    name, confidence, missing_items, all_present = result
                    
                    # Process face match result
                    if name is not None:
                        was_updated = person_tracker.add_match(pid, name, confidence)
                        if was_updated:
                            final_name = person_tracker.get_name(pid)
                            stats = person_tracker.get_vote_stats(pid)
                            vprint(f"[{frame_count}|{_ts()}] ✓ CONFIRMED person {pid} as {final_name}")
                            vprint(f"[{frame_count}|{_ts()}]   Vote stats: {stats}")
                    
                    # Process PPE detection result
                    if missing_items is not None:
                        person_tracker.add_ppe_result(pid, missing_items, all_present)
                        ppe_status = person_tracker.get_ppe_status(pid)
                        if ppe_status['is_non_compliant']:
                            vprint(f"[{frame_count}|{_ts()}] ⚠ PPE VIOLATION - person {pid} missing: {', '.join(ppe_status['missing'])}")
                    
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
                        ppe_status = person_tracker.get_ppe_status(person_id)
                        save_person_info(person_id, name, person['crop'], output_dir, ppe_status=ppe_status)
                        
                        # Remove from our tracking
                        person_tracker.remove_person(person_id)
                        in_flight.discard(person_id)
                        
                        vprint(f"[{frame_count}|{_ts()}] Person {person_id} departed - Name: {name}")
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
                                vprint(f"[{frame_count}|{_ts()}] Queued face job for person {person_id} (attempt {attempts})")
                
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
                
                # Draw PPE compliance status
                ppe_status = person_tracker.get_ppe_status(person_id)
                if ppe_status['is_non_compliant']:
                    # Non-compliant: red background with missing items
                    missing_text = f"MISSING: {','.join(ppe_status['missing']).upper()}"
                    color = (0, 0, 255)  # Red for non-compliant
                else:
                    # Compliant: green
                    missing_text = "COMPLIANT"
                    color = (0, 255, 0)  # Green for compliant
                cv2.putText(frame, missing_text, (cx - 10, cy + 45),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Draw "TRACKED" status if ready to save
                if person['ready_to_save']:
                    cv2.putText(frame, "TRACKED", (cx - 10, cy + 70),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Clean up departed IDs immediately after processing
            if had_departures:
                cleared = identification_service.cleanup_departed()
                if cleared > 0:
                    vprint(f"[{frame_count}|{_ts()}] Cleaned up {cleared} departed IDs from memory")
            
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
                vprint(f"[CLEANUP] Cleared {cleared} departed person IDs from memory")
                last_cleanup = time.time()
        
        # Handle remaining tracked persons on exit
        vprint(f"\nProcessed {frame_count} frames")
        remaining_crops = identification_service.get_all_tracked_crops()
        for person_data in remaining_crops:
            person_id = person_data['id']
            crop = person_data['crop']
            
            name = person_tracker.get_name(person_id)
            ppe_status = person_tracker.get_ppe_status(person_id)
            save_person_info(person_id, name, crop, output_dir, ppe_status=ppe_status)
            
            vprint(f"[EXIT] Saved person_id_{person_id} - Name: {name} (quality {person_data['quality']:.2f})")
        
        vprint(f"Total persons saved: {len(identification_service.saved_ids)}")
        
    except KeyboardInterrupt:
        vprint(f"\n\nInterrupted. Processed {frame_count} frames")
        # Save remaining on interrupt
        remaining_crops = identification_service.get_all_tracked_crops()
        for person_data in remaining_crops:
            person_id = person_data['id']
            name = person_tracker.get_name(person_id)
            ppe_status = person_tracker.get_ppe_status(person_id)
            save_person_info(person_id, name, person_data['crop'], output_dir, ppe_status=ppe_status)
        vprint(f"Total persons saved: {len(identification_service.saved_ids)}")
    except Exception as e:
        vprint(f"Error: {e}")
        import traceback
        if verbose:
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
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output (default: False)')
    
    args = parser.parse_args()
    run_integrated_demo(camera_id=args.camera, fps=args.fps, duration=args.duration, 
                       edge_margin=args.edge_margin, verbose=args.verbose)


if __name__ == '__main__':
    main()
