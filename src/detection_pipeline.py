"""
Detection Pipeline

Handles the core detection and tracking logic:
- Human detection and tracking
- Face detection, matching, and PPE detection in background threads
- Identity voting system
- PPE compliance tracking
"""

import cv2
import time
import queue
import threading
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import os
from human_detection.human_identifier import HumanIdentificationService
from face_detection.face_detector import detect_face
from face_matching.face_matcher import FaceMatcher
from ppe_detection.ppe_detector import PPEDetector


class FaceWorker:
    """Runs face detection + matching + PPE detection in a single background thread.

    Main thread submits (person_id, crop_image) via ``submit()``.
    Results are stored in ``results`` dict keyed by person_id and can be
    polled with ``drain_results()``.
    """

    def __init__(self, face_matcher, ppe_detector: PPEDetector = None, num_workers: int = 2, maxsize: int = 8, verbose: bool = False, session_logger=None):
        self._matcher = face_matcher
        self._ppe_detector = ppe_detector
        self._verbose = verbose
        self._session_logger = session_logger
        # Bounded queue: if full, main thread skips instead of stacking up work
        self._q: queue.Queue = queue.Queue(maxsize=maxsize)
        self._results: dict = {}   # person_id -> (name, confidence, missing_items, all_present)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._pool = ThreadPoolExecutor(max_workers=num_workers, thread_name_prefix="FaceWorker")
        # Dispatch loop thread feeds jobs from queue into the thread pool
        self._dispatcher = threading.Thread(target=self._dispatch, daemon=True, name="FaceDispatcher")
        self._dispatcher.start()
        
        # Warm up models with dummy inference to eliminate startup lag
        self._warmup()

    def submit(self, person_id, crop, job_type='face+ppe') -> bool:
        """Queue a job. Returns False (dropped) if queue is full.
        
        Args:
            person_id: Person tracking ID
            crop: Image crop to process
            job_type: 'face+ppe' (both), 'face' (face only), or 'ppe' (ppe only)
        """
        try:
            self._q.put_nowait((person_id, crop, job_type))
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
    
    def _warmup(self):
        """Pre-initialize models by running inference on dummy image to eliminate startup lag."""
        try:
            # Create a small dummy image (480x360)
            dummy_crop = np.zeros((360, 480, 3), dtype=np.uint8)
            # Run one inference through the pipeline to warm up all models
            self._process(person_id=-1, crop=dummy_crop)
            # Drain the result so it doesn't pollute actual results
            with self._lock:
                self._results.clear()
        except Exception:
            # Warmup failures are non-fatal
            pass

    def _process(self, person_id, crop, job_type='face+ppe'):
        """Runs in a thread-pool worker: face detect + match and/or PPE detect.
        
        Args:
            person_id: Person tracking ID
            crop: Image crop to process
            job_type: 'face+ppe' (both), 'face' (face only), or 'ppe' (ppe only)
        """
        face_result = (None, None)
        ppe_result = ([], True)  # (missing_items, all_present)
        
        # Process face detection/matching if requested
        if job_type in ['face+ppe', 'face']:
            try:
                # Face detection using insightface RetinaFace (robust to various angles and scales)
                if self._session_logger:
                    self._session_logger(f"Person {person_id}: Face detection started")
                
                face_crop = detect_face(crop)
                if face_crop is not None:
                    if self._verbose:
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                        print(f"[{timestamp}] [Person {person_id}] Face detected, attempting face matching...")
                    if self._session_logger:
                        self._session_logger(f"Person {person_id}: Face detected in crop")
                    
                    # Get match result and all candidate scores for debugging
                    name, confidence = self._matcher.match_face_image(face_crop)
                    all_scores = self._matcher.get_all_match_scores(face_crop) if hasattr(self._matcher, 'get_all_match_scores') else {}
                    
                    if name and name.lower() != "unknown":
                        if self._verbose:
                            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                            scores_str = " | ".join([f"{n}: {c:.4f}" for n, c in all_scores.items()]) if all_scores else "N/A"
                            print(f"[{timestamp}] [Person {person_id}] Face matching SUCCESS: {name} (confidence: {confidence:.4f}) | All candidates: {scores_str}")
                        if self._session_logger:
                            scores_str = " | ".join([f"{n}: {c:.4f}" for n, c in all_scores.items()]) if all_scores else "N/A"
                            self._session_logger(f"Person {person_id}: Face matching SUCCEEDED - Identity: {name} (confidence: {confidence:.4f}) | All candidates: {scores_str}")
                        face_result = (name, confidence)
                    else:
                        if self._verbose:
                            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                            scores_str = " | ".join([f"{n}: {c:.4f}" for n, c in all_scores.items()]) if all_scores else "N/A"
                            print(f"[{timestamp}] [Person {person_id}] Face matching FAILED: No match found (highest confidence: {confidence:.4f}) | All candidates: {scores_str}")
                        if self._session_logger:
                            scores_str = " | ".join([f"{n}: {c:.4f}" for n, c in all_scores.items()]) if all_scores else "N/A"
                            self._session_logger(f"Person {person_id}: Face matching FAILED - No match found (highest confidence: {confidence:.4f}) | All candidates: {scores_str}")
                else:
                    if self._verbose:
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                        print(f"[{timestamp}] [Person {person_id}] Face detection FAILED: No face detected in crop")
                    if self._session_logger:
                        self._session_logger(f"Person {person_id}: Face detection FAILED - No face detected in crop")
            except Exception as e:
                if self._verbose:
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                    print(f"[{timestamp}] [Person {person_id}] Face matching FAILED: Exception - {str(e)}")
                if self._session_logger:
                    self._session_logger(f"Person {person_id}: Face matching FAILED - Exception: {str(e)}")
        
        # Process PPE detection if requested
        if job_type in ['face+ppe', 'ppe']:
            try:
                # PPE detection (on full resolution - preserves fine details for accuracy)
                if self._ppe_detector is not None:
                    if self._session_logger:
                        self._session_logger(f"Person {person_id}: PPE detection started")
                    ppe_detection_result = self._ppe_detector.detect(crop)
                    ppe_result = (ppe_detection_result['missing'], ppe_detection_result['all_present'])
                    
                    if self._session_logger:
                        missing_items = ppe_detection_result.get('missing', [])
                        all_present = ppe_detection_result.get('all_present', False)
                        if all_present:
                            self._session_logger(f"Person {person_id}: PPE detection SUCCEEDED - All required equipment present")
                        else:
                            self._session_logger(f"Person {person_id}: PPE detection SUCCEEDED - Missing equipment: {', '.join(missing_items)}")
            except Exception as e:
                if self._session_logger:
                    self._session_logger(f"Person {person_id}: PPE detection FAILED - Exception: {str(e)}")
        
        # Store result
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
            person_id, crop, job_type = item
            self._pool.submit(self._process, person_id, crop, job_type)
            self._q.task_done()


class PersonTracker:
    """Tracks person IDs with their identified names and PPE compliance using voting"""
    
    def __init__(self, face_confidence_threshold=0.17, min_votes=1, ppe_confidence_threshold=0.75, ppe_min_votes=1):
        self.person_votes = {}  # person_id -> {name: [confidences]}
        self.person_names = {}  # person_id -> final decided name
        self.face_attempts = {}  # person_id -> number of face matching attempts
        self.last_attempt_time = {}  # person_id -> timestamp of last face attempt
        # Cosine-similarity threshold (0–1, higher = better).  Only votes
        # at or above this value are counted (replaces old LBPH "below" logic).
        self.face_confidence_threshold = face_confidence_threshold
        self.min_votes = min_votes  # Minimum votes needed to confirm identity
        
        # PPE tracking
        self.ppe_votes = {}  # person_id -> {frozenset(detected_equipment): count}
        self.ppe_status = {}  # person_id -> {'missing': [items], 'compliant': bool, 'is_non_compliant': bool}
        self.ppe_confidence_threshold = ppe_confidence_threshold  # Confidence threshold for PPE votes
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
        if confidence is not None and confidence >= self.face_confidence_threshold:
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


class DetectionPipeline:
    """Orchestrates the detection pipeline: human detection, face detection, matching, and PPE detection"""
    
    SESSION_LOG_PATH = "src/Output/session_log.txt"
    
    def __init__(self, ppe_requirements=None, min_tracking_time=0.5, edge_margin=0, verbose=False, face_confidence_threshold=0.25, ppe_confidence_threshold=0.75):
        """
        Initialize the detection pipeline.
        
        Args:
            ppe_requirements: List of required PPE items (e.g., ['helmet', 'vest'])
            min_tracking_time: Minimum time before a person is tracked (seconds)
            edge_margin: Pixels from edge to filter detections
            verbose: Enable verbose logging
            face_confidence_threshold: Confidence threshold for face matching (0-1, default 0.25)
            ppe_confidence_threshold: Confidence threshold for PPE detection (0-1, default 0.75)
        """
        self.verbose = verbose
        self.ppe_requirements = ppe_requirements or []
        self._log_lock = threading.Lock()
        
        # Clear and initialize session log
        self._init_session_log()
        self._log(f"=== Detection Pipeline Session Started ===")
        
        # Initialize services
        self.identification_service = HumanIdentificationService(
            edge_margin=edge_margin,
            min_tracking_time=min_tracking_time,
            max_centroid_distance=150,
            conf_threshold=0.15
        )
        
        # Initialize face worker and PPE detector
        self.face_worker = None
        self.ppe_detector = None
        self._init_ppe_detector()
        self._init_face_worker()
        
        # Person tracking
        self.person_tracker = PersonTracker(
            face_confidence_threshold=face_confidence_threshold,
            min_votes=1,  # Edge optimization: confirm on first high-confidence match
            ppe_confidence_threshold=ppe_confidence_threshold,
            ppe_min_votes=1  # Single PPE detection is enough (only runs once per person)
        )
        self.in_flight = set()  # Tracks person IDs with pending face detection jobs
        self.ppe_processed = set()  # Tracks person IDs that have completed PPE detection
        self.face_cooldown = 0.05  # Reduced for faster edge detection
    
    def _init_ppe_detector(self):
        """Initialize PPE detector"""
        try:
            self.ppe_detector = PPEDetector(requirements=self.ppe_requirements)
            self._vprint("PPE detector initialized successfully")
            self._log("PPE detector initialized successfully")
        except Exception as e:
            self._vprint(f"Warning: PPE detector initialization failed: {e}")
            self._log(f"Warning: PPE detector initialization failed: {e}")
            self.ppe_detector = None
    
    def _init_face_worker(self):
        """Initialize face worker with insightface FaceMatcher (edge-optimized buffalo_sc)"""
        try:
            self._vprint("Initializing face detection models and warming up...")
            self._log("Initializing face detection models and warming up...")
            face_matcher = FaceMatcher()
            # 4 workers and maxsize=10 for optimal edge device performance
            self.face_worker = FaceWorker(face_matcher, ppe_detector=self.ppe_detector, num_workers=4, maxsize=10, verbose=self.verbose, session_logger=self._log)
            self._vprint("Face detection models initialized and warmed up successfully")
            self._log("Face detection models initialized and warmed up successfully")
        except Exception as e:
            self._vprint(f"Warning: Face matcher initialization failed: {e}")
            self._log(f"Warning: Face matcher initialization failed: {e}")
            self.face_worker = None
    
    def _init_session_log(self):
        """Clear and initialize the session log file"""
        try:
            log_dir = os.path.dirname(self.SESSION_LOG_PATH)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)
            # Clear the log file
            with open(self.SESSION_LOG_PATH, 'w') as f:
                f.write("")
        except Exception as e:
            print(f"Warning: Could not initialize session log: {e}")
    
    def _log(self, message: str):
        """Write a message to the session log file with timestamp"""
        try:
            with self._log_lock:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                with open(self.SESSION_LOG_PATH, 'a') as f:
                    f.write(f"[{timestamp}] {message}\n")
        except Exception as e:
            if self.verbose:
                print(f"Warning: Could not write to session log: {e}")
    
    def _vprint(self, *args, **kwargs):
        """Conditional verbose print"""
        if self.verbose:
            print(*args, **kwargs)
    
    def process_frame(self, frame):
        """
        Process a frame through the detection pipeline.
        
        Args:
            frame: Input video frame (numpy array)
            
        Returns:
            List of person detections with tracking info
        """
        # Process through human identification
        results = self.identification_service.process_frame(frame)
        
        # Log detected persons and track them
        for person in results:
            person_id = person['id']
            if person['bbox'] is not None:
                self._log(f"Person {person_id}: Detected in frame")
            if person['ready_to_save']:
                self._log(f"Person {person_id}: Tracking confirmed (ready_to_save=True)")
        
        # Drain completed face/PPE results
        if self.face_worker is not None:
            for pid, result in self.face_worker.drain_results().items():
                self.in_flight.discard(pid)
                name, confidence, missing_items, all_present = result
                
                # Process face match result
                if name is not None:
                    updated = self.person_tracker.add_match(pid, name, confidence)
                    if updated:
                        self._log(f"Person {pid}: Identity confirmed as {name} (confidence: {confidence:.4f})")
                
                # Process PPE detection result - mark as processed
                if missing_items is not None:
                    self.person_tracker.add_ppe_result(pid, missing_items, all_present)
                    self.ppe_processed.add(pid)  # Track that PPE has been processed
                
                self.person_tracker.increment_attempts(pid)
        
        # Submit new face detection jobs for unidentified persons
        for person in results:
            person_id = person['id']
            if (person['bbox'] is not None
                    and person['ready_to_save']
                    and self.face_worker is not None
                    and not self.person_tracker.is_identified(person_id)
                    and person_id not in self.in_flight
                    and self.person_tracker.can_attempt(person_id, self.face_cooldown)):
                crop = person.get('crop')
                if crop is not None:
                    # First submission: do both face and PPE; later: face-only retries
                    if person_id not in self.ppe_processed:
                        job_type = 'face+ppe'
                        self._log(f"Person {person_id}: Face+PPE detection job submitted")
                    else:
                        job_type = 'face'
                        self._log(f"Person {person_id}: Face detection retry submitted")
                    
                    submitted = self.face_worker.submit(person_id, crop.copy(), job_type=job_type)
                    if submitted:
                        self.in_flight.add(person_id)
        
        return results
    
    def get_remaining_crops(self):
        """Get all remaining tracked person crops"""
        return self.identification_service.get_all_tracked_crops()
    
    def cleanup_departed(self):
        """Clean up departed person IDs from memory"""
        return self.identification_service.cleanup_departed()
    
    def stop(self):
        """Stop the pipeline and cleanup resources"""
        if self.face_worker is not None:
            self.face_worker.stop()
