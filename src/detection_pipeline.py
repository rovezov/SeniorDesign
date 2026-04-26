"""
Detection Pipeline

Handles the core detection and tracking logic:
- Human detection and tracking
- Face detection, matching, and PPE detection in background threads
- Identity voting system
- PPE compliance tracking (Real-time and Cumulative)
"""

import cv2
import time
import queue
import threading
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import os
import logging
from human_detection.human_identifier import HumanIdentificationService
from face_detection.face_detector import detect_face
from face_matching.face_matcher import FaceMatcher
from ppe_detection.ppe_detector import PPEDetector


class _DryRunHumanIdentificationService:
    """Human service shim used when human model is silenced."""

    def __init__(self):
        self.saved_ids = set()

    def process_frame(self, frame):
        return []

    def get_all_tracked_crops(self):
        return []

    def cleanup_departed(self):
        return 0


class FaceWorker:
    """Runs face detection + matching + PPE detection in a single background thread.

    Main thread submits (person_id, crop_image) via ``submit()``.
    Results are stored in ``results`` dict keyed by person_id and can be
    polled with ``drain_results()``.
    """

    def __init__(self, face_matcher, ppe_detector: PPEDetector = None, num_workers: int = 2, maxsize: int = 8, verbose: bool = False, session_logger=None, enable_face_matching: bool = True, silenced_models=None):
        self._matcher = face_matcher
        self._ppe_detector = ppe_detector
        self._verbose = verbose
        self._session_logger = session_logger
        self._enable_face_matching = bool(enable_face_matching)
        self._silenced_models = set(silenced_models or [])
        self._silence_face = 'face' in self._silenced_models
        self._silence_matching = 'matching' in self._silenced_models
        self._silence_ppe = 'ppe' in self._silenced_models
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
            warmup_job = 'face+ppe' if (self._enable_face_matching and self._matcher is not None) else 'ppe'
            self._process(person_id=-1, crop=dummy_crop, job_type=warmup_job)
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
        if job_type in ['face+ppe', 'face'] and self._silence_face:
            # Dry run: skip face detector + matcher, return a valid unknown result.
            face_result = ("Unknown", 0.0)
            if self._session_logger:
                self._session_logger(f"Person {person_id}: Face model SILENCED (dry run)")
        elif job_type in ['face+ppe', 'face'] and self._enable_face_matching:
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
                    
                    # Get match result. Candidate scores are only computed in verbose/debug mode.
                    if self._silence_matching or self._matcher is None:
                        name, confidence = "Unknown", 0.0
                        all_scores = {}
                        if self._session_logger:
                            self._session_logger(f"Person {person_id}: Face matching model SILENCED (dry run)")
                    else:
                        name, confidence = self._matcher.match_face_image(face_crop)
                        all_scores = {}
                        if self._verbose and hasattr(self._matcher, 'get_all_match_scores'):
                            all_scores = self._matcher.get_all_match_scores(face_crop)
                    
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
                if self._silence_ppe:
                    ppe_result = ([], True)
                    if self._session_logger:
                        self._session_logger(f"Person {person_id}: PPE model SILENCED (dry run)")
                elif self._ppe_detector is not None:
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
    
    def __init__(self, face_confidence_threshold=0.4, min_votes=1, ppe_confidence_threshold=0.75, ppe_min_votes=1):
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
        self.ppe_status = {}  # person_id -> dict with current and cumulative stats
        self.last_ppe_time = {} # person_id -> timestamp of last PPE attempt
        self.ppe_confidence_threshold = ppe_confidence_threshold  
        self.ppe_min_votes = ppe_min_votes

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

    def can_attempt(self, person_id, initial_cooldown: float = 1.0, slow_cooldown: float = 3.0, fast_attempts: int = 5) -> bool:
        """Return True if enough time has passed since the last attempt.

        Uses a faster retry cadence for the first few attempts, then slows down
        to reduce CPU load when a person remains unidentified.
        """
        last = self.last_attempt_time.get(person_id)
        if last is None:
            return True

        attempts = self.face_attempts.get(person_id, 0)
        cooldown_seconds = initial_cooldown if attempts < fast_attempts else slow_cooldown
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

    def can_attempt_ppe(self, person_id, cooldown: float = 1.0) -> bool:
        """Check if enough time has passed to sample PPE again (default 1 sec)."""
        last = self.last_ppe_time.get(person_id)
        if last is None:
            return True
        return (time.time() - last) >= cooldown

    def increment_ppe_attempts(self, person_id):
        """Record the timestamp of a completed PPE check."""
        self.last_ppe_time[person_id] = time.time()

    def add_ppe_result(self, person_id, missing_items, all_present):
        """Add a PPE detection result and calculate both real-time and cumulative compliance."""
        if person_id not in self.ppe_votes:
            self.ppe_votes[person_id] = {}
        if person_id not in self.ppe_status:
            self.ppe_status[person_id] = {
                'current_missing': [],
                'current_compliant': True,
                'cumulative_missing': [],
                'cumulative_compliant': True,
                'total_checks': 0,
                'compliant_checks': 0
            }
        
        status = self.ppe_status[person_id]
        
        # 1. Update REAL-TIME instantaneous status for the overlay display
        status['current_missing'] = missing_items
        status['current_compliant'] = all_present
        
        # 2. Update CUMULATIVE status for the final saved entry
        status['total_checks'] += 1
        if all_present:
            status['compliant_checks'] += 1
            
        # Store vote as frozenset of missing items for counting specific gear
        missing_key = frozenset(missing_items)
        self.ppe_votes[person_id][missing_key] = self.ppe_votes[person_id].get(missing_key, 0) + 1
        
        # Calculate majority compliance (they must be compliant > 50% of the time tracked)
        compliance_ratio = status['compliant_checks'] / status['total_checks']
        status['cumulative_compliant'] = compliance_ratio >= 0.5
        
        # If cumulatively non-compliant, find the most consistently missing items
        if not status['cumulative_compliant']:
            # Filter out compliant votes to find what is ACTUALLY missing most often
            non_compliant_votes = {k: v for k, v in self.ppe_votes[person_id].items() if len(k) > 0}
            if non_compliant_votes:
                best_missing_key = max(non_compliant_votes.keys(), key=lambda k: non_compliant_votes[k])
                status['cumulative_missing'] = sorted(list(best_missing_key))
            else:
                 status['cumulative_missing'] = status['current_missing']
        else:
            status['cumulative_missing'] = []

    def get_ppe_status(self, person_id):
        """Get the dual PPE status for a person."""
        return self.ppe_status.get(person_id, {
            'current_missing': [], 'current_compliant': True,
            'cumulative_missing': [], 'cumulative_compliant': True
        })

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
        if person_id in self.last_ppe_time:
            del self.last_ppe_time[person_id]


class DetectionPipeline:
    """Orchestrates the detection pipeline: human detection, face detection, matching, and PPE detection"""
    
    def __init__(self, ppe_requirements=None, min_tracking_time=0.5, edge_margin=0, verbose=False, face_confidence_threshold=0.4, ppe_confidence_threshold=0.75, face_workers=1, enable_face_matching=True, silenced_models=None):
        """
        Initialize the detection pipeline.
        
        Args:
            ppe_requirements: List of required PPE items (e.g., ['helmet', 'vest'])
            min_tracking_time: Minimum time before a person is tracked (seconds)
            edge_margin: Pixels from edge to filter detections
            verbose: Enable verbose logging
            face_confidence_threshold: Confidence threshold for face matching (0-1, default 0.4)
            ppe_confidence_threshold: Confidence threshold for PPE detection (0-1, default 0.75)
        """
        self.verbose = verbose
        self.ppe_requirements = ppe_requirements or []
        self.face_workers = max(1, int(face_workers))
        self.enable_face_matching = bool(enable_face_matching)
        self.silenced_models = set(silenced_models or [])
        
        # Clear and initialize session log
        self._init_session_log()
        self._log(f"=== Detection Pipeline Session Started ===")
        if self.silenced_models:
            self._log(f"Silenced models (dry run): {', '.join(sorted(self.silenced_models))}")
        
        # Initialize services
        if 'human' in self.silenced_models:
            self._log("Human model is SILENCED (dry run mode)")
            self.identification_service = _DryRunHumanIdentificationService()
        else:
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
            ppe_min_votes=1  # Continuous sampling logic handled via cooldown now
        )
        self.in_flight = {}  # Tracks person IDs with pending face detection jobs
        self.face_cooldown = 1.0  # Initial retry interval for face matching
        self.face_retry_limit = 5
        self.face_retry_slow_cooldown = 3.0
        
        # Thermal throttling: pause face detection if CPU overheats
        self._thermal_throttled = False
        self._temp_override_threshold = 75.0  # Celsius - pause submissions above this
        self._temp_resume_threshold = 60.0   # Celsius - resume below this
        self._face_matching_disabled = False  # Permanently disable if CPU reaches critical temp
        self._critical_temp_threshold = 80.0  # Celsius - hard disable face matching above this
        self._last_thermal_check = 0.0
        self._thermal_check_interval = 1.0
    
    def _init_ppe_detector(self):
        """Initialize PPE detector"""
        if 'ppe' in self.silenced_models:
            self._log("PPE model is SILENCED (dry run mode)")
            self.ppe_detector = None
            return
        try:
            self.ppe_detector = PPEDetector(requirements=self.ppe_requirements)
            self._vprint("PPE detector initialized successfully")
            self._log("PPE detector initialized successfully")
        except Exception as e:
            self._vprint(f"Warning: PPE detector initialization failed: {e}")
            self._log(f"Warning: PPE detector initialization failed: {e}")
            self.ppe_detector = None
    
    def _init_face_worker(self):
        """Initialize face worker with a powerful ResNet-50 face matcher."""
        try:
            self._vprint("Initializing face/PPE worker and warming up...")
            self._log("Initializing face/PPE worker and warming up...")
            # Uses ResNet-50 (buffalo_l) via InsightFace with landmark alignment for best accuracy.
            disable_matching = ('matching' in self.silenced_models) or ('face' in self.silenced_models)
            face_matcher = FaceMatcher() if (self.enable_face_matching and not disable_matching) else None
            if not self.enable_face_matching:
                self._log("Face matching is DISABLED for this run; identities will remain Unknown")
            elif 'face' in self.silenced_models:
                self._log("Face model is SILENCED (dry run mode)")
            elif 'matching' in self.silenced_models:
                self._log("Face matching model is SILENCED (dry run mode)")
            else:
                backend_name = getattr(face_matcher, "_mode", "unknown") if face_matcher is not None else "unknown"
                if backend_name == "insightface":
                    self._log("Face matching is ENABLED with ResNet-50 ArcFace (buffalo_l) + landmark alignment")
                elif backend_name == "onnx":
                    self._log("Face matching is ENABLED with ONNX ArcFace model")
                else:
                    self._log("Face matching is ENABLED with unknown backend selection")
            # Keep worker count conservative on embedded systems to reduce CPU pressure.
            self.face_worker = FaceWorker(
                face_matcher,
                ppe_detector=self.ppe_detector,
                num_workers=self.face_workers,
                maxsize=6,
                verbose=self.verbose,
                session_logger=self._log,
                enable_face_matching=self.enable_face_matching,
                silenced_models=self.silenced_models,
            )
            self._vprint("Face/PPE worker initialized and warmed up successfully")
            self._log("Face/PPE worker initialized and warmed up successfully")
        except Exception as e:
            self._vprint(f"Warning: Face matcher initialization failed: {e}")
            self._log(f"Warning: Face matcher initialization failed: {e}")
            self.face_worker = None
    
    def _init_session_log(self):
        """Log system ready (actual logging is handled by main.py setup_logging)"""
        pass
    
    def _log(self, message: str):
        """Log a message using the standard Python logging system (asynchronous + buffered by QueueListener)"""
        logging.info(message)
    
    def _vprint(self, *args, **kwargs):
        """Conditional verbose print"""
        if self.verbose:
            print(*args, **kwargs)
    
    def _check_and_update_thermal_throttle(self):
        """Check CPU temperature and update thermal throttle state."""
        now = time.time()
        if now - self._last_thermal_check < self._thermal_check_interval:
            return
        self._last_thermal_check = now

        try:
            temp_path = "/sys/class/thermal/thermal_zone0/temp"
            with open(temp_path, "r") as f:
                raw = f.read().strip()
            temp_c = float(raw) / 1000.0 if float(raw) > 1000 else float(raw)
            
            # CRITICAL: Hard disable face matching if temperature exceeds safety threshold
            if not self._face_matching_disabled and temp_c >= self._critical_temp_threshold:
                self._face_matching_disabled = True
                self._thermal_throttled = True
                self._log(f"CRITICAL THERMAL: Permanently disabled face matching (temp={temp_c:.1f}°C >= {self._critical_temp_threshold}°C)")
                return
            
            # Normal throttling logic (only if not critically disabled)
            if self._thermal_throttled and temp_c < self._temp_resume_threshold:
                self._thermal_throttled = False
                self._log(f"Thermal: Resumed face detection (temp={temp_c:.1f}°C)")
            elif not self._thermal_throttled and temp_c > self._temp_override_threshold:
                self._thermal_throttled = True
                self._log(f"Thermal: Throttled face detection (temp={temp_c:.1f}°C > {self._temp_override_threshold}°C)")
        except Exception:
            pass  # Silent fail on temp read

    def process_frame(self, frame):
        # Check thermal status and potentially throttle
        self._check_and_update_thermal_throttle()

        if 'human' in self.silenced_models:
            return []
        
        results = self.identification_service.process_frame(frame)
        
        # Drain completed face/PPE results
        if self.face_worker is not None:
            for pid, result in self.face_worker.drain_results().items():
                # Pop the job type so we know what just finished
                job_type = self.in_flight.pop(pid, None) 
                
                name, confidence, missing_items, all_present = result
                
                # Process face match result
                if name is not None:
                    updated = self.person_tracker.add_match(pid, name, confidence)
                    if updated:
                        self._log(f"Person {pid}: Identity confirmed as {name} (confidence: {confidence:.4f})")
                
                # CRITICAL FIX: Always increment face attempts if a face job ran, 
                # even if it failed, to properly trigger the cooldown!
                if job_type in ['face', 'face+ppe'] or job_type is None:
                    self.person_tracker.increment_attempts(pid)
                
                # Process PPE detection result
                if missing_items is not None:
                    self.person_tracker.add_ppe_result(pid, missing_items, all_present)
                
                # Increment PPE attempts to trigger the PPE cooldown
                if job_type in ['ppe', 'face+ppe'] or job_type is None:
                    self.person_tracker.increment_ppe_attempts(pid)
        
        # Submit new face/PPE detection jobs
        for person in results:
            person_id = person['id']
            if (person['bbox'] is not None
                    and person['ready_to_save']
                    and self.face_worker is not None
                    and person_id not in self.in_flight
                    and not self._thermal_throttled
                    and not self._face_matching_disabled):
                
                crop = person.get('crop')
                if crop is not None:
                    
                    # Evaluate what jobs we need to run
                    needs_face = (not self.person_tracker.is_identified(person_id)
                                  and self.person_tracker.can_attempt(
                                      person_id,
                                      initial_cooldown=self.face_cooldown,
                                      slow_cooldown=self.face_retry_slow_cooldown,
                                      fast_attempts=self.face_retry_limit,
                                  ))
                    
                    # Sample PPE every 1.5 seconds to build the cumulative score without overloading
                    needs_ppe = self.person_tracker.can_attempt_ppe(person_id, cooldown=1.5)
                    
                    job_type = None
                    if not self.enable_face_matching:
                        if needs_ppe: 
                            job_type = 'ppe'
                    else:
                        if needs_face and needs_ppe:
                            job_type = 'face+ppe'
                        elif needs_face:
                            job_type = 'face'
                        elif needs_ppe:
                            job_type = 'ppe'
                    
                    if job_type is not None:
                        submitted = self.face_worker.submit(person_id, crop.copy(), job_type=job_type)
                        if submitted:
                            # Save the job type so we know what to cooldown later
                            self.in_flight[person_id] = job_type  
        
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