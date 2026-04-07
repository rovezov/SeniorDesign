"""
Display Module - Visualization and Overlay Rendering

Handles real-time video display with detection overlays.
Renders person tracking visualizations, PPE compliance status, and statistics.
"""

import cv2
import time
import os
from detection_pipeline import DetectionPipeline


def _ts() -> str:
    """Current wall-clock time as HH:MM:SS.mmm for log prefixes."""
    t = time.localtime()
    ms = int((time.time() % 1) * 1000)
    return f"{t.tm_hour:02d}:{t.tm_min:02d}:{t.tm_sec:02d}.{ms:03d}"


class DisplayRenderer:
    """Handles rendering overlays on video frames"""
    
    @staticmethod
    def draw_person_overlays(frame, person, pipeline):
        """Draw person tracking overlays on frame"""
        person_id = person['id']
        
        if person['bbox'] is not None:
            # Draw bounding box
            x, y, w, h = person['bbox']
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Draw centroid
            cx, cy = person['centroid']
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
            
            # Draw ID label
            cv2.putText(frame, f"ID {person_id}", (cx - 10, cy - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Draw name label
            name = pipeline.person_tracker.get_name(person_id)
            cv2.putText(frame, name, (cx - 10, cy + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # Draw PPE compliance status
            ppe_status = pipeline.person_tracker.get_ppe_status(person_id)
            if ppe_status['is_non_compliant']:
                # Non-compliant: red text with missing items
                missing_text = f"MISSING: {','.join(ppe_status['missing']).upper()}"
                color = (0, 0, 255)  # Red for non-compliant
            else:
                # Compliant: green text
                missing_text = "COMPLIANT"
                color = (0, 255, 0)  # Green for compliant
            cv2.putText(frame, missing_text, (cx - 10, cy + 45),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw "TRACKED" status if ready to save
            if person['ready_to_save']:
                cv2.putText(frame, "TRACKED", (cx - 10, cy + 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    @staticmethod
    def draw_statistics_overlay(frame, results, pipeline, avg_fps):
        """Draw statistics overlay on frame"""
        detected_count = sum(1 for p in results if p['bbox'] is not None)
        identified_count = sum(1 for p in results if pipeline.person_tracker.is_identified(p['id']))
        
        cv2.putText(frame, f"FPS: {avg_fps:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        cv2.putText(frame, f"Detected: {detected_count}", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Tracked: {len(results)}", (10, 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Identified: {identified_count}", (10, 130),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Saved: {len(pipeline.identification_service.saved_ids)}", (10, 160),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
