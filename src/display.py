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
                       cv2.FONT_HERSHEY_SIMPLEX, 0.805, (0, 255, 255), 2)

            # Draw name label
            name = pipeline.person_tracker.get_name(person_id)
            cv2.putText(frame, f"Name: {name}", (cx - 10, cy + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.69, (255, 255, 0), 2)

            # Draw PPE compliance status + missing items
            ppe_status = pipeline.person_tracker.get_ppe_status(person_id)
            
            # Use 'current_compliant' for real-time video overlay
            if not ppe_status.get('current_compliant', True):
                compliance_text = "NON-COMPLIANT"
                missing_items = ppe_status.get('current_missing', [])
                missing_text = "Missing: None" if not missing_items else f"Missing: {', '.join(missing_items)}"
                color = (0, 0, 255)  # Red for non-compliant
            else:
                compliance_text = "COMPLIANT"
                missing_text = "Missing: None"
                color = (0, 255, 0)  # Green for compliant

            cv2.putText(frame, f"Compliance: {compliance_text}", (cx - 10, cy + 45),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.575, color, 2)
            cv2.putText(frame, missing_text, (cx - 10, cy + 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.575, color, 2)

    @staticmethod
    def draw_statistics_overlay(frame, results, pipeline, current_fps, avg_fps=None):
        """Draw FPS overlay on frame."""
        cv2.putText(frame, f"FPS: {current_fps:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.15, (255, 255, 0), 2)
    
