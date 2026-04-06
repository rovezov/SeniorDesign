"""
PPE Detector – Visual Test Script

For each image in test_images/:
  1. Runs the HumanIdentificationService to detect and crop every person.
  2. Runs PPEDetector on each crop.
  3. Prints a per-image summary, draws annotated bounding boxes in the original
     image coordinate space, and saves annotated_<filename>.jpg.

Usage (from project root):
    python src/ppe_detection/test_ppe_detector.py

Optional flags:
    --no-display   Skip cv2.imshow (useful on headless machines).
    --conf 0.4     Override the PPE confidence threshold (default 0.3).
    --req helmet vest    Override required PPE items (default: helmet vest).
"""

import argparse
import os
import sys

import cv2

# Allow running from any cwd
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.ppe_detection.ppe_detector import PPEDetector
from src.human_detection.human_identifier import HumanIdentificationService


# ── Colour palette for bounding boxes (BGR) ──────────────────────────────────
_LABEL_COLOURS = {
    "helmet": (0, 255, 0),    # green
    "vest":   (0, 200, 255),  # amber
}
_DEFAULT_COLOUR = (200, 200, 200)


def draw_results(image: "cv2.Mat", result: dict, draw_banner: bool = True) -> "cv2.Mat":
    """Return a copy of *image* annotated with detections and an optional status banner."""
    annotated = image.copy()

    for det in result["detected"]:
        label = det["label"]
        conf  = det["confidence"]
        x, y, w, h = det["bbox"]
        colour = _LABEL_COLOURS.get(label, _DEFAULT_COLOUR)

        cv2.rectangle(annotated, (x, y), (x + w, y + h), colour, 2)
        text = f"{label} {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(annotated, (x, y - th - 8), (x + tw + 4, y), colour, -1)
        cv2.putText(annotated, text, (x + 2, y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    if draw_banner:
        banner_colour = (0, 180, 0) if result["all_present"] else (0, 0, 220)
        if result["all_present"]:
            banner_text = "ALL PPE PRESENT"
        else:
            banner_text = "MISSING: " + ", ".join(result["missing"]).upper()
        h_img = annotated.shape[0]
        
        cv2.rectangle(annotated, (0, h_img - 36), (annotated.shape[1], h_img), banner_colour, -1)
        cv2.putText(annotated, banner_text, (8, h_img - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return annotated


def print_result(filename: str, result: dict) -> None:
    """Pretty-print one image's detection result."""
    print(f"\n{'─' * 55}")
    print(f"  Image : {filename}")
    print(f"  Status: {'✔  ALL PRESENT' if result['all_present'] else '✘  VIOLATION'}")
    if result["missing"]:
        print("MISSING: " + ", ".join(result["missing"]).upper())
    if result["detected"]:
        print("  Detected:")
        for det in result["detected"]:
            x, y, w, h = det["bbox"]
            print(f"    • {det['label']:<10} conf={det['confidence']:.2f}  "
                  f"bbox=[{x},{y},{w},{h}]")
    else:
        print("  Detected: (none)")

    if result["missing"]:
        print(f"  Missing : {', '.join(result['missing'])}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visual PPE detection test")
    parser.add_argument("--no-display", action="store_true",
                        help="Skip cv2.imshow (headless mode)")
    parser.add_argument("--conf", type=float, default=0.3,
                        help="PPE confidence threshold (default: 0.3)")
    parser.add_argument("--req", nargs="+", default=["helmet", "vest"],
                        metavar="ITEM", help="Required PPE items (default: helmet vest)")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(script_dir, "test_images")

    # ── Initialise detectors ─────────────────────────────────────────────────
    print("Initialising HumanIdentificationService...")
    human_detector = HumanIdentificationService()

    print(f"Initialising PPEDetector")
    print(f"  Required PPE  : {args.req}")
    print(f"  Conf threshold: {args.conf}")
    ppe_detector = PPEDetector(requirements=args.req, conf_threshold=args.conf)

    # ── Discover test images ─────────────────────────────────────────────────
    supported = {".jpg", ".jpeg", ".png"}
    image_files = sorted(
        f for f in os.listdir(images_dir)
        if os.path.splitext(f)[1].lower() in supported
        and not f.startswith("annotated_")
    )

    if not image_files:
        print(f"\nNo test images found in: {images_dir}")
        print("Add .jpg / .jpeg / .png files and re-run.")
        sys.exit(0)

    print(f"\nFound {len(image_files)} image(s) in {images_dir}\n")

    passed = 0
    failed = 0

    # ── Process each image ───────────────────────────────────────────────────
    for filename in image_files:
        image_path = os.path.join(images_dir, filename)
        image = cv2.imread(image_path)
        if image is None:
            print(f"  [WARN] Could not load {filename}, skipping.")
            continue

        print(f"\n{'─' * 55}")
        print(f"  Image : {filename}")

        # Step 1 – detect person bounding boxes
        person_bboxes = human_detector._detect_humans(image)

        if not person_bboxes:
            print("  [WARN] No people detected — running PPE on full image as fallback.")
            person_bboxes = [(0, 0, image.shape[1], image.shape[0])]

        print(f"  People found : {len(person_bboxes)}")

        annotated = image.copy()
        image_all_present = True
        image_missing: set[str] = set()

        for person_idx, (px, py, pw, ph) in enumerate(person_bboxes):
            crop = human_detector._crop_person(image, (px, py, pw, ph))
            if crop is None or crop.size == 0:
                continue

            # Step 2 – run PPE detection on the crop
            result = ppe_detector.detect(crop)

            # Map crop-relative bboxes back to full-image coordinates
            full_detections = []
            for det in result["detected"]:
                cx, cy, cw, ch = det["bbox"]
                full_detections.append({
                    "label":      det["label"],
                    "confidence": det["confidence"],
                    "bbox":       [px + cx, py + cy, cw, ch],
                })

            status = "✔  ALL PRESENT" if result["all_present"] else "✘  VIOLATION"
            print("MISSING: " + ", ".join(result["missing"]).upper())
            print(f"  Person {person_idx + 1}: {status}")
            if full_detections:
                for det in full_detections:
                    x, y, w, h = det["bbox"]
                    print(f"    • {det['label']:<10} conf={det['confidence']:.2f}  "
                          f"bbox=[{x},{y},{w},{h}]")
            else:
                print("    Detected: (none)")

            if result["missing"]:
                print(f"    Missing : {', '.join(result['missing'])}")
                image_missing.update(result["missing"])
                image_all_present = False

            # Draw person bounding box
            cv2.rectangle(annotated, (px, py), (px + pw, py + ph), (255, 165, 0), 2)
            cv2.putText(annotated, f"Person {person_idx + 1}", (px + 4, py + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 165, 0), 2)

            # Draw PPE boxes mapped back to full-image space
            annotated = draw_results(annotated, {"detected": full_detections,
                                                  "missing": result["missing"],
                                                  "all_present": result["all_present"]},
                                     draw_banner=False)

        # Draw one overall status banner at the bottom
        banner_colour = (0, 180, 0) if image_all_present else (0, 0, 220)
        banner_text = ("ALL PPE PRESENT" if image_all_present
                       else "MISSING: " + ", ".join(sorted(image_missing)).upper())
        h_img = annotated.shape[0]
        cv2.rectangle(annotated, (0, h_img - 36), (annotated.shape[1], h_img), banner_colour, -1)
        cv2.putText(annotated, banner_text, (8, h_img - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if image_all_present:
            passed += 1
        else:
            failed += 1

        save_name = "annotated_" + os.path.splitext(filename)[0] + ".jpg"
        save_path = os.path.join(images_dir, save_name)
        cv2.imwrite(save_path, annotated)
        print(f"  Saved : {save_name}")

        if not args.no_display:
            cv2.imshow(f"PPE Detection – {filename}", annotated)
            print("  Press any key to continue...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    # ── Summary ──────────────────────────────────────────────────────────────
    total = passed + failed
    print(f"\n{'═' * 55}")
    print(f"  Results: {passed}/{total} images with all PPE present")
    print(f"{'═' * 55}\n")

    # ── Demo: update_requirements ─────────────────────────────────────────────
    print("Demo – update_requirements(['helmet']):")
    ppe_detector.update_requirements(["helmet"])
    print(f"  New requirements: {ppe_detector.requirements}")
    demo_image = cv2.imread(os.path.join(images_dir, image_files[0]))
    if demo_image is not None:
        demo_bboxes = human_detector._detect_humans(demo_image)
        if not demo_bboxes:
            demo_bboxes = [(0, 0, demo_image.shape[1], demo_image.shape[0])]
        crop = human_detector._crop_person(demo_image, demo_bboxes[0])
        if crop is not None:
            demo_result = ppe_detector.detect(crop)
            print(f"  Re-ran on '{image_files[0]}': all_present={demo_result['all_present']}, "
                  f"missing={demo_result['missing']}")


if __name__ == "__main__":
    main()
