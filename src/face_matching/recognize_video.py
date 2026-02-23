import sys
import cv2
from face_matcher import FaceMatcher


def _main():
    if len(sys.argv) < 2:
        print("Usage: python recognize_video.py /path/to/cropped_face.jpg")
        sys.exit(1)

    face_path = sys.argv[1]

    matcher = FaceMatcher()
    # Try load existing model; if not present, train from known faces and save
    try:
        matcher.load()
    except Exception:
        matcher.train_and_save()

    img = cv2.imread(face_path)
    if img is None:
        print(f"Unable to read image: {face_path}")
        sys.exit(2)

    name, conf = matcher.match_face_image(img)
    print((name, conf))


if __name__ == "__main__":
    _main()