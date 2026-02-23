import os
import cv2

from face_detector import detect_face


IMAGE_FOLDER = "face_detection_images"


def main():
    for filename in os.listdir(IMAGE_FOLDER):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        image_path = os.path.join(IMAGE_FOLDER, filename)
        print(f"\nProcessing: {filename}")

        # Load image
        image = cv2.imread(image_path)

        if image is None:
            print("Couldn't load image")
            continue

        face = detect_face(image)

        if face is None:
            print("No face detected")
            continue

        print("Face detected!")

        #result
        cv2.imshow("Original", image)
        cv2.imshow("Face Crop", face)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        save_path = os.path.join("face_detection_images", f"croppedFace_{filename}")
        cv2.imwrite(save_path, face)


if __name__ == "__main__":
    main()