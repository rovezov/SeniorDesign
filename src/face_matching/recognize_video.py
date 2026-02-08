import cv2
import os
import numpy as np
import cvzone

from cvzone.FaceDetectionModule import FaceDetector

KNOWN_DIR = "known_faces"
IMG_SIZE = (200, 200)   # all faces resized to same size for LBPH


# -------------------------------------------------------------
# LOAD TRAINING DATA
# -------------------------------------------------------------
print("[INFO] Loading known faces...")

faces = []
labels = []
label_map = {}  # id → name
current_label = 0

for person_name in os.listdir(KNOWN_DIR):
    person_path = os.path.join(KNOWN_DIR, person_name)

    if not os.path.isdir(person_path):
        continue

    label_map[current_label] = person_name

    for filename in os.listdir(person_path):
        file_path = os.path.join(person_path, filename)
        img = cv2.imread(file_path)

        if img is None:
            continue

        # Detect + crop face using CVZone’s detector
        detector = FaceDetector()
        _, bboxs = detector.findFaces(img, draw=False)

        if bboxs:
            x, y, w, h = bboxs[0]["bbox"]
            face_crop = img[y:y+h, x:x+w]

            # Normalize & store
            gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, IMG_SIZE)

            faces.append(gray)
            labels.append(current_label)

    current_label += 1

faces = np.array(faces)
labels = np.array(labels)

print(f"[INFO] Loaded {len(faces)} face samples.")


# -------------------------------------------------------------
# TRAIN LBPH FACE RECOGNIZER
# -------------------------------------------------------------
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(faces, labels)

print("[INFO] Training complete!")


# -------------------------------------------------------------
# WEBCAM RECOGNITION LOOP
# -------------------------------------------------------------
cap = cv2.VideoCapture(0)
detector = FaceDetector()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame, bboxs = detector.findFaces(frame, draw=False)

    for bbox in bboxs:
        x, y, w, h = bbox["bbox"]

        face_crop = frame[y:y+h, x:x+w]

        if face_crop.size > 0:
            gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, IMG_SIZE)

            # Predict identity
            label_id, confidence = recognizer.predict(gray)

            # Lower confidence = better match
            if confidence < 70:  # tweak threshold
                name = label_map.get(label_id, "Unknown")
            else:
                name = "Unknown"

            # Draw UI
            cvzone.cornerRect(frame, (x, y, w, h))
            cvzone.putTextRect(frame, f"{name} ({int(confidence)})", (x, y - 10),
                               scale=1, thickness=2)

    cv2.imshow("CVZone Face Recognition (LBPH)", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()