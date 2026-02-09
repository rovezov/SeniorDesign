import os
from face_detector import detect_faces

HERE = os.path.dirname(__file__)

# List of test images
test_images = [
    "image1.jpeg",
    "image2.jpeg",
    "image3.jpeg"
]

for i, img_name in enumerate(test_images):
    person_image = os.path.join(
        HERE,
        "face_detection_images",
        img_name
    )

    person_id = i + 1

    print(f"\nTesting image: {img_name}")

    faces = detect_faces(person_image, person_id)

    print("Detected faces:")
    for f in faces:
        print(f)
