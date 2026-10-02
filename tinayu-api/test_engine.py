from pathlib import Path

from tinayu_engine import (
    load_image,
    detect_face,
    detect_landmarks,
)


IMAGE_PATH = Path("test.jpg")


with open(IMAGE_PATH, "rb") as file:
    image_bytes = file.read()


image = load_image(image_bytes)

print(
    f"Image shape: {image.shape}"
)


face_result, confidence = detect_face(image)

print(
    f"Face detected: {confidence:.2%}"
)


landmarks = detect_landmarks(image)

print(
    f"Landmarks detected: {len(landmarks)}"
)

print("✅ Tinayu engine foundation works!")

