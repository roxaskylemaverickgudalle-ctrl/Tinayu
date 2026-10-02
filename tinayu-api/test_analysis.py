from pathlib import Path

from tinayu_engine import analyze_image


IMAGE_PATH = Path("test.jpg")


with open(IMAGE_PATH, "rb") as file:
    image_bytes = file.read()


result = analyze_image(image_bytes)


print("\n========== TINAYU ANALYSIS ==========\n")

print(
    "Success:",
    result["success"]
)

print(
    "Image:",
    result["image"]
)

print(
    "Face confidence:",
    result["face"]["confidence"]
)

print(
    "Landmarks:",
    result["face"]["landmarks"]
)

print(
    "Raw skin RGB:",
    result["colors"]["skin_raw_rgb"]
)

print(
    "Normalized skin RGB:",
    result["colors"]["skin_normalized_rgb"]
)

print(
    "Skin LAB:",
    result["colors"]["skin_lab"]
)

print("\n=====================================\n")