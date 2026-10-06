import sys
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

MODEL_PATH = "face_landmarker.task"


# These are the same general cheek-side regions we want to inspect.
# The diagnostic is intended to study pixel distribution, not change
# Tinayu's production extraction logic.
LEFT_CHEEK = [
    50, 101, 118, 117, 111, 123, 147, 187,
    205, 206, 207, 216, 212, 202, 194, 182,
]

RIGHT_CHEEK = [
    280, 330, 347, 346, 340, 352, 376, 411,
    425, 426, 427, 436, 432, 422, 414, 406,
]


def temperature_score(lab_a, lab_b):
    a = float(lab_a)
    b = float(lab_b)

    signal = (
        ((a - 8.0) / 8.0) * 0.40
        + ((b - 14.0) / 8.0) * 0.60
    )

    score = 1.0 / (1.0 + np.exp(-signal))

    return float(np.clip(score, 0.0, 1.0))


def classify_temperature(score):
    if score >= 0.55:
        return "Warm"

    if score <= 0.45:
        return "Cool"

    return "Neutral"


def rgb_to_lab(pixels):
    pixels = np.asarray(pixels, dtype=np.float32)

    if pixels.size == 0:
        return np.empty((0, 3), dtype=np.float32)

    pixels = np.clip(pixels, 0, 255).astype(np.uint8)

    return cv2.cvtColor(
        pixels.reshape(-1, 1, 3),
        cv2.COLOR_RGB2LAB,
    ).reshape(-1, 3).astype(np.float32)


def polygon_mask(image_shape, landmarks, indices):
    height, width = image_shape[:2]

    points = []

    for index in indices:
        landmark = landmarks[index]

        x = int(
            np.clip(
                landmark.x * width,
                0,
                width - 1,
            )
        )

        y = int(
            np.clip(
                landmark.y * height,
                0,
                height - 1,
            )
        )

        points.append([x, y])

    points = np.asarray(points, dtype=np.int32)

    mask = np.zeros(
        (height, width),
        dtype=np.uint8,
    )

    if len(points) >= 3:
        cv2.fillConvexPoly(
            mask,
            points,
            255,
        )

    return mask


def extract_pixels(image, mask):
    rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB,
    )

    pixels = rgb[mask > 0]

    if pixels.size == 0:
        return np.empty(
            (0, 3),
            dtype=np.float32,
        )

    return pixels.astype(np.float32)


def p10_p90_range(values):
    if len(values) == 0:
        return 0.0

    return float(
        np.percentile(values, 90)
        - np.percentile(values, 10)
    )


def describe_cheek(name, pixels):
    print(f"\n  {name}")

    if len(pixels) == 0:
        print("    No pixels detected")
        return None

    lab = rgb_to_lab(pixels)

    rgb_mean = np.mean(
        pixels,
        axis=0,
    )

    rgb_median = np.median(
        pixels,
        axis=0,
    )

    rgb_std = np.std(
        pixels,
        axis=0,
    )

    lab_mean = np.mean(
        lab,
        axis=0,
    )

    lab_median = np.median(
        lab,
        axis=0,
    )

    lab_std = np.std(
        lab,
        axis=0,
    )

    score = temperature_score(
        lab_mean[1],
        lab_mean[2],
    )

    temperature = classify_temperature(score)

    print(
        f"    Pixel count:       {len(pixels)}"
    )

    print(
        "    RGB mean:          "
        f"[{rgb_mean[0]:.1f}, "
        f"{rgb_mean[1]:.1f}, "
        f"{rgb_mean[2]:.1f}]"
    )

    print(
        "    RGB median:        "
        f"[{rgb_median[0]:.1f}, "
        f"{rgb_median[1]:.1f}, "
        f"{rgb_median[2]:.1f}]"
    )

    print(
        "    RGB std:           "
        f"[{rgb_std[0]:.1f}, "
        f"{rgb_std[1]:.1f}, "
        f"{rgb_std[2]:.1f}]"
    )

    print(
        "    RGB P10-P90:       "
        f"[{p10_p90_range(pixels[:, 0]):.1f}, "
        f"{p10_p90_range(pixels[:, 1]):.1f}, "
        f"{p10_p90_range(pixels[:, 2]):.1f}]"
    )

    print(
        "    LAB mean:          "
        f"[{lab_mean[0]:.1f}, "
        f"{lab_mean[1]:.1f}, "
        f"{lab_mean[2]:.1f}]"
    )

    print(
        "    LAB median:        "
        f"[{lab_median[0]:.1f}, "
        f"{lab_median[1]:.1f}, "
        f"{lab_median[2]:.1f}]"
    )

    print(
        "    LAB std:           "
        f"[{lab_std[0]:.1f}, "
        f"{lab_std[1]:.1f}, "
        f"{lab_std[2]:.1f}]"
    )

    print(
        "    LAB a P10-P90:     "
        f"{p10_p90_range(lab[:, 1]):.1f}"
    )

    print(
        "    LAB b P10-P90:     "
        f"{p10_p90_range(lab[:, 2]):.1f}"
    )

    print(
        f"    Temperature score: "
        f"{score:.4f} {temperature}"
    )

    return {
        "count": len(pixels),
        "rgb_mean": rgb_mean,
        "rgb_median": rgb_median,
        "rgb_std": rgb_std,
        "lab_mean": lab_mean,
        "lab_median": lab_median,
        "lab_std": lab_std,
        "temperature_score": score,
        "temperature": temperature,
    }


def analyze_image(image_path, face_landmarker):
    image = cv2.imread(
        str(image_path)
    )

    if image is None:
        print(
            f"\nCould not read image: "
            f"{image_path.name}"
        )
        return

    rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB,
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb,
    )

    result = face_landmarker.detect(
        mp_image
    )

    print("\n" + "=" * 80)
    print(image_path.name)
    print("=" * 80)

    if not result.face_landmarks:
        print("  No face detected")
        return

    landmarks = result.face_landmarks[0]

    left_mask = polygon_mask(
        image.shape,
        landmarks,
        LEFT_CHEEK,
    )

    right_mask = polygon_mask(
        image.shape,
        landmarks,
        RIGHT_CHEEK,
    )

    left_pixels = extract_pixels(
        image,
        left_mask,
    )

    right_pixels = extract_pixels(
        image,
        right_mask,
    )

    left = describe_cheek(
        "LEFT CHEEK",
        left_pixels,
    )

    right = describe_cheek(
        "RIGHT CHEEK",
        right_pixels,
    )

    if left is None or right is None:
        return

    score_delta = abs(
        left["temperature_score"]
        - right["temperature_score"]
    )

    lab_a_delta = abs(
        left["lab_mean"][1]
        - right["lab_mean"][1]
    )

    lab_b_delta = abs(
        left["lab_mean"][2]
        - right["lab_mean"][2]
    )

    pixel_ratio = (
        min(
            left["count"],
            right["count"],
        )
        /
        max(
            left["count"],
            right["count"],
        )
    )

    print("\n  LEFT vs RIGHT")

    print(
        f"    Temperature score delta: "
        f"{score_delta:.4f}"
    )

    print(
        f"    LAB a mean delta:         "
        f"{lab_a_delta:.2f}"
    )

    print(
        f"    LAB b mean delta:         "
        f"{lab_b_delta:.2f}"
    )

    print(
        f"    Pixel-count ratio:        "
        f"{pixel_ratio:.3f}"
    )

    if score_delta >= 0.10:
        print(
            "    Temperature agreement:   HIGH DISAGREEMENT"
        )
    elif score_delta >= 0.05:
        print(
            "    Temperature agreement:   MODERATE DISAGREEMENT"
        )
    else:
        print(
            "    Temperature agreement:   GOOD"
        )

    max_ab_std = max(
        left["lab_std"][1],
        left["lab_std"][2],
        right["lab_std"][1],
        right["lab_std"][2],
    )

    if max_ab_std >= 8:
        print(
            "    Color distribution:       WIDE"
        )
    else:
        print(
            "    Color distribution:       RELATIVELY TIGHT"
        )


def main():
    if len(sys.argv) != 2:
        print(
            "Usage:\n"
            "python evaluate_cheek_pixel_distribution.py "
            "<sample_directory>"
        )
        sys.exit(1)

    sample_directory = Path(
        sys.argv[1]
    )

    if not sample_directory.exists():
        print(
            f"Directory not found: "
            f"{sample_directory}"
        )
        sys.exit(1)

    image_paths = sorted(
        path
        for path in sample_directory.iterdir()
        if path.is_file()
        and path.suffix.lower()
        in IMAGE_EXTENSIONS
    )

    if not image_paths:
        print(
            f"No supported images found in: "
            f"{sample_directory}"
        )
        sys.exit(1)

    base_options = mp.tasks.BaseOptions(
        model_asset_path=MODEL_PATH
    )

    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=(
            mp.tasks.vision.RunningMode.IMAGE
        ),
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )

    with mp.tasks.vision.FaceLandmarker.create_from_options(
        options
    ) as face_landmarker:

        for image_path in image_paths:
            analyze_image(
                image_path,
                face_landmarker,
            )


if __name__ == "__main__":
    main()