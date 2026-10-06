import sys
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
MODEL_PATH = "face_landmarker.task"


# ---------------------------------------------------------------------------
# MediaPipe cheek landmark regions
# ---------------------------------------------------------------------------

LEFT_CHEEK = [
    50, 101, 118, 117, 111, 123, 147, 187,
    205, 206, 207, 216, 212, 202, 194, 182,
]

RIGHT_CHEEK = [
    280, 330, 347, 346, 340, 352, 376, 411,
    425, 426, 427, 436, 432, 422, 414, 406,
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def rgb_to_lab(pixels):
    if len(pixels) == 0:
        return np.empty((0, 3), dtype=np.float32)

    pixels = np.clip(
        pixels,
        0,
        255,
    ).astype(np.uint8)

    return cv2.cvtColor(
        pixels.reshape(-1, 1, 3),
        cv2.COLOR_RGB2LAB,
    ).reshape(-1, 3).astype(np.float32)


def temperature_score_from_lab(lab_a, lab_b):
    """
    Diagnostic-only version.

    MediaPipe/OpenCV LAB values are converted back to the approximate
    centered a/b representation used by Tinayu's production formula.
    """

    # OpenCV LAB:
    #   a/b are encoded around 128.
    #
    # Tinayu production logic uses approximately centered a/b values.
    a = float(lab_a) - 128.0
    b = float(lab_b) - 128.0

    signal = (
        ((a - 8.0) / 8.0) * 0.40
        + ((b - 14.0) / 8.0) * 0.60
    )

    score = 1.0 / (1.0 + np.exp(-signal))

    return float(
        np.clip(score, 0.0, 1.0)
    )


def polygon_points(image_shape, landmarks, indices):
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

    return np.asarray(
        points,
        dtype=np.float32,
    )


def create_polygon_mask(
    image_shape,
    landmarks,
    indices,
):
    height, width = image_shape[:2]

    points = polygon_points(
        image_shape,
        landmarks,
        indices,
    ).astype(np.int32)

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


def create_centered_mask(
    image_shape,
    landmarks,
    indices,
    scale,
):
    """
    Shrinks a cheek polygon toward its centroid.

    scale:
        1.00 = broad/full region
        0.70 = medium region
        0.45 = central region
    """

    height, width = image_shape[:2]

    points = polygon_points(
        image_shape,
        landmarks,
        indices,
    )

    center = np.mean(
        points,
        axis=0,
    )

    centered = (
        center
        + (points - center) * scale
    )

    centered = np.round(
        centered
    ).astype(np.int32)

    centered[:, 0] = np.clip(
        centered[:, 0],
        0,
        width - 1,
    )

    centered[:, 1] = np.clip(
        centered[:, 1],
        0,
        height - 1,
    )

    mask = np.zeros(
        (height, width),
        dtype=np.uint8,
    )

    cv2.fillConvexPoly(
        mask,
        centered,
        255,
    )

    return mask


def extract_pixels(
    image,
    mask,
):
    rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB,
    )

    pixels = rgb[
        mask > 0
    ]

    return pixels.astype(
        np.float32
    )


def analyze_region(
    pixels,
):
    if len(pixels) == 0:
        return None

    lab = rgb_to_lab(
        pixels
    )

    rgb_mean = np.mean(
        pixels,
        axis=0,
    )

    lab_mean = np.mean(
        lab,
        axis=0,
    )

    lab_std = np.std(
        lab,
        axis=0,
    )

    temperature_score = (
        temperature_score_from_lab(
            lab_mean[1],
            lab_mean[2],
        )
    )

    return {
        "count": len(pixels),
        "rgb_mean": rgb_mean,
        "lab_mean": lab_mean,
        "lab_std": lab_std,
        "temperature_score": temperature_score,
    }


def print_region(
    label,
    left,
    right,
):
    if left is None or right is None:
        print(
            f"\n  {label}"
        )
        print(
            "    Unable to analyze"
        )
        return None

    score_delta = abs(
        left["temperature_score"]
        - right["temperature_score"]
    )

    a_delta = abs(
        (
            left["lab_mean"][1]
            - 128
        )
        -
        (
            right["lab_mean"][1]
            - 128
        )
    )

    b_delta = abs(
        (
            left["lab_mean"][2]
            - 128
        )
        -
        (
            right["lab_mean"][2]
            - 128
        )
    )

    count_ratio = (
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

    print(
        f"\n  {label}"
    )

    print(
        f"    Left pixels:        "
        f"{left['count']}"
    )

    print(
        f"    Right pixels:       "
        f"{right['count']}"
    )

    print(
        f"    Pixel ratio:        "
        f"{count_ratio:.3f}"
    )

    print(
        f"    Left temp score:    "
        f"{left['temperature_score']:.4f}"
    )

    print(
        f"    Right temp score:   "
        f"{right['temperature_score']:.4f}"
    )

    print(
        f"    Score delta:        "
        f"{score_delta:.4f}"
    )

    print(
        f"    LAB a delta:        "
        f"{a_delta:.2f}"
    )

    print(
        f"    LAB b delta:        "
        f"{b_delta:.2f}"
    )

    print(
        f"    Left LAB a/b std:   "
        f"{left['lab_std'][1]:.2f} / "
        f"{left['lab_std'][2]:.2f}"
    )

    print(
        f"    Right LAB a/b std:  "
        f"{right['lab_std'][1]:.2f} / "
        f"{right['lab_std'][2]:.2f}"
    )

    return {
        "score_delta": score_delta,
        "a_delta": a_delta,
        "b_delta": b_delta,
        "count_ratio": count_ratio,
    }


def analyze_image(
    image_path,
    face_landmarker,
):
    image = cv2.imread(
        str(image_path)
    )

    if image is None:
        print(
            f"\nCould not read "
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
        print(
            "  No face detected"
        )
        return

    landmarks = result.face_landmarks[0]

    region_results = {}

    for label, scale in [
        ("BROAD 100%", 1.00),
        ("MEDIUM 70%", 0.70),
        ("CENTRAL 45%", 0.45),
    ]:

        left_mask = create_centered_mask(
            image.shape,
            landmarks,
            LEFT_CHEEK,
            scale,
        )

        right_mask = create_centered_mask(
            image.shape,
            landmarks,
            RIGHT_CHEEK,
            scale,
        )

        left_pixels = extract_pixels(
            image,
            left_mask,
        )

        right_pixels = extract_pixels(
            image,
            right_mask,
        )

        left = analyze_region(
            left_pixels
        )

        right = analyze_region(
            right_pixels
        )

        region_results[label] = print_region(
            label,
            left,
            right,
        )

    # ---------------------------------------------------------------
    # Best-region comparison
    # ---------------------------------------------------------------

    valid = {
        key: value
        for key, value
        in region_results.items()
        if value is not None
    }

    if valid:
        best = min(
            valid.items(),
            key=lambda item: item[1]["score_delta"],
        )

        print(
            "\n  MOST STABLE BY "
            "LEFT/RIGHT TEMPERATURE DELTA:"
        )

        print(
            f"    {best[0]} "
            f"(delta {best[1]['score_delta']:.4f})"
        )


def main():
    if len(sys.argv) != 2:
        print(
            "Usage:\n"
            "python evaluate_cheek_region_stability.py "
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
        if (
            path.is_file()
            and path.suffix.lower()
            in IMAGE_EXTENSIONS
        )
    )

    if not image_paths:
        print(
            "No supported images found."
        )
        sys.exit(1)

    base_options = mp.tasks.BaseOptions(
        model_asset_path=MODEL_PATH
    )

    options = (
        mp.tasks.vision.FaceLandmarkerOptions(
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
    )

    with (
        mp.tasks.vision.FaceLandmarker
        .create_from_options(options)
        as face_landmarker
    ):
        for image_path in image_paths:
            analyze_image(
                image_path,
                face_landmarker,
            )


if __name__ == "__main__":
    main()