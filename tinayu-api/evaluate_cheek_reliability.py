import sys
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
MODEL_PATH = "face_landmarker.task"


LEFT_CHEEK = [
    50, 101, 118, 117, 111, 123, 147, 187,
    205, 206, 207, 216, 212, 202, 194, 182,
]

RIGHT_CHEEK = [
    280, 330, 347, 346, 340, 352, 376, 411,
    425, 426, 427, 436, 432, 422, 414, 406,
]


REGIONS = {
    "BROAD 100%": 1.00,
    "MEDIUM 70%": 0.70,
    "CENTRAL 45%": 0.45,
}


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


def temperature_score(lab_a, lab_b):
    # Convert OpenCV LAB a/b back to the centered representation
    # used by the Tinayu temperature formula.
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


def polygon_points(
    image_shape,
    landmarks,
    indices,
):
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


def create_centered_mask(
    image_shape,
    landmarks,
    indices,
    scale,
):
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

    scaled = (
        center
        + (points - center) * scale
    )

    scaled = np.round(
        scaled
    ).astype(np.int32)

    scaled[:, 0] = np.clip(
        scaled[:, 0],
        0,
        width - 1,
    )

    scaled[:, 1] = np.clip(
        scaled[:, 1],
        0,
        height - 1,
    )

    mask = np.zeros(
        (height, width),
        dtype=np.uint8,
    )

    cv2.fillConvexPoly(
        mask,
        scaled,
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

    return rgb[
        mask > 0
    ].astype(np.float32)


def analyze_cheek(pixels):
    if len(pixels) == 0:
        return None

    lab = rgb_to_lab(
        pixels
    )

    lab_mean = np.mean(
        lab,
        axis=0,
    )

    lab_std = np.std(
        lab,
        axis=0,
    )

    temperature = temperature_score(
        lab_mean[1],
        lab_mean[2],
    )

    return {
        "count": len(pixels),
        "lab_mean": lab_mean,
        "lab_std": lab_std,
        "temperature": temperature,
    }


def region_metrics(
    left,
    right,
):
    if left is None or right is None:
        return None

    left_count = left["count"]
    right_count = right["count"]

    if max(
        left_count,
        right_count,
    ) == 0:
        return None

    pixel_ratio = (
        min(
            left_count,
            right_count,
        )
        /
        max(
            left_count,
            right_count,
        )
    )

    score_delta = abs(
        left["temperature"]
        - right["temperature"]
    )

    a_delta = abs(
        (
            left["lab_mean"][1]
            - 128.0
        )
        -
        (
            right["lab_mean"][1]
            - 128.0
        )
    )

    b_delta = abs(
        (
            left["lab_mean"][2]
            - 128.0
        )
        -
        (
            right["lab_mean"][2]
            - 128.0
        )
    )

    a_spread = (
        left["lab_std"][1]
        + right["lab_std"][1]
    ) / 2.0

    b_spread = (
        left["lab_std"][2]
        + right["lab_std"][2]
    ) / 2.0

    return {
        "pixel_ratio": pixel_ratio,
        "score_delta": score_delta,
        "a_delta": a_delta,
        "b_delta": b_delta,
        "a_spread": a_spread,
        "b_spread": b_spread,
    }


def reliability_score(metrics):
    """
    Higher = theoretically more reliable.

    Components:

    1. Pixel balance:
       Equal left/right sampling is better.

    2. Chromatic agreement:
       Smaller left/right LAB a/b difference is better.

    3. Internal spread:
       Lower LAB chromatic spread is better.

    This is deliberately a diagnostic heuristic.
    It is NOT production logic yet.
    """

    pixel_balance = metrics["pixel_ratio"]

    chromatic_delta = (
        metrics["a_delta"]
        + metrics["b_delta"]
    )

    chromatic_agreement = 1.0 / (
        1.0 + chromatic_delta / 4.0
    )

    spread = (
        metrics["a_spread"]
        + metrics["b_spread"]
    )

    spread_quality = 1.0 / (
        1.0 + spread / 4.0
    )

    score = (
        pixel_balance * 0.40
        + chromatic_agreement * 0.40
        + spread_quality * 0.20
    )

    return float(
        np.clip(
            score,
            0.0,
            1.0,
        )
    )


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
        return None

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
        return None

    landmarks = result.face_landmarks[0]

    results = {}

    for region_name, scale in REGIONS.items():

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

        left = analyze_cheek(
            left_pixels
        )

        right = analyze_cheek(
            right_pixels
        )

        metrics = region_metrics(
            left,
            right,
        )

        if metrics is None:
            continue

        reliability = reliability_score(
            metrics
        )

        metrics["reliability"] = reliability

        results[region_name] = metrics

        print(
            f"\n  {region_name}"
        )

        print(
            f"    Pixel ratio:       "
            f"{metrics['pixel_ratio']:.3f}"
        )

        print(
            f"    Score delta:       "
            f"{metrics['score_delta']:.4f}"
        )

        print(
            f"    LAB a delta:       "
            f"{metrics['a_delta']:.2f}"
        )

        print(
            f"    LAB b delta:       "
            f"{metrics['b_delta']:.2f}"
        )

        print(
            f"    LAB a spread:      "
            f"{metrics['a_spread']:.2f}"
        )

        print(
            f"    LAB b spread:      "
            f"{metrics['b_spread']:.2f}"
        )

        print(
            f"    Reliability:       "
            f"{reliability:.4f}"
        )

    if not results:
        return None

    predicted = max(
        results.items(),
        key=lambda item: item[1]["reliability"],
    )

    actual = min(
        results.items(),
        key=lambda item: item[1]["score_delta"],
    )

    print(
        "\n  SELECTION COMPARISON"
    )

    print(
        f"    Predicted best:     "
        f"{predicted[0]}"
    )

    print(
        f"    Actual lowest Δ:    "
        f"{actual[0]}"
    )

    if predicted[0] == actual[0]:
        print(
            "    Prediction:         MATCH"
        )
        match = True
    else:
        print(
            "    Prediction:         MISMATCH"
        )
        match = False

    return {
        "results": results,
        "predicted": predicted[0],
        "actual": actual[0],
        "match": match,
    }


def main():
    if len(sys.argv) != 2:
        print(
            "Usage:\n"
            "python evaluate_cheek_reliability.py "
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

    matches = 0
    total = 0

    with (
        mp.tasks.vision.FaceLandmarker
        .create_from_options(options)
        as face_landmarker
    ):
        for image_path in image_paths:

            result = analyze_image(
                image_path,
                face_landmarker,
            )

            if result is None:
                continue

            total += 1

            if result["match"]:
                matches += 1

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(
        f"Images analyzed:     {total}"
    )

    print(
        f"Reliability matches:  {matches}"
    )

    if total:
        print(
            f"Prediction accuracy: "
            f"{matches / total * 100:.1f}%"
        )

    print(
        "\nInterpretation:"
    )

    print(
        "  High accuracy means the reliability heuristic"
    )

    print(
        "  may be useful for adaptive cheek selection."
    )

    print(
        "  Low accuracy means we should not use it"
    )

    print(
        "  to automatically choose a production region yet."
    )


if __name__ == "__main__":
    main()