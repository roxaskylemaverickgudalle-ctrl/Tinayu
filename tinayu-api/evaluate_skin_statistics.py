
from pathlib import Path
import sys

import cv2
import mediapipe as mp
import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parent))

from tinayu_engine import (
    landmarks_to_pixels,
    polygon_mask,
    temperature_score,
)


VALIDATION_DIR = Path("validation_samples")


SAMPLE_NAMES = {
    "bright_indoor": "Bright indoor",
    "different_camera": "Different camera",
    "outdoor": "Outdoor",
    "slightly_different_angle": "Slightly different angle",
    "without_makeup": "Without makeup",
    "normal_indoor": "Normal indoor",
    "makeover": "Makeover",
}


LEFT_INDICES = [
    50,
    101,
    118,
    119,
    100,
    47,
]


RIGHT_INDICES = [
    280,
    330,
    347,
    348,
    329,
    277,
]


def classify(score):
    if score >= 0.55:
        return "Warm"

    if score <= 0.45:
        return "Cool"

    return "Neutral"


def find_images():
    extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
    }

    if not VALIDATION_DIR.exists():
        return []

    return sorted(
        path
        for path in VALIDATION_DIR.rglob("*")
        if path.is_file()
        and path.suffix.lower() in extensions
    )


def match_sample_name(path):
    stem = path.stem.lower()

    normalized = (
        stem
        .replace("-", "_")
        .replace(" ", "_")
    )

    for key, display_name in SAMPLE_NAMES.items():
        if key in normalized:
            return display_name

    return path.stem


def get_face_landmarks(image_rgb, landmarker):
    """
    Run the same MediaPipe Face Landmarker family used by Tinayu.

    This evaluator only needs the face landmark coordinates.
    """

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=image_rgb,
    )

    result = landmarker.detect(
        mp_image
    )

    if not result.face_landmarks:
        raise ValueError(
            "No face landmarks detected."
        )

    return result.face_landmarks[0]


def get_cheek_pixels(image_rgb, landmarks):
    """
    Reproduce extract_skin_color() exactly up to the
    point where production currently calculates np.mean().
    """

    points = landmarks_to_pixels(
        landmarks,
        image_rgb.shape,
    )

    left_points = points[
        LEFT_INDICES
    ]

    right_points = points[
        RIGHT_INDICES
    ]

    left_mask = polygon_mask(
        image_rgb.shape,
        left_points,
    )

    right_mask = polygon_mask(
        image_rgb.shape,
        right_points,
    )

    combined_mask = cv2.bitwise_or(
        left_mask,
        right_mask,
    )

    pixels = image_rgb[
        combined_mask > 0
    ]

    if len(pixels) == 0:
        raise ValueError(
            "No cheek pixels extracted."
        )

    return pixels


def rgb_to_lab_pixels(pixels):
    """
    Convert RGB pixels to OpenCV LAB and then into the
    same approximate L/a/b coordinate convention used
    by the engine.
    """

    rgb_uint8 = np.clip(
        np.round(pixels),
        0,
        255,
    ).astype(np.uint8)

    rgb_image = rgb_uint8.reshape(
        -1,
        1,
        3,
    )

    bgr_image = cv2.cvtColor(
        rgb_image,
        cv2.COLOR_RGB2BGR,
    )

    lab_image = cv2.cvtColor(
        bgr_image,
        cv2.COLOR_BGR2LAB,
    )

    lab = lab_image.reshape(
        -1,
        3,
    ).astype(np.float32)

    lab[:, 0] = (
        lab[:, 0] * 100.0 / 255.0
    )

    lab[:, 1] -= 128.0
    lab[:, 2] -= 128.0

    return lab


def lab_from_rgb(rgb):
    """
    Convert one representative RGB value to LAB.
    """

    pixels = np.asarray(
        rgb,
        dtype=np.float32,
    ).reshape(
        1,
        3,
    )

    return np.mean(
        rgb_to_lab_pixels(pixels),
        axis=0,
    )


def score_from_lab(lab):
    return temperature_score(
        float(lab[1]),
        float(lab[2]),
    )


def trimmed_pixels(pixels, percent=10):
    """
    Remove pixels outside the per-channel 10th–90th
    percentile range.

    Evaluator only.
    """

    low = np.percentile(
        pixels,
        percent,
        axis=0,
    )

    high = np.percentile(
        pixels,
        100 - percent,
        axis=0,
    )

    mask = np.all(
        (pixels >= low)
        & (pixels <= high),
        axis=1,
    )

    result = pixels[mask]

    if len(result) == 0:
        return pixels

    return result


def calculate_statistics(pixels):
    """
    Calculate five representative skin statistics
    from the exact same cheek mask.
    """

    # ---------------------------------------------------------------
    # 1. Current production method
    # ---------------------------------------------------------------

    mean_rgb = np.mean(
        pixels,
        axis=0,
    )

    mean_lab = lab_from_rgb(
        mean_rgb
    )

    # ---------------------------------------------------------------
    # 2. Median RGB
    # ---------------------------------------------------------------

    median_rgb = np.median(
        pixels,
        axis=0,
    )

    median_rgb_lab = lab_from_rgb(
        median_rgb
    )

    # ---------------------------------------------------------------
    # 3. Trimmed RGB mean
    # ---------------------------------------------------------------

    trimmed_rgb_pixels = trimmed_pixels(
        pixels
    )

    trimmed_rgb = np.mean(
        trimmed_rgb_pixels,
        axis=0,
    )

    trimmed_rgb_lab = lab_from_rgb(
        trimmed_rgb
    )

    # ---------------------------------------------------------------
    # 4. Direct median LAB
    # ---------------------------------------------------------------

    lab_pixels = rgb_to_lab_pixels(
        pixels
    )

    median_lab = np.median(
        lab_pixels,
        axis=0,
    )

    # ---------------------------------------------------------------
    # 5. Direct trimmed LAB
    # ---------------------------------------------------------------

    trimmed_lab_pixels = trimmed_pixels(
        lab_pixels
    )

    trimmed_lab = np.mean(
        trimmed_lab_pixels,
        axis=0,
    )

    return {
        "mean_rgb": mean_rgb,
        "mean_lab": mean_lab,
        "mean_score": score_from_lab(
            mean_lab
        ),

        "median_rgb": median_rgb,
        "median_rgb_lab": median_rgb_lab,
        "median_rgb_score": score_from_lab(
            median_rgb_lab
        ),

        "trimmed_rgb": trimmed_rgb,
        "trimmed_rgb_lab": trimmed_rgb_lab,
        "trimmed_rgb_score": score_from_lab(
            trimmed_rgb_lab
        ),

        "median_lab": median_lab,
        "median_lab_score": score_from_lab(
            median_lab
        ),

        "trimmed_lab": trimmed_lab,
        "trimmed_lab_score": score_from_lab(
            trimmed_lab
        ),

        "trimmed_rgb_count": len(
            trimmed_rgb_pixels
        ),

        "trimmed_lab_count": len(
            trimmed_lab_pixels
        ),
    }


def main():
    print("=" * 105)
    print(
        "TINAYU REAL CHEEK-PIXEL STATISTICS DIAGNOSTIC"
    )
    print("=" * 105)

    print(
        "\nThis reproduces the production cheek mask and compares "
        "different statistics over the SAME extracted pixels."
    )

    print(
        "\nProduction behavior remains unchanged."
    )

    image_paths = find_images()

    if not image_paths:
        print(
            f"\nERROR: No images found under "
            f"{VALIDATION_DIR}"
        )
        return

    # ---------------------------------------------------------------
    # MediaPipe setup
    # ---------------------------------------------------------------

    model_path = (
        Path("face_landmarker.task")
    )

    if not model_path.exists():
        print(
            "\nERROR: face_landmarker.task was not found."
        )

        print(
            "Run:"
        )

        print(
            "  ls *.task"
        )

        print(
            "and paste the result here."
        )

        return

    base_options = (
        mp.tasks.BaseOptions(
            model_asset_path=str(
                model_path
            )
        )
    )

    options = (
        mp.tasks.vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=(
                mp.tasks.vision.RunningMode.IMAGE
            ),
            num_faces=1,
        )
    )

    all_rows = []

    with mp.tasks.vision.FaceLandmarker.create_from_options(
        options
    ) as landmarker:

        for image_path in image_paths:
            sample_name = match_sample_name(
                image_path
            )

            print(
                f"\nProcessing: {sample_name}"
            )

            image_bgr = cv2.imread(
                str(image_path)
            )

            if image_bgr is None:
                print(
                    "  ERROR: Could not read image."
                )
                continue

            image_rgb = cv2.cvtColor(
                image_bgr,
                cv2.COLOR_BGR2RGB,
            )

            try:
                landmarks = get_face_landmarks(
                    image_rgb,
                    landmarker,
                )

                pixels = get_cheek_pixels(
                    image_rgb,
                    landmarks,
                )

                stats = calculate_statistics(
                    pixels
                )

                stats["sample"] = sample_name
                stats["pixels"] = len(pixels)

                all_rows.append(
                    stats
                )

                print(
                    f"  cheek pixels: "
                    f"{len(pixels)}"
                )

            except Exception as exc:
                print(
                    f"  ERROR: "
                    f"{type(exc).__name__}: {exc}"
                )

    # ---------------------------------------------------------------
    # Score comparison
    # ---------------------------------------------------------------

    print("\n" + "=" * 105)
    print("TEMPERATURE SCORE BY STATISTIC")
    print("=" * 105)

    for row in all_rows:
        print(
            f"\n{row['sample']}"
        )

        print(
            f"  Mean RGB       "
            f"{row['mean_score']:.4f} "
            f"{classify(row['mean_score'])}"
        )

        print(
            f"  Median RGB     "
            f"{row['median_rgb_score']:.4f} "
            f"{classify(row['median_rgb_score'])}"
        )

        print(
            f"  Trimmed RGB    "
            f"{row['trimmed_rgb_score']:.4f} "
            f"{classify(row['trimmed_rgb_score'])}"
        )

        print(
            f"  Median LAB     "
            f"{row['median_lab_score']:.4f} "
            f"{classify(row['median_lab_score'])}"
        )

        print(
            f"  Trimmed LAB    "
            f"{row['trimmed_lab_score']:.4f} "
            f"{classify(row['trimmed_lab_score'])}"
        )

    # ---------------------------------------------------------------
    # RGB comparison
    # ---------------------------------------------------------------

    print("\n" + "=" * 105)
    print("REPRESENTATIVE RGB VALUES")
    print("=" * 105)

    for row in all_rows:
        print(
            f"\n{row['sample']}"
        )

        print(
            f"  Mean RGB:    "
            f"{np.round(row['mean_rgb'], 1)}"
        )

        print(
            f"  Median RGB:  "
            f"{np.round(row['median_rgb'], 1)}"
        )

        print(
            f"  Trimmed RGB: "
            f"{np.round(row['trimmed_rgb'], 1)}"
        )

    # ---------------------------------------------------------------
    # LAB comparison
    # ---------------------------------------------------------------

    print("\n" + "=" * 105)
    print("REPRESENTATIVE LAB VALUES")
    print("=" * 105)

    for row in all_rows:
        print(
            f"\n{row['sample']}"
        )

        print(
            f"  Mean RGB → LAB: "
            f"{np.round(row['mean_lab'], 2)}"
        )

        print(
            f"  Median RGB → LAB: "
            f"{np.round(row['median_rgb_lab'], 2)}"
        )

        print(
            f"  Trimmed RGB → LAB: "
            f"{np.round(row['trimmed_rgb_lab'], 2)}"
        )

        print(
            f"  Median LAB: "
            f"{np.round(row['median_lab'], 2)}"
        )

        print(
            f"  Trimmed LAB: "
            f"{np.round(row['trimmed_lab'], 2)}"
        )

    # ---------------------------------------------------------------
    # Spread
    # ---------------------------------------------------------------

    print("\n" + "=" * 105)
    print("STATISTIC-TO-STATISTIC TEMPERATURE SPREAD")
    print("=" * 105)

    score_keys = [
        "mean_score",
        "median_rgb_score",
        "trimmed_rgb_score",
        "median_lab_score",
        "trimmed_lab_score",
    ]

    for row in all_rows:
        scores = [
            row[key]
            for key in score_keys
        ]

        minimum = min(scores)
        maximum = max(scores)

        print(
            f"{row['sample']:<28} "
            f"min={minimum:.4f} "
            f"max={maximum:.4f} "
            f"spread={maximum - minimum:.4f}"
        )

    # ---------------------------------------------------------------
    # Agreement
    # ---------------------------------------------------------------

    print("\n" + "=" * 105)
    print("CLASSIFICATION AGREEMENT WITH PRODUCTION MEAN")
    print("=" * 105)

    disagreements = 0
    comparisons = 0

    for row in all_rows:
        production_class = classify(
            row["mean_score"]
        )

        for key in score_keys[1:]:
            alternative_class = classify(
                row[key]
            )

            comparisons += 1

            if alternative_class != production_class:
                disagreements += 1

                print(
                    f"{row['sample']:<28} "
                    f"production={production_class:<7} "
                    f"alternative={alternative_class:<7} "
                    f"score={row[key]:.4f}"
                )

    if comparisons:
        print(
            f"\nDisagreements: "
            f"{disagreements}/{comparisons} "
            f"({disagreements / comparisons * 100:.1f}%)"
        )

    # ---------------------------------------------------------------
    # Final diagnostic
    # ---------------------------------------------------------------

    print("\n" + "=" * 105)
    print("DIAGNOSTIC")
    print("=" * 105)

    print(
        """
Interpretation:

1. If all statistics produce nearly identical scores,
   the current mean is probably not the main issue.

2. If median/trimmed statistics move borderline samples
   substantially while strong-Warm samples stay stable,
   cheek-pixel variation is contributing to uncertainty.

3. If an alternative statistic consistently improves the
   borderline signal without changing strong samples,
   it becomes a candidate for a controlled production experiment.

No production code was changed.
"""
    )


if __name__ == "__main__":
    main()


