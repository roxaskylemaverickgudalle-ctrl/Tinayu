
import sys
from pathlib import Path

import cv2
import numpy as np

from tinayu_engine import (
    load_image,
    detect_face,
    detect_landmarks,
    extract_skin_color,
    rgb_to_lab,
    temperature_score,
    classify_temperature_stable,
)


# ============================================================
# CONFIGURATION
# ============================================================

# Small RGB perturbations applied to the ACTUAL cheek pixels.
#
# Each perturbation is applied equally to R/G/B so that we can
# test how small overall measurement changes affect the exact
# production RGB -> LAB -> temperature pipeline.
#
# This is deliberately conservative.
RGB_PERTURBATIONS = [
    (-4, -4, -4),
    (-3, -3, -3),
    (-2, -2, -2),
    (-1, -1, -1),
    (0, 0, 0),
    (1, 1, 1),
    (2, 2, 2),
    (3, 3, 3),
    (4, 4, 4),
]


# ============================================================
# LANDMARK HELPERS
# ============================================================

def landmarks_to_pixel_points(
    image_rgb,
    landmarks,
):
    """
    Convert MediaPipe NormalizedLandmark objects
    into pixel coordinates.
    """

    height, width = image_rgb.shape[:2]

    return np.array(
        [
            [
                int(landmark.x * width),
                int(landmark.y * height),
            ]
            for landmark in landmarks
        ],
        dtype=np.int32,
    )


# ============================================================
# PRODUCTION CHEEK PIXELS
# ============================================================

def get_production_cheek_pixels(
    image_rgb,
    landmarks,
):
    """
    Reproduce the production cheek masks and return
    the actual RGB pixels used by the production extractor.
    """

    points = landmarks_to_pixel_points(
        image_rgb,
        landmarks,
    )

    left_indices = [
        50,
        101,
        118,
        119,
        100,
        47,
    ]

    right_indices = [
        280,
        330,
        347,
        348,
        329,
        277,
    ]

    left_points = points[
        left_indices
    ]

    right_points = points[
        right_indices
    ]

    left_mask = np.zeros(
        image_rgb.shape[:2],
        dtype=np.uint8,
    )

    right_mask = np.zeros(
        image_rgb.shape[:2],
        dtype=np.uint8,
    )

    cv2.fillPoly(
        left_mask,
        [left_points],
        255,
    )

    cv2.fillPoly(
        right_mask,
        [right_points],
        255,
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
            "Unable to extract production cheek pixels."
        )

    return pixels


# ============================================================
# TEMPERATURE CLASSIFICATION
# ============================================================

def calculate_boundary_distance(
    score
):
    """
    Distance from the nearest production temperature
    decision boundary.

    Production boundaries:

        <= 0.45 -> Cool
        0.45-0.55 -> Neutral
        >= 0.55 -> Warm
    """

    score = float(score)

    if score < 0.45:
        return float(
            0.45 - score
        )

    if score > 0.55:
        return float(
            score - 0.55
        )

    return float(
        min(
            score - 0.45,
            0.55 - score,
        )
    )


def calculate_confidence(
    score
):
    """
    Diagnostic stability score.

    This is NOT probability.

    0.00 = directly on a decision boundary
    1.00 = at least 0.10 away from a boundary
    """

    distance = (
        calculate_boundary_distance(
            score
        )
    )

    confidence = np.clip(
        distance / 0.10,
        0.0,
        1.0,
    )

    return float(
        confidence
    )


def classify_confidence(
    score
):
    """
    Two-level confidence diagnostic.

    HIGH:
        >= 0.05 from boundary

    BORDERLINE:
        < 0.05 from boundary

    This is intentionally simpler than the previous
    LOW/MEDIUM/HIGH system.
    """

    distance = (
        calculate_boundary_distance(
            score
        )
    )

    if distance < 0.05:
        return "BORDERLINE"

    return "HIGH"


# ============================================================
# EXACT PRODUCTION MEASUREMENT
# ============================================================

def calculate_production_temperature(
    pixels
):
    """
    Reproduce the production extraction pipeline:

        cheek pixels
            ↓
        mean RGB
            ↓
        LAB
            ↓
        temperature score
            ↓
        temperature classification
    """

    mean_rgb = (
        np.mean(
            pixels,
            axis=0,
        )
        .astype(int)
        .tolist()
    )

    lab = rgb_to_lab(
        mean_rgb
    )

    score = temperature_score(
        lab[1],
        lab[2],
    )

    classification = (
        classify_temperature_stable(
            lab[1],
            lab[2],
        )
    )

    return {
        "rgb": mean_rgb,
        "lab": lab,
        "score": float(
            score
        ),
        "classification": (
            classification
        ),
        "confidence": (
            calculate_confidence(
                score
            )
        ),
        "confidence_label": (
            classify_confidence(
                score
            )
        ),
    }


# ============================================================
# PERTURBATION
# ============================================================

def perturb_pixels(
    pixels,
    delta_r,
    delta_g,
    delta_b,
):
    """
    Apply a small RGB perturbation to the actual
    production cheek pixels.

    Values are clipped to valid RGB range.
    """

    perturbed = (
        pixels.astype(
            np.float32
        ).copy()
    )

    perturbed[:, 0] += delta_r
    perturbed[:, 1] += delta_g
    perturbed[:, 2] += delta_b

    perturbed = np.clip(
        perturbed,
        0,
        255,
    )

    return perturbed


# ============================================================
# IMAGE ANALYSIS
# ============================================================

def analyze_image(
    image_path
):
    """
    Analyze one image using the exact production
    cheek-pixel pipeline.
    """

    image_rgb = load_image(
        str(image_path)
    )

    detect_face(
        image_rgb
    )

    landmarks = detect_landmarks(
        image_rgb
    )

    # Confirm that the production extractor itself
    # succeeds on this image.
    production_skin_rgb = (
        extract_skin_color(
            image_rgb,
            landmarks,
        )
    )

    production_skin_lab = (
        rgb_to_lab(
            production_skin_rgb
        )
    )

    production_score = (
        temperature_score(
            production_skin_lab[1],
            production_skin_lab[2],
        )
    )

    production_classification = (
        classify_temperature_stable(
            production_skin_lab[1],
            production_skin_lab[2],
        )
    )

    production_confidence = (
        calculate_confidence(
            production_score
        )
    )

    production_confidence_label = (
        classify_confidence(
            production_score
        )
    )

    # Get the exact pixels used by production.
    pixels = (
        get_production_cheek_pixels(
            image_rgb,
            landmarks,
        )
    )

    # Verify the evaluator's reconstruction matches
    # extract_skin_color().
    reconstructed = (
        np.mean(
            pixels,
            axis=0,
        )
        .astype(int)
        .tolist()
    )

    reconstruction_matches = (
        reconstructed
        == production_skin_rgb
    )

    baseline = (
        calculate_production_temperature(
            pixels
        )
    )

    perturbations = []

    for (
        delta_r,
        delta_g,
        delta_b,
    ) in RGB_PERTURBATIONS:

        perturbed_pixels = (
            perturb_pixels(
                pixels,
                delta_r,
                delta_g,
                delta_b,
            )
        )

        result = (
            calculate_production_temperature(
                perturbed_pixels
            )
        )

        result[
            "delta_r"
        ] = delta_r

        result[
            "delta_g"
        ] = delta_g

        result[
            "delta_b"
        ] = delta_b

        perturbations.append(
            result
        )

    return {
        "production_skin_rgb": (
            production_skin_rgb
        ),
        "production_skin_lab": (
            production_skin_lab
        ),
        "production_score": float(
            production_score
        ),
        "production_classification": (
            production_classification
        ),
        "production_confidence": float(
            production_confidence
        ),
        "production_confidence_label": (
            production_confidence_label
        ),
        "pixel_count": int(
            len(pixels)
        ),
        "reconstruction_matches": (
            reconstruction_matches
        ),
        "baseline": baseline,
        "perturbations": perturbations,
    }


# ============================================================
# PRINT IMAGE RESULT
# ============================================================

def print_result(
    image_path,
    result,
):
    baseline = result[
        "baseline"
    ]

    perturbations = result[
        "perturbations"
    ]

    scores = [
        item["score"]
        for item in perturbations
    ]

    confidences = [
        item["confidence"]
        for item in perturbations
    ]

    baseline_classification = (
        baseline[
            "classification"
        ]
    )

    baseline_confidence_label = (
        baseline[
            "confidence_label"
        ]
    )

    classification_flips = sum(
        item["classification"]
        != baseline_classification
        for item in perturbations
    )

    confidence_label_flips = sum(
        item["confidence_label"]
        != baseline_confidence_label
        for item in perturbations
    )

    print()
    print("=" * 70)
    print(image_path.name)
    print("=" * 70)

    print(
        f"Production RGB:             "
        f"{result['production_skin_rgb']}"
    )

    print(
        f"Production LAB:             "
        f"{result['production_skin_lab']}"
    )

    print(
        f"Production pixels:          "
        f"{result['pixel_count']}"
    )

    print(
        f"Pixel reconstruction:       "
        f"{'MATCH' if result['reconstruction_matches'] else 'MISMATCH'}"
    )

    print()

    print(
        f"Baseline score:              "
        f"{baseline['score']:.4f}"
    )

    print(
        f"Baseline classification:     "
        f"{baseline['classification']}"
    )

    print(
        f"Boundary distance:           "
        f"{calculate_boundary_distance(baseline['score']):.4f}"
    )

    print(
        f"Confidence score:             "
        f"{baseline['confidence']:.2f}"
    )

    print(
        f"Confidence label:             "
        f"{baseline['confidence_label']}"
    )

    print()

    print(
        f"Perturbed score range:        "
        f"{min(scores):.4f} - "
        f"{max(scores):.4f}"
    )

    print(
        f"Perturbed confidence range:   "
        f"{min(confidences):.2f} - "
        f"{max(confidences):.2f}"
    )

    print(
        f"Classification flips:          "
        f"{classification_flips}/"
        f"{len(perturbations)}"
    )

    print(
        f"Confidence-label flips:        "
        f"{confidence_label_flips}/"
        f"{len(perturbations)}"
    )

    print()

    print(
        "Perturbation results:"
    )

    for item in perturbations:
        print(
            f"  RGB "
            f"({item['delta_r']:+d},"
            f"{item['delta_g']:+d},"
            f"{item['delta_b']:+d}) "
            f"score={item['score']:.4f} "
            f"class={item['classification']:<7} "
            f"confidence="
            f"{item['confidence_label']}"
        )


# ============================================================
# SUMMARY
# ============================================================

def print_summary(
    results
):
    total_images = len(
        results
    )

    total_tests = (
        total_images
        * len(RGB_PERTURBATIONS)
    )

    classification_flips = 0
    confidence_flips = 0
    reconstruction_failures = 0

    for result in results:
        if not result[
            "reconstruction_matches"
        ]:
            reconstruction_failures += 1

        baseline = result[
            "baseline"
        ]

        for item in result[
            "perturbations"
        ]:

            if (
                item["classification"]
                != baseline[
                    "classification"
                ]
            ):
                classification_flips += 1

            if (
                item["confidence_label"]
                != baseline[
                    "confidence_label"
                ]
            ):
                confidence_flips += 1

    print()
    print()
    print("=" * 70)
    print("GLOBAL SUMMARY")
    print("=" * 70)

    print(
        f"Images tested:                 "
        f"{total_images}"
    )

    print(
        f"Perturbations per image:       "
        f"{len(RGB_PERTURBATIONS)}"
    )

    print(
        f"Total perturbation tests:       "
        f"{total_tests}"
    )

    print()

    print(
        f"Production reconstruction "
        f"failures:                       "
        f"{reconstruction_failures}"
    )

    print()

    print(
        f"Classification flips:           "
        f"{classification_flips}/"
        f"{total_tests} "
        f"("
        f"{classification_flips / total_tests * 100:.2f}%"
        f")"
    )

    print(
        f"Confidence-label flips:         "
        f"{confidence_flips}/"
        f"{total_tests} "
        f"("
        f"{confidence_flips / total_tests * 100:.2f}%"
        f")"
    )

    print()

    print(
        "Baseline confidence labels:"
    )

    for label in [
        "BORDERLINE",
        "HIGH",
    ]:
        count = sum(
            result[
                "baseline"
            ][
                "confidence_label"
            ]
            == label
            for result in results
        )

        print(
            f"  {label:<10}: {count}"
        )


# ============================================================
# MAIN
# ============================================================

def main():
    if len(sys.argv) < 2:
        print(
            "Usage:"
        )

        print(
            "  python "
            "evaluate_exact_confidence_perturbation.py "
            "validation_samples"
        )

        return

    folder = Path(
        sys.argv[1]
    )

    if not folder.exists():
        print(
            f"Folder not found: {folder}"
        )

        return

    image_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
    }

    image_paths = sorted(
        [
            path
            for path in folder.iterdir()
            if path.is_file()
            and path.suffix.lower()
            in image_extensions
        ]
    )

    if not image_paths:
        print(
            f"No images found in {folder}"
        )

        return

    results = []

    for image_path in image_paths:
        try:
            result = analyze_image(
                image_path
            )

            print_result(
                image_path,
                result,
            )

            results.append(
                {
                    "name": image_path.name,
                    **result,
                }
            )

        except Exception as exc:
            print()
            print("=" * 70)
            print(image_path.name)
            print("=" * 70)
            print(
                f"ERROR: {exc}"
            )

    if not results:
        print()
        print(
            "No successful results."
        )

        return

    print_summary(
        results
    )


if __name__ == "__main__":
    main()
