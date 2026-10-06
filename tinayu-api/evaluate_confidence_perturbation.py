
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

# Small perturbations around the measured LAB a/b values.
#
# We intentionally keep these relatively small because the goal
# is to test whether the CONFIDENCE DIAGNOSTIC is stable under
# realistic measurement noise.
PERTURBATIONS = [
    (-2.0, 0.0),
    (-1.5, 0.0),
    (-1.0, 0.0),
    (-0.5, 0.0),
    (0.0, 0.0),
    (0.5, 0.0),
    (1.0, 0.0),
    (1.5, 0.0),
    (2.0, 0.0),

    (0.0, -2.0),
    (0.0, -1.5),
    (0.0, -1.0),
    (0.0, -0.5),
    (0.0, 0.5),
    (0.0, 1.0),
    (0.0, 1.5),
    (0.0, 2.0),

    (-1.0, -1.0),
    (-1.0, 1.0),
    (1.0, -1.0),
    (1.0, 1.0),

    (-2.0, -2.0),
    (-2.0, 2.0),
    (2.0, -2.0),
    (2.0, 2.0),
]


# ============================================================
# CLASSIFICATION CONFIDENCE
# ============================================================

def calculate_boundary_distance(score):
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
        return float(0.45 - score)

    if score > 0.55:
        return float(score - 0.55)

    return float(
        min(
            score - 0.45,
            0.55 - score,
        )
    )


def calculate_classification_confidence(score):
    """
    Diagnostic confidence from distance to the nearest
    temperature boundary.

    This is NOT probability.
    """

    distance = calculate_boundary_distance(
        score
    )

    return float(
        np.clip(
            distance / 0.10,
            0.0,
            1.0,
        )
    )


def classify_classification_confidence(score):
    """
    Confidence label.

    LOW:
        < 0.02 from boundary

    MEDIUM:
        0.02-0.05 from boundary

    HIGH:
        >= 0.05 from boundary
    """

    distance = calculate_boundary_distance(
        score
    )

    if distance < 0.02:
        return "LOW"

    if distance < 0.05:
        return "MEDIUM"

    return "HIGH"


# ============================================================
# CHEEK CONSISTENCY
# ============================================================

def calculate_cheek_consistency(
    left_score,
    right_score,
    left_classification,
    right_classification,
):
    """
    Evaluate left/right cheek consistency independently
    from overall classification confidence.
    """

    side_gap = abs(
        float(left_score)
        - float(right_score)
    )

    agreement = (
        left_classification
        == right_classification
    )

    if not agreement:
        label = "LOW"

    elif side_gap >= 0.10:
        label = "LOW"

    elif side_gap >= 0.05:
        label = "MEDIUM"

    else:
        label = "HIGH"

    consistency_score = np.clip(
        1.0 - (side_gap / 0.10),
        0.0,
        1.0,
    )

    return {
        "label": label,
        "score": float(consistency_score),
        "gap": float(side_gap),
        "agreement": bool(agreement),
    }


# ============================================================
# CHEEK EXTRACTION
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


def extract_cheek_lab(
    image_rgb,
    landmarks,
):
    """
    Extract the production left/right cheek regions
    and return their LAB measurements.
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

    def region_lab(polygon):
        mask = np.zeros(
            image_rgb.shape[:2],
            dtype=np.uint8,
        )

        cv2.fillPoly(
            mask,
            [polygon],
            255,
        )

        pixels = image_rgb[
            mask > 0
        ]

        if len(pixels) == 0:
            return None

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

        return {
            "rgb": mean_rgb,
            "lab": lab,
            "pixels": int(
                len(pixels)
            ),
        }

    return (
        region_lab(left_points),
        region_lab(right_points),
    )


# ============================================================
# PERTURBATION ANALYSIS
# ============================================================

def analyze_perturbation(
    base_left_lab,
    base_right_lab,
    delta_a,
    delta_b,
):
    """
    Apply the same LAB a/b perturbation to both cheeks.

    This isolates how measurement noise affects the
    confidence diagnostic.
    """

    left_lab = list(
        base_left_lab
    )

    right_lab = list(
        base_right_lab
    )

    left_lab[1] += delta_a
    left_lab[2] += delta_b

    right_lab[1] += delta_a
    right_lab[2] += delta_b

    left_score = temperature_score(
        left_lab[1],
        left_lab[2],
    )

    right_score = temperature_score(
        right_lab[1],
        right_lab[2],
    )

    left_classification = (
        classify_temperature_stable(
            left_lab[1],
            left_lab[2],
        )
    )

    right_classification = (
        classify_temperature_stable(
            right_lab[1],
            right_lab[2],
        )
    )

    # Combined production-like signal:
    #
    # The actual production extraction calculates one
    # combined cheek RGB mean. For this diagnostic,
    # we approximate the combined signal by averaging
    # the two side LAB a/b values.
    combined_a = (
        left_lab[1]
        + right_lab[1]
    ) / 2.0

    combined_b = (
        left_lab[2]
        + right_lab[2]
    ) / 2.0

    combined_score = temperature_score(
        combined_a,
        combined_b,
    )

    combined_classification = (
        classify_temperature_stable(
            combined_a,
            combined_b,
        )
    )

    classification_confidence = (
        calculate_classification_confidence(
            combined_score
        )
    )

    classification_label = (
        classify_classification_confidence(
            combined_score
        )
    )

    cheek_consistency = (
        calculate_cheek_consistency(
            left_score,
            right_score,
            left_classification,
            right_classification,
        )
    )

    return {
        "score": float(
            combined_score
        ),
        "classification": (
            combined_classification
        ),
        "classification_confidence": (
            classification_confidence
        ),
        "classification_label": (
            classification_label
        ),
        "cheek_consistency": (
            cheek_consistency
        ),
    }


# ============================================================
# IMAGE ANALYSIS
# ============================================================

def analyze_image(
    image_path
):
    """
    Extract baseline cheek LAB values and run
    the complete perturbation grid.
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

    # Production combined skin measurement.
    skin_rgb = extract_skin_color(
        image_rgb,
        landmarks,
    )

    skin_lab = rgb_to_lab(
        skin_rgb
    )

    baseline_score = temperature_score(
        skin_lab[1],
        skin_lab[2],
    )

    baseline_classification = (
        classify_temperature_stable(
            skin_lab[1],
            skin_lab[2],
        )
    )

    baseline_confidence = (
        calculate_classification_confidence(
            baseline_score
        )
    )

    baseline_confidence_label = (
        classify_classification_confidence(
            baseline_score
        )
    )

    # Independent cheek measurements.
    left, right = extract_cheek_lab(
        image_rgb,
        landmarks,
    )

    if left is None or right is None:
        raise ValueError(
            "Unable to extract one or both cheek regions."
        )

    left_score = temperature_score(
        left["lab"][1],
        left["lab"][2],
    )

    right_score = temperature_score(
        right["lab"][1],
        right["lab"][2],
    )

    left_classification = (
        classify_temperature_stable(
            left["lab"][1],
            left["lab"][2],
        )
    )

    right_classification = (
        classify_temperature_stable(
            right["lab"][1],
            right["lab"][2],
        )
    )

    baseline_consistency = (
        calculate_cheek_consistency(
            left_score,
            right_score,
            left_classification,
            right_classification,
        )
    )

    perturbation_results = []

    for delta_a, delta_b in PERTURBATIONS:
        result = analyze_perturbation(
            base_left_lab=left["lab"],
            base_right_lab=right["lab"],
            delta_a=delta_a,
            delta_b=delta_b,
        )

        result["delta_a"] = delta_a
        result["delta_b"] = delta_b

        perturbation_results.append(
            result
        )

    return {
        "baseline_score": float(
            baseline_score
        ),
        "baseline_classification": (
            baseline_classification
        ),
        "baseline_confidence": float(
            baseline_confidence
        ),
        "baseline_confidence_label": (
            baseline_confidence_label
        ),
        "baseline_consistency": (
            baseline_consistency
        ),
        "left": left,
        "right": right,
        "perturbations": (
            perturbation_results
        ),
    }


# ============================================================
# PRINT IMAGE RESULTS
# ============================================================

def print_result(
    image_path,
    result,
):
    perturbations = result[
        "perturbations"
    ]

    baseline_consistency = result[
        "baseline_consistency"
    ]

    scores = [
        p["score"]
        for p in perturbations
    ]

    confidence_values = [
        p[
            "classification_confidence"
        ]
        for p in perturbations
    ]

    classification_labels = [
        p[
            "classification_label"
        ]
        for p in perturbations
    ]

    cheek_labels = [
        p[
            "cheek_consistency"
        ]["label"]
        for p in perturbations
    ]

    baseline_classification = (
        result[
            "baseline_classification"
        ]
    )

    classification_flips = sum(
        p["classification"]
        != baseline_classification
        for p in perturbations
    )

    confidence_flips = sum(
        p[
            "classification_label"
        ]
        != result[
            "baseline_confidence_label"
        ]
        for p in perturbations
    )

    cheek_confidence_flips = sum(
        p[
            "cheek_consistency"
        ]["label"]
        != baseline_consistency["label"]
        for p in perturbations
    )

    print()
    print("=" * 70)
    print(image_path.name)
    print("=" * 70)

    print(
        f"Baseline score:               "
        f"{result['baseline_score']:.4f}"
    )

    print(
        f"Baseline classification:      "
        f"{result['baseline_classification']}"
    )

    print(
        f"Baseline class confidence:    "
        f"{result['baseline_confidence']:.2f}"
    )

    print(
        f"Baseline confidence label:    "
        f"{result['baseline_confidence_label']}"
    )

    print(
        f"Baseline cheek consistency:    "
        f"{baseline_consistency['label']}"
    )

    print()

    print(
        f"Perturbed score range:        "
        f"{min(scores):.4f} - "
        f"{max(scores):.4f}"
    )

    print(
        f"Perturbed confidence range:   "
        f"{min(confidence_values):.2f} - "
        f"{max(confidence_values):.2f}"
    )

    print(
        f"Classification flips:          "
        f"{classification_flips}/"
        f"{len(perturbations)}"
    )

    print(
        f"Confidence-label flips:        "
        f"{confidence_flips}/"
        f"{len(perturbations)}"
    )

    print(
        f"Cheek-label flips:             "
        f"{cheek_confidence_flips}/"
        f"{len(perturbations)}"
    )

    print()

    print(
        "Confidence label distribution:"
    )

    for label in [
        "LOW",
        "MEDIUM",
        "HIGH",
    ]:
        count = sum(
            value == label
            for value in classification_labels
        )

        print(
            f"  {label:<6}: {count}"
        )

    print()

    print(
        "Cheek consistency distribution:"
    )

    for label in [
        "LOW",
        "MEDIUM",
        "HIGH",
    ]:
        count = sum(
            value == label
            for value in cheek_labels
        )

        print(
            f"  {label:<6}: {count}"
        )


# ============================================================
# SUMMARY
# ============================================================

def print_summary(
    results
):
    print()
    print()
    print("=" * 70)
    print("GLOBAL SUMMARY")
    print("=" * 70)

    total_images = len(
        results
    )

    total_tests = (
        total_images
        * len(PERTURBATIONS)
    )

    classification_flips = 0
    confidence_flips = 0
    cheek_flips = 0

    for result in results:
        baseline_classification = (
            result[
                "baseline_classification"
            ]
        )

        baseline_confidence_label = (
            result[
                "baseline_confidence_label"
            ]
        )

        baseline_cheek_label = (
            result[
                "baseline_consistency"
            ]["label"]
        )

        for perturbation in result[
            "perturbations"
        ]:

            if (
                perturbation[
                    "classification"
                ]
                != baseline_classification
            ):
                classification_flips += 1

            if (
                perturbation[
                    "classification_label"
                ]
                != baseline_confidence_label
            ):
                confidence_flips += 1

            if (
                perturbation[
                    "cheek_consistency"
                ]["label"]
                != baseline_cheek_label
            ):
                cheek_flips += 1

    print(
        f"Images tested:                "
        f"{total_images}"
    )

    print(
        f"Perturbations per image:      "
        f"{len(PERTURBATIONS)}"
    )

    print(
        f"Total perturbation tests:      "
        f"{total_tests}"
    )

    print()

    print(
        f"Classification flips:          "
        f"{classification_flips}/"
        f"{total_tests} "
        f"("
        f"{classification_flips / total_tests * 100:.2f}%"
        f")"
    )

    print(
        f"Confidence-label flips:        "
        f"{confidence_flips}/"
        f"{total_tests} "
        f"("
        f"{confidence_flips / total_tests * 100:.2f}%"
        f")"
    )

    print(
        f"Cheek-label flips:             "
        f"{cheek_flips}/"
        f"{total_tests} "
        f"("
        f"{cheek_flips / total_tests * 100:.2f}%"
        f")"
    )

    print()

    print(
        "Baseline confidence labels:"
    )

    for label in [
        "LOW",
        "MEDIUM",
        "HIGH",
    ]:
        count = sum(
            result[
                "baseline_confidence_label"
            ]
            == label
            for result in results
        )

        print(
            f"  {label:<6}: {count}"
        )

    print()

    print(
        "Baseline cheek consistency:"
    )

    for label in [
        "LOW",
        "MEDIUM",
        "HIGH",
    ]:
        count = sum(
            result[
                "baseline_consistency"
            ]["label"]
            == label
            for result in results
        )

        print(
            f"  {label:<6}: {count}"
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
            "evaluate_confidence_perturbation.py "
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
