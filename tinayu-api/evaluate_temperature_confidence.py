
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
    classify_temperature,
)


# ============================================================
# LANDMARK / CHEEK HELPERS
# ============================================================

def landmarks_to_pixel_points(image_rgb, landmarks):
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


def extract_cheek_statistics(image_rgb, landmarks):
    """
    Reproduce the production left/right cheek regions
    and calculate temperature independently for each side.
    """

    points = landmarks_to_pixel_points(
        image_rgb,
        landmarks,
    )

    left_indices = [50, 101, 118, 119, 100, 47]
    right_indices = [280, 330, 347, 348, 329, 277]

    left_points = points[left_indices]
    right_points = points[right_indices]

    def analyze_region(polygon):
        mask = np.zeros(
            image_rgb.shape[:2],
            dtype=np.uint8,
        )

        cv2.fillPoly(
            mask,
            [polygon],
            255,
        )

        pixels = image_rgb[mask > 0]

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

        lab = rgb_to_lab(mean_rgb)

        score = temperature_score(
            lab[1],
            lab[2],
        )

        classification = classify_temperature_stable(
            lab[1],
            lab[2],
        )

        return {
            "rgb": mean_rgb,
            "lab": lab,
            "score": float(score),
            "classification": classification,
            "pixels": int(len(pixels)),
        }

    return (
        analyze_region(left_points),
        analyze_region(right_points),
    )


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

    Larger distance means stronger classification evidence.
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
    Convert boundary distance into a diagnostic 0-1 score.

    This is NOT probability.

    0.00 = directly on a boundary
    1.00 = at least 0.10 away from the nearest boundary
    """

    distance = calculate_boundary_distance(
        score
    )

    confidence = np.clip(
        distance / 0.10,
        0.0,
        1.0,
    )

    return float(confidence)


def classify_classification_confidence(score):
    """
    Convert boundary distance into a simple label.

    HIGH:
        >= 0.05 from boundary

    MEDIUM:
        0.02-0.05 from boundary

    LOW:
        < 0.02 from boundary
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
    Evaluate left/right cheek consistency separately
    from overall classification confidence.

    This prevents a large side difference from automatically
    making the overall temperature classification LOW.
    """

    side_gap = abs(
        float(left_score)
        - float(right_score)
    )

    classification_agreement = (
        left_classification
        == right_classification
    )

    # Strong disagreement:
    # different categorical temperatures OR very large score gap.
    if not classification_agreement:
        label = "LOW"

    elif side_gap >= 0.10:
        label = "LOW"

    elif side_gap >= 0.05:
        label = "MEDIUM"

    else:
        label = "HIGH"

    # Convert gap into a diagnostic consistency score.
    #
    # 0.00 gap -> 1.00 consistency
    # 0.10+ gap -> 0.00 consistency
    consistency_score = np.clip(
        1.0 - (side_gap / 0.10),
        0.0,
        1.0,
    )

    return {
        "label": label,
        "score": float(consistency_score),
        "side_gap": float(side_gap),
        "classification_agreement": bool(
            classification_agreement
        ),
    }


# ============================================================
# IMAGE ANALYSIS
# ============================================================

def analyze_image(image_path):
    """
    Run the complete diagnostic on one image.
    """

    image_rgb = load_image(
        str(image_path)
    )

    # Production face detection.
    detect_face(image_rgb)

    # Production landmark detection.
    landmarks = detect_landmarks(
        image_rgb
    )

    # Production combined cheek extraction.
    skin_rgb = extract_skin_color(
        image_rgb,
        landmarks,
    )

    skin_lab = rgb_to_lab(
        skin_rgb
    )

    # Production combined temperature score.
    combined_score = temperature_score(
        skin_lab[1],
        skin_lab[2],
    )

    combined_classification = (
        classify_temperature_stable(
            skin_lab[1],
            skin_lab[2],
        )
    )

    original_classification = classify_temperature(
        skin_lab[1],
        skin_lab[2],
    )

    # Independent left/right cheek analysis.
    left, right = extract_cheek_statistics(
        image_rgb,
        landmarks,
    )

    if left is None or right is None:
        raise ValueError(
            "Unable to extract one or both cheek regions."
        )

    # -------------------------------
    # Classification confidence
    # -------------------------------

    boundary_distance = (
        calculate_boundary_distance(
            combined_score
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

    # -------------------------------
    # Cheek consistency
    # -------------------------------

    cheek_consistency = (
        calculate_cheek_consistency(
            left_score=left["score"],
            right_score=right["score"],
            left_classification=left[
                "classification"
            ],
            right_classification=right[
                "classification"
            ],
        )
    )

    return {
        "skin_rgb": skin_rgb,
        "skin_lab": skin_lab,

        "combined_score": float(
            combined_score
        ),

        "combined_classification": (
            combined_classification
        ),

        "original_classification": (
            original_classification
        ),

        "boundary_distance": float(
            boundary_distance
        ),

        "classification_confidence": float(
            classification_confidence
        ),

        "classification_confidence_label": (
            classification_label
        ),

        "left": left,
        "right": right,

        "cheek_consistency": (
            cheek_consistency
        ),
    }


# ============================================================
# PRINTING
# ============================================================

def print_result(image_path, result):
    """
    Print one readable diagnostic result.
    """

    left = result["left"]
    right = result["right"]
    consistency = result["cheek_consistency"]

    print()
    print("=" * 70)
    print(image_path.name)
    print("=" * 70)

    print(
        f"Combined score:            "
        f"{result['combined_score']:.4f}"
    )

    print(
        f"Classification:            "
        f"{result['combined_classification']}"
    )

    print(
        f"Original classifier:       "
        f"{result['original_classification']}"
    )

    print()

    print(
        f"Left cheek score:          "
        f"{left['score']:.4f}"
    )

    print(
        f"Left classification:       "
        f"{left['classification']}"
    )

    print(
        f"Left pixels:               "
        f"{left['pixels']}"
    )

    print()

    print(
        f"Right cheek score:         "
        f"{right['score']:.4f}"
    )

    print(
        f"Right classification:      "
        f"{right['classification']}"
    )

    print(
        f"Right pixels:              "
        f"{right['pixels']}"
    )

    print()

    print(
        f"Boundary distance:         "
        f"{result['boundary_distance']:.4f}"
    )

    print(
        f"Classification confidence: "
        f"{result['classification_confidence']:.2f}"
    )

    print(
        f"Classification label:      "
        f"{result['classification_confidence_label']}"
    )

    print()

    print(
        f"Cheek score gap:           "
        f"{consistency['side_gap']:.4f}"
    )

    print(
        f"Cheek agreement:            "
        f"{'YES' if consistency['classification_agreement'] else 'NO'}"
    )

    print(
        f"Cheek consistency score:    "
        f"{consistency['score']:.2f}"
    )

    print(
        f"Cheek consistency label:    "
        f"{consistency['label']}"
    )


# ============================================================
# SUMMARY
# ============================================================

def print_summary(results):
    """
    Print aggregate results.
    """

    print()
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    total = len(results)

    classification_high = sum(
        result[
            "classification_confidence_label"
        ]
        == "HIGH"
        for result in results
    )

    classification_medium = sum(
        result[
            "classification_confidence_label"
        ]
        == "MEDIUM"
        for result in results
    )

    classification_low = sum(
        result[
            "classification_confidence_label"
        ]
        == "LOW"
        for result in results
    )

    cheek_high = sum(
        result[
            "cheek_consistency"
        ]["label"]
        == "HIGH"
        for result in results
    )

    cheek_medium = sum(
        result[
            "cheek_consistency"
        ]["label"]
        == "MEDIUM"
        for result in results
    )

    cheek_low = sum(
        result[
            "cheek_consistency"
        ]["label"]
        == "LOW"
        for result in results
    )

    agreement_count = sum(
        result[
            "cheek_consistency"
        ]["classification_agreement"]
        for result in results
    )

    average_boundary_distance = np.mean(
        [
            result[
                "boundary_distance"
            ]
            for result in results
        ]
    )

    average_classification_confidence = np.mean(
        [
            result[
                "classification_confidence"
            ]
            for result in results
        ]
    )

    average_cheek_gap = np.mean(
        [
            result[
                "cheek_consistency"
            ]["side_gap"]
            for result in results
        ]
    )

    average_cheek_consistency = np.mean(
        [
            result[
                "cheek_consistency"
            ]["score"]
            for result in results
        ]
    )

    print(
        f"Successful samples:             "
        f"{total}"
    )

    print()

    print(
        "CLASSIFICATION CONFIDENCE"
    )

    print(
        f"  HIGH:                         "
        f"{classification_high}"
    )

    print(
        f"  MEDIUM:                       "
        f"{classification_medium}"
    )

    print(
        f"  LOW:                          "
        f"{classification_low}"
    )

    print(
        f"  Average confidence:           "
        f"{average_classification_confidence:.2f}"
    )

    print(
        f"  Average boundary distance:    "
        f"{average_boundary_distance:.4f}"
    )

    print()

    print(
        "CHEEK CONSISTENCY"
    )

    print(
        f"  HIGH:                         "
        f"{cheek_high}"
    )

    print(
        f"  MEDIUM:                       "
        f"{cheek_medium}"
    )

    print(
        f"  LOW:                          "
        f"{cheek_low}"
    )

    print(
        f"  Side agreement:              "
        f"{agreement_count}/{total}"
    )

    print(
        f"  Average cheek consistency:    "
        f"{average_cheek_consistency:.2f}"
    )

    print(
        f"  Average cheek score gap:      "
        f"{average_cheek_gap:.4f}"
    )

    print()

    print(
        "PER-IMAGE SUMMARY"
    )

    print("-" * 70)

    for result in results:
        consistency = result[
            "cheek_consistency"
        ]

        print(
            f"{result['name']:<30} "
            f"score="
            f"{result['combined_score']:.4f} "
            f"class="
            f"{result['combined_classification']:<7} "
            f"class_conf="
            f"{result['classification_confidence_label']:<6} "
            f"cheek="
            f"{consistency['label']}"
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
            "evaluate_temperature_confidence.py "
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
