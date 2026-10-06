import os
import sys
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tinayu_engine import (
    load_image,
    detect_face,
    detect_landmarks,
    landmarks_to_pixels,
    polygon_mask,
    rgb_to_lab,
    temperature_score,
    classify_temperature_stable,
)


# ============================================================
# CHEEK PIXEL EXTRACTION
# ============================================================

def extract_cheek_side_pixels(image_rgb, landmarks):
    """
    Reproduce the exact production cheek geometry,
    but keep left and right cheeks separate.
    """

    points = landmarks_to_pixels(
        landmarks,
        image_rgb.shape
    )

    left_indices = [
        50,
        101,
        118,
        119,
        100,
        47
    ]

    right_indices = [
        280,
        330,
        347,
        348,
        329,
        277
    ]

    left_points = points[left_indices]
    right_points = points[right_indices]

    left_mask = polygon_mask(
        image_rgb.shape,
        left_points
    )

    right_mask = polygon_mask(
        image_rgb.shape,
        right_points
    )

    left_pixels = image_rgb[
        left_mask > 0
    ]

    right_pixels = image_rgb[
        right_mask > 0
    ]

    if len(left_pixels) == 0:
        raise ValueError(
            "Unable to extract left cheek pixels."
        )

    if len(right_pixels) == 0:
        raise ValueError(
            "Unable to extract right cheek pixels."
        )

    return left_pixels, right_pixels


# ============================================================
# PIXEL SUMMARY
# ============================================================

def summarize_pixels(pixels):

    mean_rgb = np.mean(
        pixels,
        axis=0
    )

    mean_rgb = (
        np.round(mean_rgb)
        .astype(int)
        .tolist()
    )

    lab = rgb_to_lab(
        mean_rgb
    )

    L = float(lab[0])
    a = float(lab[1])
    b = float(lab[2])

    score = temperature_score(
        a,
        b
    )

    temperature = classify_temperature_stable(
        a,
        b
    )

    return {
        "pixels": int(len(pixels)),

        "rgb": mean_rgb,

        "lab": {
            "L": round(L, 2),
            "a": round(a, 2),
            "b": round(b, 2),
        },

        "temperature_score": float(
            score
        ),

        "temperature": temperature,
    }


# ============================================================
# ANALYZE IMAGE
# ============================================================

def analyze_sample(image_path):

    image_rgb = load_image(
        image_path
    )

    # --------------------------------------------------------
    # Use the exact production detection pipeline.
    # --------------------------------------------------------

    detect_face(
        image_rgb
    )

    landmarks = detect_landmarks(
        image_rgb
    )

    # --------------------------------------------------------
    # Extract cheeks.
    # --------------------------------------------------------

    left_pixels, right_pixels = (
        extract_cheek_side_pixels(
            image_rgb,
            landmarks
        )
    )

    # --------------------------------------------------------
    # Analyze each side.
    # --------------------------------------------------------

    left = summarize_pixels(
        left_pixels
    )

    right = summarize_pixels(
        right_pixels
    )

    # --------------------------------------------------------
    # Combined production-equivalent cheek sample.
    # --------------------------------------------------------

    combined_pixels = np.concatenate(
        [
            left_pixels,
            right_pixels
        ],
        axis=0
    )

    combined = summarize_pixels(
        combined_pixels
    )

    # --------------------------------------------------------
    # Left/right temperature disagreement.
    # --------------------------------------------------------

    score_delta = abs(
        left["temperature_score"]
        -
        right["temperature_score"]
    )

    side_agreement = (
        left["temperature"]
        ==
        right["temperature"]
    )

    return {
        "left": left,
        "right": right,
        "combined": combined,
        "score_delta": float(
            score_delta
        ),
        "side_agreement": side_agreement,
    }


# ============================================================
# SAMPLE DISCOVERY
# ============================================================

def discover_samples(sample_dir):

    extensions = (
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
    )

    files = []

    for name in sorted(
        os.listdir(sample_dir)
    ):

        path = os.path.join(
            sample_dir,
            name
        )

        if (
            os.path.isfile(path)
            and name.lower().endswith(
                extensions
            )
        ):
            files.append(path)

    return files


# ============================================================
# MAIN
# ============================================================

def main():

    if len(sys.argv) < 2:

        print(
            "Usage:\n"
            "python evaluate_cheek_side_temperature.py "
            "<sample_directory>"
        )

        return

    sample_dir = sys.argv[1]

    if not os.path.isdir(
        sample_dir
    ):

        raise FileNotFoundError(
            f"Sample directory not found: "
            f"{sample_dir}"
        )

    samples = discover_samples(
        sample_dir
    )

    if not samples:

        print(
            "No image samples found."
        )

        return

    results = []

    # ========================================================
    # PER-SAMPLE RESULTS
    # ========================================================

    for image_path in samples:

        name = os.path.basename(
            image_path
        )

        print()
        print("=" * 72)
        print(name)
        print("=" * 72)

        try:

            result = analyze_sample(
                image_path
            )

            results.append(
                (
                    name,
                    result
                )
            )

            left = result["left"]
            right = result["right"]
            combined = result["combined"]

            print(
                f"Left cheek:     "
                f"{left['temperature_score']:.4f} "
                f"{left['temperature']}"
            )

            print(
                f"Right cheek:    "
                f"{right['temperature_score']:.4f} "
                f"{right['temperature']}"
            )

            print(
                f"Combined:       "
                f"{combined['temperature_score']:.4f} "
                f"{combined['temperature']}"
            )

            print(
                f"Side delta:     "
                f"{result['score_delta']:.4f}"
            )

            print(
                f"Side agreement: "
                f"{'YES' if result['side_agreement'] else 'NO'}"
            )

            print(
                f"Pixels:         "
                f"L={left['pixels']} "
                f"R={right['pixels']}"
            )

            print(
                f"Left LAB:       "
                f"L={left['lab']['L']:.2f} "
                f"a={left['lab']['a']:.2f} "
                f"b={left['lab']['b']:.2f}"
            )

            print(
                f"Right LAB:      "
                f"L={right['lab']['L']:.2f} "
                f"a={right['lab']['a']:.2f} "
                f"b={right['lab']['b']:.2f}"
            )

        except Exception as exc:

            print(
                f"ERROR: {exc}"
            )


    # ========================================================
    # SUMMARY
    # ========================================================

    if not results:

        print()
        print(
            "No successful samples."
        )

        return

    deltas = [
        result["score_delta"]
        for _, result in results
    ]

    agreements = [
        result["side_agreement"]
        for _, result in results
    ]

    left_scores = [
        result["left"]["temperature_score"]
        for _, result in results
    ]

    right_scores = [
        result["right"]["temperature_score"]
        for _, result in results
    ]

    combined_scores = [
        result["combined"]["temperature_score"]
        for _, result in results
    ]

    print()
    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)

    print(
        f"Successful samples: "
        f"{len(results)}/{len(samples)}"
    )

    print(
        f"Side agreement:     "
        f"{sum(agreements)}/"
        f"{len(agreements)} "
        f"("
        f"{100 * np.mean(agreements):.1f}"
        f"%)"
    )

    print(
        f"Mean side delta:     "
        f"{np.mean(deltas):.4f}"
    )

    print(
        f"Median side delta:   "
        f"{np.median(deltas):.4f}"
    )

    print(
        f"Max side delta:      "
        f"{np.max(deltas):.4f}"
    )

    print(
        f"Mean left score:     "
        f"{np.mean(left_scores):.4f}"
    )

    print(
        f"Mean right score:    "
        f"{np.mean(right_scores):.4f}"
    )

    print(
        f"Mean combined score: "
        f"{np.mean(combined_scores):.4f}"
    )

    # ========================================================
    # LARGE SIDE DIFFERENCES
    # ========================================================

    print()
    print(
        "Samples with side delta >= 0.03 "
        "(diagnostic only):"
    )

    large_deltas = [
        (
            name,
            result["score_delta"]
        )
        for name, result in results
        if result["score_delta"] >= 0.03
    ]

    if large_deltas:

        for name, delta in large_deltas:

            print(
                f"  - {name}: "
                f"{delta:.4f}"
            )

    else:

        print(
            "  None"
        )


if __name__ == "__main__":
    main()