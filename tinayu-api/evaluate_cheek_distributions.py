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

    points = landmarks_to_pixels(
        landmarks,
        image_rgb.shape
    )

    left_indices = [
        50, 101, 118, 119, 100, 47
    ]

    right_indices = [
        280, 330, 347, 348, 329, 277
    ]

    left_mask = polygon_mask(
        image_rgb.shape,
        points[left_indices]
    )

    right_mask = polygon_mask(
        image_rgb.shape,
        points[right_indices]
    )

    left_pixels = image_rgb[left_mask > 0]
    right_pixels = image_rgb[right_mask > 0]

    return left_pixels, right_pixels


# ============================================================
# RGB → LAB FOR EVERY PIXEL
# ============================================================

def pixels_to_lab(pixels):

    labs = []

    for pixel in pixels:

        rgb = (
            np.round(pixel)
            .astype(int)
            .tolist()
        )

        lab = rgb_to_lab(rgb)

        labs.append([
            float(lab[0]),
            float(lab[1]),
            float(lab[2]),
        ])

    return np.asarray(labs)


# ============================================================
# PERCENTILES
# ============================================================

def percentile_summary(values):

    percentiles = np.percentile(
        values,
        [10, 25, 50, 75, 90]
    )

    return {
        "p10": round(float(percentiles[0]), 2),
        "p25": round(float(percentiles[1]), 2),
        "p50": round(float(percentiles[2]), 2),
        "p75": round(float(percentiles[3]), 2),
        "p90": round(float(percentiles[4]), 2),
        "min": round(float(np.min(values)), 2),
        "max": round(float(np.max(values)), 2),
        "mean": round(float(np.mean(values)), 2),
        "std": round(float(np.std(values)), 2),
    }


# ============================================================
# SIDE ANALYSIS
# ============================================================

def analyze_side(pixels):

    labs = pixels_to_lab(pixels)

    L = labs[:, 0]
    a = labs[:, 1]
    b = labs[:, 2]

    # Mean RGB — matches production extraction.
    mean_rgb = (
        np.mean(pixels, axis=0)
        .round()
        .astype(int)
        .tolist()
    )

    mean_lab = rgb_to_lab(
        mean_rgb
    )

    score = temperature_score(
        float(mean_lab[1]),
        float(mean_lab[2])
    )

    temperature = classify_temperature_stable(
        float(mean_lab[1]),
        float(mean_lab[2])
    )

    return {
        "pixels": len(pixels),

        "rgb_mean": mean_rgb,

        "lab_mean": {
            "L": round(float(mean_lab[0]), 2),
            "a": round(float(mean_lab[1]), 2),
            "b": round(float(mean_lab[2]), 2),
        },

        "temperature_score": round(
            float(score),
            4
        ),

        "temperature": temperature,

        "L_distribution": percentile_summary(L),
        "a_distribution": percentile_summary(a),
        "b_distribution": percentile_summary(b),
    }


# ============================================================
# ANALYZE SAMPLE
# ============================================================

def analyze_sample(image_path):

    image_rgb = load_image(
        image_path
    )

    detect_face(
        image_rgb
    )

    landmarks = detect_landmarks(
        image_rgb
    )

    left_pixels, right_pixels = (
        extract_cheek_side_pixels(
            image_rgb,
            landmarks
        )
    )

    left = analyze_side(
        left_pixels
    )

    right = analyze_side(
        right_pixels
    )

    return left, right


# ============================================================
# PRINT DISTRIBUTION
# ============================================================

def print_distribution(
    label,
    distribution
):

    print(
        f"    {label:<3} "
        f"min={distribution['min']:>6.2f} "
        f"p10={distribution['p10']:>6.2f} "
        f"p25={distribution['p25']:>6.2f} "
        f"median={distribution['p50']:>6.2f} "
        f"p75={distribution['p75']:>6.2f} "
        f"p90={distribution['p90']:>6.2f} "
        f"max={distribution['max']:>6.2f} "
        f"mean={distribution['mean']:>6.2f} "
        f"std={distribution['std']:>6.2f}"
    )


def print_side(label, side):

    print()
    print(
        f"  {label} CHEEK"
    )

    print(
        f"    Pixels: "
        f"{side['pixels']}"
    )

    print(
        f"    RGB mean: "
        f"{side['rgb_mean']}"
    )

    print(
        f"    LAB mean: "
        f"L={side['lab_mean']['L']:.2f} "
        f"a={side['lab_mean']['a']:.2f} "
        f"b={side['lab_mean']['b']:.2f}"
    )

    print(
        f"    Temperature: "
        f"{side['temperature_score']:.4f} "
        f"{side['temperature']}"
    )

    print(
        "    L distribution:"
    )

    print_distribution(
        "L",
        side["L_distribution"]
    )

    print(
        "    a distribution:"
    )

    print_distribution(
        "a",
        side["a_distribution"]
    )

    print(
        "    b distribution:"
    )

    print_distribution(
        "b",
        side["b_distribution"]
    )


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
            "python evaluate_cheek_distributions.py "
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

    for image_path in samples:

        name = os.path.basename(
            image_path
        )

        print()
        print("=" * 100)
        print(name)
        print("=" * 100)

        try:

            left, right = analyze_sample(
                image_path
            )

            print_side(
                "LEFT",
                left
            )

            print_side(
                "RIGHT",
                right
            )

            # ----------------------------------------------------
            # Direct side comparison
            # ----------------------------------------------------

            score_delta = abs(
                left["temperature_score"]
                -
                right["temperature_score"]
            )

            print()
            print(
                "  SIDE COMPARISON"
            )

            print(
                f"    Temperature delta: "
                f"{score_delta:.4f}"
            )

            print(
                f"    L mean delta:      "
                f"{abs(left['lab_mean']['L'] - right['lab_mean']['L']):.2f}"
            )

            print(
                f"    a mean delta:      "
                f"{abs(left['lab_mean']['a'] - right['lab_mean']['a']):.2f}"
            )

            print(
                f"    b mean delta:      "
                f"{abs(left['lab_mean']['b'] - right['lab_mean']['b']):.2f}"
            )

            print(
                f"    L std delta:       "
                f"{abs(left['L_distribution']['std'] - right['L_distribution']['std']):.2f}"
            )

            print(
                f"    a std delta:       "
                f"{abs(left['a_distribution']['std'] - right['a_distribution']['std']):.2f}"
            )

            print(
                f"    b std delta:       "
                f"{abs(left['b_distribution']['std'] - right['b_distribution']['std']):.2f}"
            )

        except Exception as exc:

            print(
                f"ERROR: {exc}"
            )


if __name__ == "__main__":
    main()