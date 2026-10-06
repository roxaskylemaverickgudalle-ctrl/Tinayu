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
# CHEEK EXTRACTION
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
# RGB → LAB FOR INDIVIDUAL PIXELS
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
# CURRENT PRODUCTION METHOD
# ============================================================

def raw_temperature(pixels):

    mean_rgb = (
        np.mean(pixels, axis=0)
        .round()
        .astype(int)
        .tolist()
    )

    lab = rgb_to_lab(
        mean_rgb
    )

    score = temperature_score(
        float(lab[1]),
        float(lab[2])
    )

    temperature = classify_temperature_stable(
        float(lab[1]),
        float(lab[2])
    )

    return {
        "rgb": mean_rgb,
        "lab": lab,
        "score": float(score),
        "temperature": temperature,
    }


# ============================================================
# METHOD 1
# PER-CHEEK LIGHTNESS NORMALIZATION
#
# Preserve a/b and normalize only L.
# This asks:
#
# "If brightness is the main problem, does removing
# brightness differences make the cheeks agree?"
# ============================================================

def lightness_normalized_temperature(pixels):

    labs = pixels_to_lab(pixels)

    mean_lab = np.mean(
        labs,
        axis=0
    )

    # Target L is deliberately fixed at 60,
    # matching the production normalization concept.
    normalized_lab = mean_lab.copy()

    normalized_lab[0] = 60.0

    # Recreate a/b directly.
    score = temperature_score(
        normalized_lab[1],
        normalized_lab[2]
    )

    temperature = classify_temperature_stable(
        normalized_lab[1],
        normalized_lab[2]
    )

    return {
        "lab": normalized_lab,
        "score": float(score),
        "temperature": temperature,
    }


# ============================================================
# METHOD 2
# CHROMATICITY NORMALIZATION
#
# Remove overall RGB brightness while preserving
# relative channel relationships.
#
# This is diagnostic only.
# ============================================================

def chromaticity_normalized_temperature(pixels):

    pixels_float = pixels.astype(
        np.float64
    )

    channel_sum = np.sum(
        pixels_float,
        axis=1,
        keepdims=True
    )

    valid = channel_sum[:, 0] > 0

    normalized = (
        pixels_float[valid]
        /
        channel_sum[valid]
    )

    # Convert normalized RGB back into an arbitrary
    # 0-255 representation while preserving ratios.
    normalized_rgb = (
        normalized * 255.0
    )

    mean_rgb = (
        np.mean(
            normalized_rgb,
            axis=0
        )
        .round()
        .astype(int)
        .tolist()
    )

    lab = rgb_to_lab(
        mean_rgb
    )

    score = temperature_score(
        float(lab[1]),
        float(lab[2])
    )

    temperature = classify_temperature_stable(
        float(lab[1]),
        float(lab[2])
    )

    return {
        "rgb": mean_rgb,
        "lab": lab,
        "score": float(score),
        "temperature": temperature,
    }


# ============================================================
# METHOD 3
# CHANNEL-RATIO TEMPERATURE SIGNAL
#
# Instead of relying on absolute brightness,
# compare red/green/blue ratios.
#
# This is NOT proposed production logic.
# It is only a diagnostic signal.
# ============================================================

def channel_ratio_summary(pixels):

    pixels_float = pixels.astype(
        np.float64
    )

    r = pixels_float[:, 0]
    g = pixels_float[:, 1]
    b = pixels_float[:, 2]

    rg = r / np.maximum(
        g,
        1.0
    )

    rb = r / np.maximum(
        b,
        1.0
    )

    gb = g / np.maximum(
        b,
        1.0
    )

    return {
        "R/G": float(np.mean(rg)),
        "R/B": float(np.mean(rb)),
        "G/B": float(np.mean(gb)),
    }


# ============================================================
# ANALYZE ONE SAMPLE
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

    left_raw = raw_temperature(
        left_pixels
    )

    right_raw = raw_temperature(
        right_pixels
    )

    left_L = lightness_normalized_temperature(
        left_pixels
    )

    right_L = lightness_normalized_temperature(
        right_pixels
    )

    left_chroma = chromaticity_normalized_temperature(
        left_pixels
    )

    right_chroma = chromaticity_normalized_temperature(
        right_pixels
    )

    left_ratios = channel_ratio_summary(
        left_pixels
    )

    right_ratios = channel_ratio_summary(
        right_pixels
    )

    return {
        "left_raw": left_raw,
        "right_raw": right_raw,

        "left_L": left_L,
        "right_L": right_L,

        "left_chroma": left_chroma,
        "right_chroma": right_chroma,

        "left_ratios": left_ratios,
        "right_ratios": right_ratios,
    }


# ============================================================
# PRINT
# ============================================================

def print_method(
    label,
    left,
    right
):

    delta = abs(
        left["score"]
        -
        right["score"]
    )

    print()
    print(
        f"  {label}"
    )

    print(
        f"    Left:   "
        f"{left['score']:.4f} "
        f"{left['temperature']}"
    )

    print(
        f"    Right:  "
        f"{right['score']:.4f} "
        f"{right['temperature']}"
    )

    print(
        f"    Delta:  "
        f"{delta:.4f}"
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
            "python evaluate_cheek_normalization.py "
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
        print("=" * 80)
        print(name)
        print("=" * 80)

        try:

            result = analyze_sample(
                image_path
            )

            print_method(
                "CURRENT RAW",
                result["left_raw"],
                result["right_raw"]
            )

            print_method(
                "LIGHTNESS NORMALIZED",
                result["left_L"],
                result["right_L"]
            )

            print_method(
                "CHROMATICITY NORMALIZED",
                result["left_chroma"],
                result["right_chroma"]
            )

            # ----------------------------------------------------
            # Channel ratios
            # ----------------------------------------------------

            left = result["left_ratios"]
            right = result["right_ratios"]

            print()
            print(
                "  CHANNEL RATIOS"
            )

            print(
                f"    R/G: "
                f"{left['R/G']:.4f} "
                f"vs "
                f"{right['R/G']:.4f} "
                f"(delta "
                f"{abs(left['R/G'] - right['R/G']):.4f})"
            )

            print(
                f"    R/B: "
                f"{left['R/B']:.4f} "
                f"vs "
                f"{right['R/B']:.4f} "
                f"(delta "
                f"{abs(left['R/B'] - right['R/B']):.4f})"
            )

            print(
                f"    G/B: "
                f"{left['G/B']:.4f} "
                f"vs "
                f"{right['G/B']:.4f} "
                f"(delta "
                f"{abs(left['G/B'] - right['G/B']):.4f})"
            )

        except Exception as exc:

            print(
                f"ERROR: {exc}"
            )


if __name__ == "__main__":
    main()