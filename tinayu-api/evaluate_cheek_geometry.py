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
# CHEEK GEOMETRY
# ============================================================

LEFT_INDICES = [50, 101, 118, 119, 100, 47]
RIGHT_INDICES = [280, 330, 347, 348, 329, 277]


def extract_geometry_regions(image_rgb, landmarks):
    """
    Build several versions of each production cheek polygon.

    CURRENT:
        Exact production polygon.

    INNER_80:
        Shrink polygon toward its centroid by 20%.

    INNER_60:
        Shrink polygon toward its centroid by 40%.

    The smaller regions are diagnostic only.
    """

    points = landmarks_to_pixels(
        landmarks,
        image_rgb.shape
    )

    height, width = image_rgb.shape[:2]

    def make_regions(indices):

        polygon = points[indices].astype(
            np.float32
        )

        centroid = np.mean(
            polygon,
            axis=0
        )

        regions = {}

        # ----------------------------------------------------
        # Current production region
        # ----------------------------------------------------

        current_mask = polygon_mask(
            image_rgb.shape,
            polygon.astype(int)
        )

        regions["CURRENT"] = current_mask

        # ----------------------------------------------------
        # Inner regions
        # ----------------------------------------------------

        for name, scale in [
            ("INNER_80", 0.80),
            ("INNER_60", 0.60),
        ]:

            inner_polygon = (
                centroid
                +
                (polygon - centroid) * scale
            )

            inner_polygon[:, 0] = np.clip(
                inner_polygon[:, 0],
                0,
                width - 1
            )

            inner_polygon[:, 1] = np.clip(
                inner_polygon[:, 1],
                0,
                height - 1
            )

            mask = polygon_mask(
                image_rgb.shape,
                inner_polygon.astype(int)
            )

            regions[name] = mask

        return regions

    return (
        make_regions(LEFT_INDICES),
        make_regions(RIGHT_INDICES)
    )


# ============================================================
# TEMPERATURE
# ============================================================

def analyze_pixels(pixels):

    if len(pixels) == 0:
        raise ValueError(
            "Region contains no pixels."
        )

    mean_rgb = (
        np.mean(
            pixels,
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
        "pixels": int(len(pixels)),
        "rgb": mean_rgb,
        "L": float(lab[0]),
        "a": float(lab[1]),
        "b": float(lab[2]),
        "score": float(score),
        "temperature": temperature,
    }


# ============================================================
# SAMPLE
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

    left_regions, right_regions = (
        extract_geometry_regions(
            image_rgb,
            landmarks
        )
    )

    result = {}

    for region_name in [
        "CURRENT",
        "INNER_80",
        "INNER_60",
    ]:

        left_pixels = image_rgb[
            left_regions[region_name] > 0
        ]

        right_pixels = image_rgb[
            right_regions[region_name] > 0
        ]

        left = analyze_pixels(
            left_pixels
        )

        right = analyze_pixels(
            right_pixels
        )

        result[region_name] = {
            "left": left,
            "right": right,
            "delta": abs(
                left["score"]
                -
                right["score"]
            ),
            "agreement": (
                left["temperature"]
                ==
                right["temperature"]
            ),
        }

    return result


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
# PRINT
# ============================================================

def print_result(
    region_name,
    result
):

    left = result["left"]
    right = result["right"]

    print()
    print(
        f"  {region_name}"
    )

    print(
        f"    Left:     "
        f"{left['score']:.4f} "
        f"{left['temperature']} "
        f"(n={left['pixels']})"
    )

    print(
        f"    Right:    "
        f"{right['score']:.4f} "
        f"{right['temperature']} "
        f"(n={right['pixels']})"
    )

    print(
        f"    Delta:    "
        f"{result['delta']:.4f}"
    )

    print(
        f"    Agreement: "
        f"{'YES' if result['agreement'] else 'NO'}"
    )

    print(
        f"    Left LAB: "
        f"L={left['L']:.2f} "
        f"a={left['a']:.2f} "
        f"b={left['b']:.2f}"
    )

    print(
        f"    Right LAB:"
        f" L={right['L']:.2f} "
        f"a={right['a']:.2f} "
        f"b={right['b']:.2f}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    if len(sys.argv) < 2:

        print(
            "Usage:\n"
            "python evaluate_cheek_geometry.py "
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

    # Used for aggregate comparison.
    aggregate = {
        "CURRENT": [],
        "INNER_80": [],
        "INNER_60": [],
    }

    agreement = {
        "CURRENT": [],
        "INNER_80": [],
        "INNER_60": [],
    }

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

            for region_name in [
                "CURRENT",
                "INNER_80",
                "INNER_60",
            ]:

                print_result(
                    region_name,
                    result[region_name]
                )

                aggregate[
                    region_name
                ].append(
                    result[region_name][
                        "delta"
                    ]
                )

                agreement[
                    region_name
                ].append(
                    result[region_name][
                        "agreement"
                    ]
                )

        except Exception as exc:

            print(
                f"ERROR: {exc}"
            )

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print("=" * 80)
    print("GEOMETRY SUMMARY")
    print("=" * 80)

    for region_name in [
        "CURRENT",
        "INNER_80",
        "INNER_60",
    ]:

        deltas = aggregate[
            region_name
        ]

        agreements = agreement[
            region_name
        ]

        if not deltas:
            continue

        print()
        print(
            region_name
        )

        print(
            f"  Mean side delta:   "
            f"{np.mean(deltas):.4f}"
        )

        print(
            f"  Median side delta: "
            f"{np.median(deltas):.4f}"
        )

        print(
            f"  Max side delta:    "
            f"{np.max(deltas):.4f}"
        )

        print(
            f"  Side agreement:    "
            f"{sum(agreements)}/"
            f"{len(agreements)} "
            f"("
            f"{100 * np.mean(agreements):.1f}"
            f"%)"
        )


if __name__ == "__main__":
    main()