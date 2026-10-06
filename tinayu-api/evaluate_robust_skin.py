
import json
import statistics
import sys
from pathlib import Path

import cv2
import numpy as np

from tinayu_engine import (
    analyze_image,
    decode_image,
    detect_face,
    detect_landmarks,
    landmarks_to_pixels,
    polygon_mask,
    normalize_skin_profile,
    rgb_to_lab,
    rgb_to_hsv_features,
    build_automatic_profile,
)


SUPPORTED = {".jpg", ".jpeg", ".png", ".webp"}


# ============================================================
# CURRENT TINAYU SKIN EXTRACTION
# ============================================================

CHEEK_LEFT = [
    50,
    101,
    118,
    119,
    100,
    47,
]

CHEEK_RIGHT = [
    280,
    330,
    347,
    348,
    329,
    277,
]


def build_cheek_mask(image_rgb, landmarks):
    points = landmarks_to_pixels(
        landmarks,
        image_rgb.shape
    )

    left_mask = polygon_mask(
        image_rgb.shape,
        points[CHEEK_LEFT]
    )

    right_mask = polygon_mask(
        image_rgb.shape,
        points[CHEEK_RIGHT]
    )

    return cv2.bitwise_or(
        left_mask,
        right_mask
    )


def current_mean_rgb(image_rgb, mask):
    pixels = image_rgb[
        mask > 0
    ]

    if len(pixels) == 0:
        raise ValueError(
            "No cheek pixels found."
        )

    return (
        np.mean(
            pixels,
            axis=0
        )
        .astype(int)
        .tolist()
    )


# ============================================================
# ROBUST LAB-MEDIAN EXTRACTION
# ============================================================

def robust_median_rgb(
    image_rgb,
    mask,
    trim_percentile=10
):
    """
    Experimental robust skin estimator.

    Steps:
        1. Collect cheek pixels.
        2. Convert RGB pixels to LAB.
        3. Remove extreme LAB pixels using percentile trimming.
        4. Take the median LAB value.
        5. Convert the representative LAB color back to RGB.

    This function does NOT modify Tinayu's engine.
    """

    pixels = image_rgb[
        mask > 0
    ].astype(
        np.uint8
    )

    if len(pixels) == 0:
        raise ValueError(
            "No cheek pixels found."
        )

    # Convert every pixel to LAB.
    lab_pixels = cv2.cvtColor(
        pixels.reshape(
            -1,
            1,
            3
        ),
        cv2.COLOR_RGB2LAB
    ).reshape(
        -1,
        3
    ).astype(
        np.float32
    )

    # OpenCV LAB uses:
    # L: 0-255
    # a: 0-255
    # b: 0-255
    #
    # Convert to approximately the same LAB scale
    # used by Tinayu's rgb_to_lab():
    #
    # L: 0-100
    # a: centered around 0
    # b: centered around 0

    lab_scaled = np.empty_like(
        lab_pixels
    )

    lab_scaled[:, 0] = (
        lab_pixels[:, 0]
        * 100.0
        / 255.0
    )

    lab_scaled[:, 1] = (
        lab_pixels[:, 1]
        - 128.0
    )

    lab_scaled[:, 2] = (
        lab_pixels[:, 2]
        - 128.0
    )

    # --------------------------------------------------------
    # ROBUST OUTLIER FILTER
    # --------------------------------------------------------

    lower = np.percentile(
        lab_scaled,
        trim_percentile,
        axis=0
    )

    upper = np.percentile(
        lab_scaled,
        100.0 - trim_percentile,
        axis=0
    )

    keep = np.all(
        (
            lab_scaled >= lower
        )
        &
        (
            lab_scaled <= upper
        ),
        axis=1
    )

    filtered_lab = lab_scaled[
        keep
    ]

    if len(filtered_lab) < 10:
        filtered_lab = lab_scaled

    # --------------------------------------------------------
    # MEDIAN REPRESENTATIVE COLOR
    # --------------------------------------------------------

    representative_lab = np.median(
        filtered_lab,
        axis=0
    )

    # Convert back into OpenCV LAB representation.
    cv_lab = np.array(
        [[
            [
                representative_lab[0]
                * 255.0
                / 100.0,

                representative_lab[1]
                + 128.0,

                representative_lab[2]
                + 128.0,
            ]
        ]],
        dtype=np.uint8
    )

    rgb = cv2.cvtColor(
        cv_lab,
        cv2.COLOR_LAB2RGB
    )[0, 0]

    return (
        rgb.astype(int).tolist(),
        representative_lab.astype(float).tolist(),
        len(pixels),
        len(filtered_lab),
    )


# ============================================================
# PROFILE HELPERS
# ============================================================

def profile_from_rgb(
    rgb,
    hair_rgb,
    eye_rgb
):
    normalized = normalize_skin_profile(
        rgb
    )

    normalized_rgb = (
        normalized["rgb"]
        .astype(int)
        .tolist()
    )

    profile = build_automatic_profile(
        normalized_rgb,
        hair_rgb,
        eye_rgb
    )

    skin = profile["skin"]
    heuristics = profile["heuristics"]

    return {
        "normalized_rgb": normalized_rgb,
        "lab": skin["lab"],
        "hue": skin["hue"],
        "chroma": skin["chroma"],
        "temperature_strength": heuristics[
            "temperature_strength"
        ],
        "temperature": heuristics[
            "temperature"
        ],
        "season": heuristics[
            "suggested_season"
        ],
        "saturation": heuristics[
            "skin_saturation"
        ],
    }


# ============================================================
# IMAGE ANALYSIS
# ============================================================

def analyze_ab(path):
    with path.open("rb") as file:
        image_bytes = file.read()

    # Existing production pipeline.
    current_result = analyze_image(
        image_bytes
    )

    # Decode and detect landmarks again for the
    # experimental extraction.
    image_rgb = decode_image(
        image_bytes
    )

    face_result = detect_face(
        image_rgb
    )

    landmarks = detect_landmarks(
        image_rgb
    )

    face_confidence = float(
        face_result
        .detections[0]
        .categories[0]
        .score
    )

    mask = build_cheek_mask(
        image_rgb,
        landmarks
    )

    # Existing mean estimator.
    current_rgb = current_mean_rgb(
        image_rgb,
        mask
    )

    # Experimental robust estimator.
    robust_rgb, robust_lab, total_pixels, filtered_pixels = (
        robust_median_rgb(
            image_rgb,
            mask
        )
    )

    # Pull the same hair/eye information used by
    # the production pipeline.
    #
    # We retrieve these from the existing result where possible.
    colors = current_result.get(
        "colors",
        {}
    )

    hair_rgb = colors.get(
        "hair_rgb"
    )

    eye_rgb = colors.get(
        "eye_rgb"
    )

    current_profile = current_result.get(
        "profile",
        {}
    )

    current_heuristics = current_profile.get(
        "heuristics",
        {}
    )

    robust_profile = profile_from_rgb(
        robust_rgb,
        hair_rgb,
        eye_rgb
    )

    current_skin = current_profile.get(
        "skin",
        {}
    )

    current_lab = current_skin.get(
        "lab",
        {}
    )

    current_strength = current_heuristics.get(
        "temperature_strength"
    )

    return {
        "file": path.name,

        "current": {
            "raw_rgb": current_rgb,
            "normalized_rgb": colors.get(
                "skin_normalized_rgb"
            ),
            "lab": current_lab,
            "temperature_strength": current_strength,
            "temperature": current_heuristics.get(
                "temperature"
            ),
            "season": current_heuristics.get(
                "suggested_season"
            ),
            "saturation": current_heuristics.get(
                "skin_saturation"
            ),
        },

        "robust": robust_profile,

        "experiment": {
            "total_cheek_pixels": total_pixels,
            "pixels_after_trim": filtered_pixels,
            "pixels_removed": (
                total_pixels
                - filtered_pixels
            ),
            "trim_percent": 10,
            "robust_raw_rgb": robust_rgb,
            "robust_raw_lab": {
                "L": round(
                    robust_lab[0],
                    2
                ),
                "a": round(
                    robust_lab[1],
                    2
                ),
                "b": round(
                    robust_lab[2],
                    2
                ),
            },
            "temperature_strength_delta": round(
                float(
                    robust_profile[
                        "temperature_strength"
                    ]
                )
                - float(
                    current_strength
                ),
                3
            ),
        },

        "quality": current_result.get(
            "quality",
            {}
        ),

        "face_confidence": round(
            face_confidence,
            4
        ),
    }


# ============================================================
# SUMMARY
# ============================================================

def safe_mean(values):
    values = [
        float(value)
        for value in values
        if value is not None
    ]

    return round(
        statistics.mean(values),
        3
    ) if values else None


def safe_median(values):
    values = [
        float(value)
        for value in values
        if value is not None
    ]

    return round(
        statistics.median(values),
        3
    ) if values else None


def main():
    directory = Path(
        sys.argv[1]
        if len(sys.argv) > 1
        else "validation_samples"
    )

    if not directory.exists():
        print(
            f"ERROR: Directory not found: {directory}"
        )
        return 1

    files = sorted(
        path
        for path in directory.iterdir()
        if (
            path.is_file()
            and path.suffix.lower()
            in SUPPORTED
        )
    )

    if len(files) < 2:
        print(
            "ERROR: At least 2 validation images are required."
        )
        return 1

    print()
    print(
        "=" * 64
    )
    print(
        "TINAYU ROBUST SKIN EXTRACTION A/B EXPERIMENT"
    )
    print(
        "=" * 64
    )
    print()
    print(
        "CURRENT:"
    )
    print(
        "  Mean RGB across all cheek pixels."
    )
    print()
    print(
        "ROBUST:"
    )
    print(
        "  LAB percentile trimming + median LAB."
    )
    print()
    print(
        "No Tinayu engine values will be modified."
    )
    print()

    results = []
    failures = []

    for index, path in enumerate(
        files,
        start=1
    ):
        print(
            f"[{index}/{len(files)}] {path.name}"
        )

        try:
            result = analyze_ab(
                path
            )

            results.append(
                result
            )

            current = result[
                "current"
            ]

            robust = result[
                "robust"
            ]

            experiment = result[
                "experiment"
            ]

            print()
            print(
                "  CURRENT"
            )
            print(
                f"    RGB: {current['raw_rgb']}"
            )
            print(
                f"    LAB: {current['lab']}"
            )
            print(
                f"    temp strength: "
                f"{current['temperature_strength']}"
            )
            print(
                f"    temp: {current['temperature']}"
            )
            print(
                f"    season: {current['season']}"
            )

            print()
            print(
                "  ROBUST"
            )
            print(
                f"    RGB: {robust['normalized_rgb']}"
            )
            print(
                f"    LAB: {robust['lab']}"
            )
            print(
                f"    temp strength: "
                f"{robust['temperature_strength']}"
            )
            print(
                f"    temp: {robust['temperature']}"
            )
            print(
                f"    season: {robust['season']}"
            )

            print()
            print(
                "  EXPERIMENT"
            )
            print(
                f"    cheek pixels: "
                f"{experiment['total_cheek_pixels']}"
            )
            print(
                f"    retained: "
                f"{experiment['pixels_after_trim']}"
            )
            print(
                f"    removed: "
                f"{experiment['pixels_removed']}"
            )
            print(
                f"    temp strength delta: "
                f"{experiment['temperature_strength_delta']:+.3f}"
            )

            print()

        except Exception as error:
            failures.append({
                "file": path.name,
                "error": str(error),
            })

            print(
                f"  ERROR: {error}"
            )
            print()

    # ========================================================
    # SUMMARY
    # ========================================================

    current_strengths = [
        item["current"][
            "temperature_strength"
        ]
        for item in results
    ]

    robust_strengths = [
        item["robust"][
            "temperature_strength"
        ]
        for item in results
    ]

    current_seasons = [
        item["current"][
            "season"
        ]
        for item in results
    ]

    robust_seasons = [
        item["robust"][
            "season"
        ]
        for item in results
    ]

    current_temp = [
        item["current"][
            "temperature"
        ]
        for item in results
    ]

    robust_temp = [
        item["robust"][
            "temperature"
        ]
        for item in results
    ]

    deltas = [
        item["experiment"][
            "temperature_strength_delta"
        ]
        for item in results
    ]

    print(
        "=" * 64
    )
    print(
        "A/B SUMMARY"
    )
    print(
        "=" * 64
    )
    print()

    print(
        "TEMPERATURE STRENGTH"
    )
    print(
        f"  Current mean:  "
        f"{safe_mean(current_strengths)}"
    )
    print(
        f"  Robust mean:   "
        f"{safe_mean(robust_strengths)}"
    )
    print(
        f"  Current median:"
        f" {safe_median(current_strengths)}"
    )
    print(
        f"  Robust median: "
        f"{safe_median(robust_strengths)}"
    )
    print(
        f"  Mean delta:    "
        f"{safe_mean(deltas):+.3f}"
    )
    print()

    print(
        "TEMPERATURE"
    )
    print(
        f"  Current: {current_temp}"
    )
    print(
        f"  Robust:  {robust_temp}"
    )
    print()

    print(
        "SEASONS"
    )
    print(
        f"  Current: {current_seasons}"
    )
    print(
        f"  Robust:  {robust_seasons}"
    )
    print()

    current_neutral = sum(
        season == "Neutral"
        for season in current_seasons
    )

    robust_neutral = sum(
        season == "Neutral"
        for season in robust_seasons
    )

    current_warm = sum(
        season == "Warm Spring"
        for season in current_seasons
    )

    robust_warm = sum(
        season == "Warm Spring"
        for season in robust_seasons
    )

    print(
        "SEASON COUNTS"
    )
    print(
        f"  Current Neutral:      "
        f"{current_neutral}"
    )
    print(
        f"  Robust Neutral:       "
        f"{robust_neutral}"
    )
    print(
        f"  Current Warm Spring:  "
        f"{current_warm}"
    )
    print(
        f"  Robust Warm Spring:   "
        f"{robust_warm}"
    )
    print()

    print(
        "BORDERLINE SAMPLES"
    )

    for item in results:
        current_strength = float(
            item["current"][
                "temperature_strength"
            ]
        )

        robust_strength = float(
            item["robust"][
                "temperature_strength"
            ]
        )

        if (
            0.52 <= current_strength <= 0.62
            or
            0.52 <= robust_strength <= 0.62
        ):
            print(
                f"  {item['file']}"
            )
            print(
                f"    current: "
                f"{current_strength:.3f} "
                f"({item['current']['season']})"
            )
            print(
                f"    robust:  "
                f"{robust_strength:.3f} "
                f"({item['robust']['season']})"
            )

    print()

    print(
        "INTERPRETATION"
    )

    if (
        robust_neutral
        < current_neutral
        and
        robust_warm
        >= current_warm
    ):
        print(
            "  Robust extraction reduces Neutral drift."
        )
    elif (
        robust_neutral
        == current_neutral
        and
        robust_warm
        == current_warm
    ):
        print(
            "  Robust extraction does not change "
            "season distribution."
        )
    else:
        print(
            "  Robust extraction changes the "
            "season distribution."
        )

    print()

    # ========================================================
    # SAVE REPORT
    # ========================================================

    output_dir = Path(
        "validation_samples"
    )

    output_dir.mkdir(
        exist_ok=True
    )

    output_path = (
        output_dir
        /
        "robust_skin_ab_report.json"
    )

    report = {
        "experiment": (
            "Current mean RGB vs "
            "LAB percentile-trimmed median"
        ),
        "engine_modified": False,
        "image_count": len(files),
        "successful": len(results),
        "failed": len(failures),
        "results": results,
        "summary": {
            "current_temperature_strength": {
                "mean": safe_mean(
                    current_strengths
                ),
                "median": safe_median(
                    current_strengths
                ),
            },
            "robust_temperature_strength": {
                "mean": safe_mean(
                    robust_strengths
                ),
                "median": safe_median(
                    robust_strengths
                ),
            },
            "temperature_strength_deltas": deltas,
            "current_temperatures": current_temp,
            "robust_temperatures": robust_temp,
            "current_seasons": current_seasons,
            "robust_seasons": robust_seasons,
        },
        "failures": failures,
    }

    with output_path.open(
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            report,
            file,
            indent=2
        )

    print(
        f"Saved report: {output_path}"
    )
    print()

    print(
        "=" * 64
    )
    print(
        "EXPERIMENT COMPLETE"
    )
    print(
        "=" * 64
    )
    print()

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
