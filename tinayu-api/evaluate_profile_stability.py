"""
Tinayu Profile Stability + Diagnostic Evaluation

Purpose:
    Evaluate consistency across multiple real photos of the same person
    and expose the color measurements responsible for profile changes.

This is NOT a ground-truth personal-color accuracy test.

The evaluator:
    - runs the real Tinayu analysis pipeline
    - prints raw skin RGB
    - prints normalized skin RGB
    - calculates raw LAB
    - reports normalized LAB
    - reports raw/normalized hue and chroma
    - reports temperature strength
    - reports distance from temperature boundaries
    - reports season / temperature / saturation / contrast
    - reports image quality and skin pixel count
    - saves a detailed JSON diagnostic report

No Tinayu engine values are modified.
"""

import json
import os
import sys
from collections import Counter

import numpy as np

from tinayu_engine import (
    analyze_image,
    rgb_to_lab,
    rgb_to_hsv_features,
    temperature_score,
)


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_DIRECTORY = "validation_samples"
REPORT_FILENAME = "tinayu_stability_diagnostic_report.json"


# ============================================================
# HELPERS
# ============================================================

def read_image_bytes(path):
    with open(path, "rb") as file:
        return file.read()


def calculate_chroma(lab):
    lab = np.asarray(lab, dtype=float)

    return float(
        np.sqrt(
            lab[1] ** 2
            +
            lab[2] ** 2
        )
    )


def calculate_color_features(rgb):
    """
    Calculate LAB + HSV-derived hue/chroma from an RGB value.
    """

    rgb_array = np.asarray(
        rgb,
        dtype=np.uint8
    )

    lab = rgb_to_lab(
        rgb_array
    )

    hsv = rgb_to_hsv_features(
        rgb_array
    )

    return {
        "rgb": rgb_array.astype(int).tolist(),

        "lab": {
            "L": round(float(lab[0]), 2),
            "a": round(float(lab[1]), 2),
            "b": round(float(lab[2]), 2),
        },

        "hue": round(
            float(hsv["hue"]),
            2
        ),

        "chroma": round(
            calculate_chroma(lab),
            2
        )
    }


def calculate_temperature_diagnostics(lab):
    """
    Explain where the temperature strength sits relative to
    Tinayu's current season boundaries.

    Current engine boundaries:

        <= 0.42  -> Cool
        0.42-0.55 -> Neutral
        0.55-0.58 -> Borderline Warm
        >= 0.58 -> Warm season

    This function does NOT alter those boundaries.
    """

    temperature_strength = temperature_score(
        lab[1],
        lab[2]
    )

    strength = float(
        temperature_strength
    )

    if strength <= 0.42:
        band = "Strong Cool"

    elif strength < 0.45:
        band = "Borderline Cool"

    elif strength < 0.55:
        band = "Neutral"

    elif strength < 0.58:
        band = "Borderline Warm"

    else:
        band = "Strong Warm"

    return {
        "temperature_strength": round(
            strength,
            3
        ),

        "temperature_strength_band": band,

        "distance_to_neutral_warm_boundary": round(
            strength - 0.55,
            3
        ),

        "distance_to_strong_warm_boundary": round(
            strength - 0.58,
            3
        ),

        "distance_to_cool_neutral_boundary": round(
            strength - 0.42,
            3
        )
    }


def safe_number(value, digits=3):
    if value is None:
        return None

    try:
        return round(
            float(value),
            digits
        )
    except (TypeError, ValueError):
        return None


def get_profile_value(profile, *keys, default=None):
    """
    Safely walk nested dictionaries.
    """

    current = profile

    for key in keys:
        if not isinstance(current, dict):
            return default

        current = current.get(
            key,
            default
        )

        if current is default:
            return default

    return current


def get_recommendation_names(recommendations, category):
    """
    Extract recommendation names when the engine returns the
    standard structured recommendation format.

    This is intentionally defensive because recommendation
    formatting is not the focus of this diagnostic evaluator.
    """

    if not isinstance(recommendations, dict):
        return []

    values = recommendations.get(
        category,
        []
    )

    if not isinstance(values, list):
        return []

    names = []

    for item in values:

        if isinstance(item, str):
            names.append(item)
            continue

        if not isinstance(item, dict):
            continue

        name = (
            item.get("name")
            or item.get("color")
            or item.get("label")
        )

        if name:
            names.append(
                str(name)
            )

    return names


def overlap_percentage(list_a, list_b):
    """
    Calculate percentage overlap using the smaller list as the
    denominator.
    """

    if not list_a or not list_b:
        return None

    set_a = set(list_a)
    set_b = set(list_b)

    denominator = min(
        len(set_a),
        len(set_b)
    )

    if denominator == 0:
        return None

    overlap = len(
        set_a.intersection(set_b)
    )

    return round(
        overlap / denominator * 100.0,
        1
    )


# ============================================================
# SINGLE IMAGE ANALYSIS
# ============================================================

def analyze_validation_image(path):
    """
    Run Tinayu's real analysis pipeline and build a diagnostic
    record without changing the engine.
    """

    image_bytes = read_image_bytes(
        path
    )

    result = analyze_image(
        image_bytes
    )

    if not result.get("success"):
        return {
            "success": False,
            "error": result.get(
                "error",
                "Unknown analysis error"
            )
        }

    colors = result.get(
        "colors",
        {}
    )

    profile = result.get(
        "profile",
        {}
    )

    quality = result.get(
        "quality",
        {}
    )

    raw_rgb = colors.get(
        "skin_raw_rgb"
    )

    normalized_rgb = colors.get(
        "skin_normalized_rgb"
    )

    if raw_rgb is None:
        raise ValueError(
            "Analysis result is missing colors.skin_raw_rgb"
        )

    if normalized_rgb is None:
        raise ValueError(
            "Analysis result is missing colors.skin_normalized_rgb"
        )

    # --------------------------------------------------------
    # RAW COLOR FEATURES
    # --------------------------------------------------------

    raw_features = calculate_color_features(
        raw_rgb
    )

    raw_lab_array = rgb_to_lab(
        np.asarray(
            raw_rgb,
            dtype=np.uint8
        )
    )

    # --------------------------------------------------------
    # NORMALIZED COLOR FEATURES
    # --------------------------------------------------------

    normalized_features = calculate_color_features(
        normalized_rgb
    )

    normalized_lab_array = rgb_to_lab(
        np.asarray(
            normalized_rgb,
            dtype=np.uint8
        )
    )

    # --------------------------------------------------------
    # TEMPERATURE DIAGNOSTICS
    # --------------------------------------------------------

    raw_temperature = calculate_temperature_diagnostics(
        raw_lab_array
    )

    normalized_temperature = calculate_temperature_diagnostics(
        normalized_lab_array
    )

    # --------------------------------------------------------
    # PROFILE VALUES
    # --------------------------------------------------------

    heuristics = profile.get(
        "heuristics",
        {}
    )

    season = heuristics.get(
        "suggested_season"
    )

    temperature = heuristics.get(
        "temperature"
    )

    saturation = heuristics.get(
        "skin_saturation"
    )

    contrast = heuristics.get(
        "contrast_level"
    )

    # --------------------------------------------------------
    # NORMALIZED DELTAS
    # --------------------------------------------------------

    normalized_delta = {
        "rgb": [
            int(normalized_rgb[index])
            -
            int(raw_rgb[index])
            for index in range(3)
        ],

        "lab": {
            "L": round(
                normalized_features["lab"]["L"]
                -
                raw_features["lab"]["L"],
                2
            ),

            "a": round(
                normalized_features["lab"]["a"]
                -
                raw_features["lab"]["a"],
                2
            ),

            "b": round(
                normalized_features["lab"]["b"]
                -
                raw_features["lab"]["b"],
                2
            )
        },

        "hue": round(
            normalized_features["hue"]
            -
            raw_features["hue"],
            2
        ),

        "chroma": round(
            normalized_features["chroma"]
            -
            raw_features["chroma"],
            2
        ),

        "temperature_strength": round(
            normalized_temperature[
                "temperature_strength"
            ]
            -
            raw_temperature[
                "temperature_strength"
            ],
            3
        )
    }

    # --------------------------------------------------------
    # RECOMMENDATIONS
    # --------------------------------------------------------

    recommendations = result.get(
        "recommendations",
        {}
    )

    recommendation_lists = {
        "clothing": get_recommendation_names(
            recommendations,
            "clothing"
        ),

        "makeup": get_recommendation_names(
            recommendations,
            "makeup"
        ),

        "accents": get_recommendation_names(
            recommendations,
            "accents"
        )
    }

    # --------------------------------------------------------
    # FINAL DIAGNOSTIC RECORD
    # --------------------------------------------------------

    return {
        "success": True,

        "file": os.path.basename(
            path
        ),

        "path": os.path.abspath(
            path
        ),

        "image": result.get(
            "image",
            {}
        ),

        "face": result.get(
            "face",
            {}
        ),

        "quality": {
            "image_confidence": quality.get(
                "image_confidence"
            ),

            "skin_pixels": quality.get(
                "skin_pixels_analyzed"
            ),

            "normalization_applied": quality.get(
                "normalization_applied"
            ),

            "hair_detected": quality.get(
                "hair_detected"
            ),

            "eyes_detected": quality.get(
                "eyes_detected"
            )
        },

        "raw_skin": raw_features,

        "normalized_skin": normalized_features,

        "normalization_delta": normalized_delta,

        "temperature_diagnostics": {
            "raw": raw_temperature,
            "normalized": normalized_temperature
        },

        "profile": {
            "season": season,
            "temperature": temperature,
            "temperature_strength": heuristics.get(
                "temperature_strength"
            ),
            "temperature_strength_band": heuristics.get(
                "temperature_strength_band"
            ),
            "skin_category": heuristics.get(
                "skin_category"
            ),
            "skin_depth": heuristics.get(
                "skin_depth"
            ),
            "skin_saturation": saturation,
            "contrast": contrast
        },

        "recommendations": recommendation_lists
    }


# ============================================================
# PRINT DIAGNOSTIC RECORD
# ============================================================

def print_diagnostic(record, index, total):
    """
    Print one image's complete diagnostic summary.
    """

    print()
    print(
        f"[{index}/{total}] "
        f"{record['file']}"
    )

    print(
        "  "
        f"season: {record['profile']['season']} | "
        f"temp: {record['profile']['temperature']} | "
        f"temp strength: "
        f"{record['profile']['temperature_strength']} | "
        f"saturation: "
        f"{record['profile']['skin_saturation']} | "
        f"contrast: "
        f"{record['profile']['contrast']}"
    )

    print(
        "  "
        f"quality: "
        f"{record['quality']['image_confidence']} | "
        f"skin pixels: "
        f"{record['quality']['skin_pixels']}"
    )

    print()
    print("  RAW SKIN")

    print(
        "    RGB: "
        f"{record['raw_skin']['rgb']}"
    )

    print(
        "    LAB: "
        f"{record['raw_skin']['lab']}"
    )

    print(
        "    hue: "
        f"{record['raw_skin']['hue']} | "
        f"chroma: "
        f"{record['raw_skin']['chroma']}"
    )

    print(
        "    temperature strength: "
        f"{record['temperature_diagnostics']['raw']['temperature_strength']}"
    )

    print()
    print("  NORMALIZED SKIN")

    print(
        "    RGB: "
        f"{record['normalized_skin']['rgb']}"
    )

    print(
        "    LAB: "
        f"{record['normalized_skin']['lab']}"
    )

    print(
        "    hue: "
        f"{record['normalized_skin']['hue']} | "
        f"chroma: "
        f"{record['normalized_skin']['chroma']}"
    )

    print(
        "    temperature strength: "
        f"{record['temperature_diagnostics']['normalized']['temperature_strength']}"
    )

    print()
    print("  NORMALIZATION DELTA")

    print(
        "    RGB delta: "
        f"{record['normalization_delta']['rgb']}"
    )

    print(
        "    LAB delta: "
        f"{record['normalization_delta']['lab']}"
    )

    print(
        "    hue delta: "
        f"{record['normalization_delta']['hue']}"
        " | "
        "chroma delta: "
        f"{record['normalization_delta']['chroma']}"
    )

    print(
        "    temperature strength delta: "
        f"{record['normalization_delta']['temperature_strength']}"
    )

    print()
    print("  TEMPERATURE BOUNDARY")

    print(
        "    band: "
        f"{record['temperature_diagnostics']['normalized']['temperature_strength_band']}"
    )

    print(
        "    distance to 0.55 Neutral/Warm boundary: "
        f"{record['temperature_diagnostics']['normalized']['distance_to_neutral_warm_boundary']}"
    )

    print(
        "    distance to 0.58 Strong Warm boundary: "
        f"{record['temperature_diagnostics']['normalized']['distance_to_strong_warm_boundary']}"
    )


# ============================================================
# SUMMARY
# ============================================================

def print_summary(records):
    successful = [
        record
        for record in records
        if record.get("success")
    ]

    failed = [
        record
        for record in records
        if not record.get("success")
    ]

    print()
    print("=" * 60)
    print("PIPELINE SUCCESS")
    print("=" * 60)

    print(
        f"Successful: "
        f"{len(successful)}/{len(records)} "
        f"("
        f"{(
            len(successful) / len(records) * 100
            if records
            else 0
        ):0.1f}%"
        ")"
    )

    print(
        f"Failed:     "
        f"{len(failed)}"
    )

    if failed:
        print()
        print("Failures:")

        for record in failed:
            print(
                f"  - {record.get('file', 'unknown')}: "
                f"{record.get('error', 'unknown error')}"
            )

    if not successful:
        return

    # --------------------------------------------------------
    # PROFILE CONSISTENCY
    # --------------------------------------------------------

    seasons = [
        record["profile"]["season"]
        for record in successful
    ]

    temperatures = [
        record["profile"]["temperature"]
        for record in successful
    ]

    saturation = [
        record["profile"]["skin_saturation"]
        for record in successful
    ]

    contrast = [
        record["profile"]["contrast"]
        for record in successful
    ]

    strengths = [
        float(
            record["profile"]["temperature_strength"]
        )
        for record in successful
        if record["profile"]["temperature_strength"]
        is not None
    ]

    print()
    print("=" * 60)
    print("PROFILE CONSISTENCY")
    print("=" * 60)

    print(
        f"Seasons: "
        f"{dict(Counter(seasons))}"
    )

    print(
        f"Temperature: "
        f"{temperatures}"
    )

    print(
        f"Temperature strength: "
        f"{[
            round(value, 3)
            for value in strengths
        ]}"
    )

    print(
        f"Saturation: "
        f"{saturation}"
    )

    print(
        f"Contrast: "
        f"{contrast}"
    )

    dominant_season = Counter(
        seasons
    ).most_common(1)[0]

    dominant_name = dominant_season[0]
    dominant_count = dominant_season[1]

    print(
        f"Dominant season: "
        f"{dominant_name} "
        f"("
        f"{dominant_count / len(seasons) * 100:0.1f}%"
        " of successful analyses)"
    )

    if strengths:
        print()
        print("Temperature strength statistics:")

        print(
            f"  Mean: "
            f"{np.mean(strengths):.3f}"
        )

        print(
            f"  Median: "
            f"{np.median(strengths):.3f}"
        )

        print(
            f"  Min: "
            f"{np.min(strengths):.3f}"
        )

        print(
            f"  Max: "
            f"{np.max(strengths):.3f}"
        )

        print(
            f"  Range: "
            f"{np.max(strengths) - np.min(strengths):.3f}"
        )

    # --------------------------------------------------------
    # BORDERLINE ANALYSIS
    # --------------------------------------------------------

    borderline = []

    for record in successful:

        strength = float(
            record["profile"]["temperature_strength"]
        )

        if 0.52 <= strength <= 0.62:

            borderline.append(
                (
                    record["file"],
                    strength,
                    record["profile"]["season"],
                    record["profile"]["temperature"]
                )
            )

    print()
    print("=" * 60)
    print("BORDERLINE TEMPERATURE ANALYSIS")
    print("=" * 60)

    print(
        "Focus range: "
        "temperature strength 0.52 - 0.62"
    )

    if not borderline:

        print(
            "No images fall inside the diagnostic range."
        )

    else:

        for (
            filename,
            strength,
            season,
            temperature
        ) in borderline:

            print(
                f"{filename}"
            )

            print(
                f"  strength: {strength:.3f} | "
                f"temperature: {temperature} | "
                f"season: {season}"
            )

            print(
                f"  from 0.55 boundary: "
                f"{strength - 0.55:+.3f}"
            )

            print(
                f"  from 0.58 warm-season boundary: "
                f"{strength - 0.58:+.3f}"
            )

    # --------------------------------------------------------
    # NORMALIZED SKIN STABILITY
    # --------------------------------------------------------

    normalized_lab = [
        record["normalized_skin"]["lab"]
        for record in successful
    ]

    normalized_hue = [
        record["normalized_skin"]["hue"]
        for record in successful
    ]

    normalized_chroma = [
        record["normalized_skin"]["chroma"]
        for record in successful
    ]

    L_values = [
        value["L"]
        for value in normalized_lab
    ]

    a_values = [
        value["a"]
        for value in normalized_lab
    ]

    b_values = [
        value["b"]
        for value in normalized_lab
    ]

    print()
    print("=" * 60)
    print("NORMALIZED SKIN STABILITY")
    print("=" * 60)

    print(
        f"LAB L   mean: "
        f"{np.mean(L_values):.2f} "
        f"median: "
        f"{np.median(L_values):.2f}"
    )

    print(
        f"LAB a   mean: "
        f"{np.mean(a_values):.2f} "
        f"median: "
        f"{np.median(a_values):.2f}"
    )

    print(
        f"LAB b   mean: "
        f"{np.mean(b_values):.2f} "
        f"median: "
        f"{np.median(b_values):.2f}"
    )

    print(
        f"Hue     mean: "
        f"{np.mean(normalized_hue):.2f} "
        f"median: "
        f"{np.median(normalized_hue):.2f}"
    )

    print(
        f"Chroma  mean: "
        f"{np.mean(normalized_chroma):.2f} "
        f"median: "
        f"{np.median(normalized_chroma):.2f}"
    )

    print(
        f"L range: "
        f"{np.ptp(L_values):.2f}"
    )

    print(
        f"a range: "
        f"{np.ptp(a_values):.2f}"
    )

    print(
        f"b range: "
        f"{np.ptp(b_values):.2f}"
    )

    print(
        f"Hue range: "
        f"{np.ptp(normalized_hue):.2f}"
    )

    print(
        f"Chroma range: "
        f"{np.ptp(normalized_chroma):.2f}"
    )

    # --------------------------------------------------------
    # NORMALIZATION EFFECT
    # --------------------------------------------------------

    raw_strengths = [
        record[
            "temperature_diagnostics"
        ][
            "raw"
        ][
            "temperature_strength"
        ]
        for record in successful
    ]

    normalized_strengths = [
        record[
            "temperature_diagnostics"
        ][
            "normalized"
        ][
            "temperature_strength"
        ]
        for record in successful
    ]

    strength_deltas = [
        normalized_strengths[index]
        -
        raw_strengths[index]
        for index in range(
            len(successful)
        )
    ]

    print()
    print("=" * 60)
    print("NORMALIZATION EFFECT ON TEMPERATURE")
    print("=" * 60)

    print(
        "Raw strength:        "
        f"{[
            round(value, 3)
            for value in raw_strengths
        ]}"
    )

    print(
        "Normalized strength: "
        f"{[
            round(value, 3)
            for value in normalized_strengths
        ]}"
    )

    print(
        "Strength delta:      "
        f"{[
            round(value, 3)
            for value in strength_deltas
        ]}"
    )

    print(
        f"Mean strength change: "
        f"{np.mean(strength_deltas):+.3f}"
    )

    # --------------------------------------------------------
    # NORMALIZATION CHECK
    # --------------------------------------------------------

    normalization_count = sum(
        1
        for record in successful
        if record["quality"]["normalization_applied"]
    )

    print()
    print("=" * 60)
    print("NORMALIZATION")
    print("=" * 60)

    print(
        f"Normalization applied: "
        f"{normalization_count}/"
        f"{len(successful)} "
        f"("
        f"{normalization_count / len(successful) * 100:0.1f}%"
        ")"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    if len(sys.argv) > 1:
        validation_directory = sys.argv[1]
    else:
        validation_directory = DEFAULT_DIRECTORY

    if not os.path.isdir(
        validation_directory
    ):

        print(
            "ERROR: Validation directory does not exist:"
        )

        print(
            os.path.abspath(
                validation_directory
            )
        )

        sys.exit(1)

    # --------------------------------------------------------
    # DISCOVER IMAGES
    # --------------------------------------------------------

    supported_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
        ".bmp"
    }

    image_paths = []

    for filename in sorted(
        os.listdir(
            validation_directory
        )
    ):

        extension = os.path.splitext(
            filename
        )[1].lower()

        if extension not in supported_extensions:
            continue

        image_paths.append(
            os.path.join(
                validation_directory,
                filename
            )
        )

    print()
    print("=" * 60)
    print("TINAYU PROFILE STABILITY + DIAGNOSTIC EVALUATION")
    print("=" * 60)

    print()
    print(
        "This test measures consistency across photos of the same"
    )

    print(
        "person. It is NOT a ground-truth accuracy test."
    )

    print()
    print(
        "Diagnostic mode:"
    )

    print(
        "- raw RGB/LAB/hue/chroma"
    )

    print(
        "- normalized RGB/LAB/hue/chroma"
    )

    print(
        "- temperature strength"
    )

    print(
        "- temperature boundary distances"
    )

    print(
        "- normalization effect"
    )

    print(
        "- image quality"
    )

    print()
    print(
        "Directory: "
        f"{os.path.abspath(validation_directory)}"
    )

    print(
        "Images:    "
        f"{len(image_paths)}"
    )

    print()
    print(
        "No Tinayu engine values will be modified."
    )

    # --------------------------------------------------------
    # RUN ANALYSIS
    # --------------------------------------------------------

    records = []

    for index, path in enumerate(
        image_paths,
        start=1
    ):

        try:

            record = analyze_validation_image(
                path
            )

        except Exception as error:

            record = {
                "success": False,
                "file": os.path.basename(
                    path
                ),
                "path": os.path.abspath(
                    path
                ),
                "error": (
                    f"{type(error).__name__}: "
                    f"{error}"
                )
            }

        records.append(
            record
        )

        if record.get("success"):

            print_diagnostic(
                record,
                index,
                len(image_paths)
            )

        else:

            print()
            print(
                f"[{index}/{len(image_paths)}] "
                f"{os.path.basename(path)}"
            )

            print(
                "  ERROR: "
                f"{record.get('error')}"
            )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print_summary(
        records
    )

    # --------------------------------------------------------
    # SAVE JSON REPORT
    # --------------------------------------------------------

    report_path = os.path.join(
        validation_directory,
        REPORT_FILENAME
    )

    report = {
        "tool": (
            "Tinayu Profile Stability "
            "+ Diagnostic Evaluation"
        ),

        "ground_truth_test": False,

        "engine_modified": False,

        "validation_directory": os.path.abspath(
            validation_directory
        ),

        "image_count": len(
            image_paths
        ),

        "successful": sum(
            1
            for record in records
            if record.get("success")
        ),

        "failed": sum(
            1
            for record in records
            if not record.get("success")
        ),

        "records": records
    }

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            report,
            file,
            indent=2
        )

    print()
    print("=" * 60)
    print("REPORT")
    print("=" * 60)

    print()
    print(
        f"Saved: {report_path}"
    )

    print()
    print(
        "Important:"
    )

    print(
        "- This evaluates stability, not personal-color ground truth."
    )

    print(
        "- A stable season does not prove the season is objectively correct."
    )

    print(
        "- The diagnostic output is intended to identify why borderline"
    )

    print(
        "  images move between Neutral and Warm Spring."
    )


if __name__ == "__main__":
    main()