"""
Tinayu Color Profile Evaluation

Evaluation-only test harness for:
- temperature classification
- skin depth classification
- skin saturation classification
- contrast classification
- season suggestion

This script tests the CURRENT classifier behavior.
It does not modify the Tinayu engine.

Run from tinayu-api with the project virtual environment active:

    python evaluate_color_profiles.py
"""

import sys

from tinayu_engine import (
    classify_temperature,
    classify_skin_category,
    classify_skin_saturation,
    classify_contrast,
    suggest_season,
)


# ============================================================
# Helpers
# ============================================================

def check_result(
    name,
    actual,
    expected,
    failures,
):
    """Compare an actual result against the expected result."""

    if actual == expected:
        print(f"[PASS] {name}: {actual}")
        return True

    print(
        f"[FAIL] {name}: "
        f"expected {expected}, got {actual}"
    )

    failures.append(
        f"{name}: expected {expected}, got {actual}"
    )

    return False


# ============================================================
# Temperature Tests
# ============================================================

def test_temperature(failures):
    print()
    print("=" * 60)
    print("TEMPERATURE CLASSIFICATION")
    print("=" * 60)

    cases = [
        {
            "name": "warm_light",
            "a": 10,
            "b": 20,
            "expected": "Warm",
        },
        {
            "name": "warm_medium",
            "a": 15,
            "b": 25,
            "expected": "Warm",
        },
        {
            "name": "warm_deep",
            "a": 8,
            "b": 30,
            "expected": "Warm",
        },
        {
            "name": "cool_light",
            "a": 5,
            "b": 5,
            "expected": "Cool",
        },
        {
            "name": "cool_medium",
            "a": 5,
            "b": 0,
            "expected": "Cool",
        },
        {
            "name": "cool_deep",
            "a": 8,
            "b": -10,
            "expected": "Cool",
        },
        {
            "name": "neutral_light",
            "a": 9,
            "b": 10,
            "expected": "Neutral",
        },
        {
            "name": "neutral_medium",
            "a": 12,
            "b": 12,
            "expected": "Neutral",
        },
        {
            "name": "neutral_deep",
            "a": 15,
            "b": 15,
            "expected": "Neutral",
        },
    ]

    for case in cases:
        actual = classify_temperature(
            case["a"],
            case["b"],
        )

        check_result(
            case["name"],
            actual,
            case["expected"],
            failures,
        )


def test_temperature_boundaries(failures):
    print()
    print("=" * 60)
    print("TEMPERATURE BOUNDARIES")
    print("=" * 60)

    cases = [
        {
            "name": "cool_edge_1",
            "a": 8,
            "b": 10,
            "expected": "Cool",
        },
        {
            "name": "cool_edge_2",
            "a": 8,
            "b": 9,
            "expected": "Cool",
        },
        {
            "name": "cool_edge_3",
            "a": 7,
            "b": 10,
            "expected": "Cool",
        },
        {
            "name": "cool_outside_a",
            "a": 9,
            "b": 10,
            "expected": "Neutral",
        },
        {
            "name": "cool_outside_b",
            "a": 8,
            "b": 11,
            "expected": "Neutral",
        },
        {
            "name": "warm_edge_1",
            "a": 8,
            "b": 18,
            "expected": "Warm",
        },
        {
            "name": "warm_edge_2",
            "a": 9,
            "b": 18,
            "expected": "Warm",
        },
        {
            "name": "warm_outside_a",
            "a": 7,
            "b": 18,
            "expected": "Neutral",
        },
        {
            "name": "warm_outside_b",
            "a": 8,
            "b": 17,
            "expected": "Neutral",
        },
    ]

    for case in cases:
        actual = classify_temperature(
            case["a"],
            case["b"],
        )

        check_result(
            case["name"],
            actual,
            case["expected"],
            failures,
        )


# ============================================================
# Skin Depth Tests
# ============================================================

def test_skin_depth(failures):
    print()
    print("=" * 60)
    print("SKIN DEPTH CLASSIFICATION")
    print("=" * 60)

    cases = [
        {
            "name": "depth_34_9",
            "lightness": 34.9,
            "expected": "Deep",
        },
        {
            "name": "depth_35",
            "lightness": 35,
            "expected": "Medium",
        },
        {
            "name": "depth_49_9",
            "lightness": 49.9,
            "expected": "Medium",
        },
        {
            "name": "depth_50",
            "lightness": 50,
            "expected": "Medium-Light",
        },
        {
            "name": "depth_67_9",
            "lightness": 67.9,
            "expected": "Medium-Light",
        },
        {
            "name": "depth_68",
            "lightness": 68,
            "expected": "Light",
        },
        {
            "name": "depth_80",
            "lightness": 80,
            "expected": "Light",
        },
    ]

    for case in cases:
        actual = classify_skin_category(
            case["lightness"]
        )

        check_result(
            case["name"],
            actual,
            case["expected"],
            failures,
        )


# ============================================================
# Skin Saturation Tests
# ============================================================

def test_skin_saturation(failures):
    print()
    print("=" * 60)
    print("SKIN SATURATION CLASSIFICATION")
    print("=" * 60)

    cases = [
        {
            "name": "saturation_17_9",
            "chroma": 17.9,
            "expected": "Muted",
        },
        {
            "name": "saturation_18",
            "chroma": 18,
            "expected": "Moderate",
        },
        {
            "name": "saturation_29_9",
            "chroma": 29.9,
            "expected": "Moderate",
        },
        {
            "name": "saturation_30",
            "chroma": 30,
            "expected": "Clear",
        },
        {
            "name": "saturation_40",
            "chroma": 40,
            "expected": "Clear",
        },
    ]

    for case in cases:
        actual = classify_skin_saturation(
            case["chroma"]
        )

        check_result(
            case["name"],
            actual,
            case["expected"],
            failures,
        )


# ============================================================
# Contrast Tests
# ============================================================

def get_contrast_level(result):
    """
    classify_contrast returns:

        (metrics, level)

    This helper safely extracts the level.
    """

    if isinstance(result, tuple):
        if len(result) >= 2:
            return result[1]

    return result


def test_contrast(failures):
    print()
    print("=" * 60)
    print("CONTRAST CLASSIFICATION")
    print("=" * 60)

    cases = [
        {
            "name": "contrast_low",
            "skin": (100, 100, 100),
            "hair": (120, 120, 120),
            "eyes": None,
            "expected": "Low",
        },
        {
            "name": "contrast_moderate",
            "skin": (100, 100, 100),
            "hair": (160, 160, 160),
            "eyes": None,
            "expected": "Moderate",
        },
        {
            "name": "contrast_high",
            "skin": (40, 40, 40),
            "hair": (180, 180, 180),
            "eyes": None,
            "expected": "High",
        },
       {
    "name": "contrast_high_with_eyes",
    "skin": (60, 60, 60),
    "hair": (120, 120, 120),
    "eyes": (240, 240, 240),
    "expected": "High",
},
    ]

    for case in cases:
        result = classify_contrast(
            skin_rgb=case["skin"],
            hair_rgb=case["hair"],
            eye_rgb=case["eyes"],
        )

        actual = get_contrast_level(result)

        check_result(
            case["name"],
            actual,
            case["expected"],
            failures,
        )


# ============================================================
# Season Tests
# ============================================================

def test_season_matrix(failures):
    print()
    print("=" * 60)
    print("SEASON CLASSIFICATION")
    print("=" * 60)

    cases = [
        {
            "name": "warm_clear_light",
            "temperature": "Warm",
            "saturation": "Clear",
            "depth": "Light",
            "expected": "Warm Spring",
        },
        {
            "name": "warm_clear_medium",
            "temperature": "Warm",
            "saturation": "Clear",
            "depth": "Medium",
            "expected": "Warm Spring",
        },
        {
            "name": "warm_muted_light",
            "temperature": "Warm",
            "saturation": "Muted",
            "depth": "Light",
            "expected": "Warm Autumn",
        },
        {
            "name": "warm_muted_medium",
            "temperature": "Warm",
            "saturation": "Muted",
            "depth": "Medium",
            "expected": "Warm Autumn",
        },
        {
            "name": "warm_moderate_medium",
            "temperature": "Warm",
            "saturation": "Moderate",
            "depth": "Medium",
            "expected": "Warm Autumn",
        },
        {
            "name": "cool_clear_light",
            "temperature": "Cool",
            "saturation": "Clear",
            "depth": "Light",
            "expected": "Cool Winter",
        },
        {
            "name": "cool_clear_deep",
            "temperature": "Cool",
            "saturation": "Clear",
            "depth": "Deep",
            "expected": "Cool Winter",
        },
        {
            "name": "cool_muted_light",
            "temperature": "Cool",
            "saturation": "Muted",
            "depth": "Light",
            "expected": "Cool Summer",
        },
        {
            "name": "neutral_clear_light",
            "temperature": "Neutral",
            "saturation": "Clear",
            "depth": "Light",
            "expected": "Spring",
        },
        {
            "name": "neutral_muted_light",
            "temperature": "Neutral",
            "saturation": "Muted",
            "depth": "Light",
            "expected": "Autumn",
        },
    ]

    for case in cases:
        actual = suggest_season(
            temperature=case["temperature"],
            saturation=case["saturation"],
            skin_depth=case["depth"],
        )

        check_result(
            case["name"],
            actual,
            case["expected"],
            failures,
        )


# ============================================================
# Combined Profile Tests
# ============================================================

def test_combined_profiles(failures):
    print()
    print("=" * 60)
    print("COMBINED COLOR PROFILES")
    print("=" * 60)

    profiles = [
        {
            "name": "warm_spring_profile",
            "temperature": "Warm",
            "saturation": "Clear",
            "depth": "Light",
            "expected": "Warm Spring",
        },
        {
            "name": "warm_autumn_profile",
            "temperature": "Warm",
            "saturation": "Muted",
            "depth": "Medium",
            "expected": "Warm Autumn",
        },
        {
            "name": "cool_winter_profile",
            "temperature": "Cool",
            "saturation": "Clear",
            "depth": "Deep",
            "expected": "Cool Winter",
        },
        {
            "name": "cool_summer_profile",
            "temperature": "Cool",
            "saturation": "Muted",
            "depth": "Light",
            "expected": "Cool Summer",
        },
        {
            "name": "neutral_spring_profile",
            "temperature": "Neutral",
            "saturation": "Clear",
            "depth": "Light",
            "expected": "Spring",
        },
        {
            "name": "neutral_autumn_profile",
            "temperature": "Neutral",
            "saturation": "Muted",
            "depth": "Medium",
            "expected": "Autumn",
        },
    ]

    for profile in profiles:
        actual = suggest_season(
            temperature=profile["temperature"],
            saturation=profile["saturation"],
            skin_depth=profile["depth"],
        )

        check_result(
            profile["name"],
            actual,
            profile["expected"],
            failures,
        )


# ============================================================
# Main
# ============================================================

def main():
    """Run the complete Tinayu color-profile evaluation."""

    failures = []

    print()
    print("=" * 60)
    print("TINAYU COLOR PROFILE EVALUATION")
    print("=" * 60)
    print()
    print("Evaluation mode: classifier validation only")
    print("No Tinayu engine values will be modified.")

    test_temperature(failures)
    test_temperature_boundaries(failures)
    test_skin_depth(failures)
    test_skin_saturation(failures)
    test_contrast(failures)
    test_season_matrix(failures)
    test_combined_profiles(failures)

    total_checks = (
        9
        + 9
        + 7
        + 5
        + 4
        + 10
        + 6
    )

    passed_checks = total_checks - len(failures)

    print()
    print("=" * 60)
    print("FINAL RESULT")
    print("=" * 60)

    print()
    print(
        f"Checks passed: {passed_checks}/{total_checks}"
    )

    print(
        f"Checks failed: {len(failures)}/{total_checks}"
    )

    if failures:
        print()
        print("Failed checks:")

        for failure in failures:
            print(f"  - {failure}")

        print()
        print("RESULT: FAIL")

        return 1

    print()
    print("RESULT: ALL CHECKS PASSED")

    return 0


if __name__ == "__main__":
    main()