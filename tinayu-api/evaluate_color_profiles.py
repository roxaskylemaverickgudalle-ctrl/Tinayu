
"""
Tinayu Color Profile Evaluation v2

Evaluation-only test harness for:

- temperature classification
- temperature boundaries
- temperature grid sweep
- skin depth classification
- skin depth boundaries
- skin saturation classification
- skin saturation boundaries
- contrast classification
- season classification
- full warm season matrix
- full cool season matrix
- full neutral season matrix
- explicit season boundaries
- combined color profiles
- season coverage

This script tests the CURRENT classifier behavior.

It does not modify the Tinayu engine.

Run from tinayu-api with the project virtual environment active:

    python evaluate_color_profiles.py
"""

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

SECTION_COUNTS = {
    "Temperature classification": 9,
    "Temperature boundaries": 14,
    "Temperature sweep": 14,
    "Skin depth classification": 7,
    "Skin depth boundaries": 9,
    "Skin saturation classification": 5,
    "Skin saturation boundaries": 6,
    "Contrast classification": 4,
    "Season matrix": 10,
    "Warm season matrix": 12,
    "Cool season matrix": 12,
    "Neutral season matrix": 12,
    "Season boundaries": 13,
    "Combined profiles": 6,
    "Season coverage": 1,
}


def check_result(name, actual, expected, failures):
    """Compare an actual result against an expected result."""

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


def print_section(title):
    print()
    print("=" * 60)
    print(title)
    print("=" * 60)


# ============================================================
# Temperature Classification
# ============================================================

def test_temperature(failures):
    print_section("TEMPERATURE CLASSIFICATION")

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


# ============================================================
# Temperature Boundaries
# ============================================================

def test_temperature_boundaries(failures):
    print_section("TEMPERATURE BOUNDARIES")

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
        {
            "name": "cool_below_boundary",
            "a": 0,
            "b": 10,
            "expected": "Cool",
        },
        {
            "name": "neutral_between_cool_and_warm",
            "a": 8,
            "b": 14,
            "expected": "Neutral",
        },
        {
            "name": "warm_above_boundary",
            "a": 8,
            "b": 19,
            "expected": "Warm",
        },
        {
            "name": "warm_a_below_threshold",
            "a": 7,
            "b": 19,
            "expected": "Neutral",
        },
        {
            "name": "warm_b_below_threshold",
            "a": 8,
            "b": 17.9,
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
# Temperature Grid Sweep
# ============================================================

def test_temperature_grid(failures):
    print_section("TEMPERATURE GRID SWEEP")

    cases = [
        {
            "name": "grid_cool_1",
            "a": -5,
            "b": -5,
            "expected": "Cool",
        },
        {
            "name": "grid_cool_2",
            "a": 0,
            "b": 0,
            "expected": "Cool",
        },
        {
            "name": "grid_cool_3",
            "a": 5,
            "b": 5,
            "expected": "Cool",
        },
        {
            # This point is intentionally Neutral because
            # b=15 is outside the Cool b<=10 boundary.
            "name": "grid_cool_4",
            "a": -2,
            "b": 15,
            "expected": "Neutral",
        },
        {
            "name": "grid_neutral_1",
            "a": 9,
            "b": 11,
            "expected": "Neutral",
        },
        {
            "name": "grid_neutral_2",
            "a": 10,
            "b": 12,
            "expected": "Neutral",
        },
        {
            "name": "grid_neutral_3",
            "a": 12,
            "b": 14,
            "expected": "Neutral",
        },
        {
            "name": "grid_neutral_4",
            "a": 15,
            "b": 15,
            "expected": "Neutral",
        },
        {
            "name": "grid_neutral_5",
            "a": 7,
            "b": 17,
            "expected": "Neutral",
        },
        {
            "name": "grid_neutral_6",
            "a": 20,
            "b": 16,
            "expected": "Neutral",
        },
        {
            "name": "grid_warm_1",
            "a": 8,
            "b": 18,
            "expected": "Warm",
        },
        {
            "name": "grid_warm_2",
            "a": 10,
            "b": 20,
            "expected": "Warm",
        },
        {
            "name": "grid_warm_3",
            "a": 15,
            "b": 25,
            "expected": "Warm",
        },
        {
            "name": "grid_warm_4",
            "a": 20,
            "b": 30,
            "expected": "Warm",
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
# Skin Depth Classification
# ============================================================

def test_skin_depth(failures):
    print_section("SKIN DEPTH CLASSIFICATION")

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
# Skin Depth Boundaries
# ============================================================

def test_skin_depth_boundaries(failures):
    print_section("SKIN DEPTH BOUNDARIES")

    cases = [
        {
            "name": "depth_34_99",
            "lightness": 34.99,
            "expected": "Deep",
        },
        {
            "name": "depth_35_exact",
            "lightness": 35,
            "expected": "Medium",
        },
        {
            "name": "depth_35_01",
            "lightness": 35.01,
            "expected": "Medium",
        },
        {
            "name": "depth_49_99",
            "lightness": 49.99,
            "expected": "Medium",
        },
        {
            "name": "depth_50_exact",
            "lightness": 50,
            "expected": "Medium-Light",
        },
        {
            "name": "depth_50_01",
            "lightness": 50.01,
            "expected": "Medium-Light",
        },
        {
            "name": "depth_67_99",
            "lightness": 67.99,
            "expected": "Medium-Light",
        },
        {
            "name": "depth_68_exact",
            "lightness": 68,
            "expected": "Light",
        },
        {
            "name": "depth_68_01",
            "lightness": 68.01,
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
# Skin Saturation Classification
# ============================================================

def test_skin_saturation(failures):
    print_section("SKIN SATURATION CLASSIFICATION")

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
# Skin Saturation Boundaries
# ============================================================

def test_skin_saturation_boundaries(failures):
    print_section("SKIN SATURATION BOUNDARIES")

    cases = [
        {
            "name": "saturation_17_99",
            "chroma": 17.99,
            "expected": "Muted",
        },
        {
            "name": "saturation_18_exact",
            "chroma": 18,
            "expected": "Moderate",
        },
        {
            "name": "saturation_18_01",
            "chroma": 18.01,
            "expected": "Moderate",
        },
        {
            "name": "saturation_29_99",
            "chroma": 29.99,
            "expected": "Moderate",
        },
        {
            "name": "saturation_30_exact",
            "chroma": 30,
            "expected": "Clear",
        },
        {
            "name": "saturation_30_01",
            "chroma": 30.01,
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
# Contrast Classification
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
    print_section("CONTRAST CLASSIFICATION")

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
# Basic Season Classification
# ============================================================

def test_season_matrix(failures):
    print_section("SEASON CLASSIFICATION")

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
# Full Warm Season Matrix
# ============================================================

def test_warm_season_matrix(failures):
    print_section("FULL WARM SEASON MATRIX")

    depths = [
        ("deep", "Deep"),
        ("medium", "Medium"),
        ("medium_light", "Medium-Light"),
        ("light", "Light"),
    ]

    cases = []

    # Clear -> Warm Spring for every depth.
    for depth_name, depth in depths:
        cases.append(
            {
                "name": f"warm_clear_{depth_name}",
                "temperature": "Warm",
                "saturation": "Clear",
                "depth": depth,
                "expected": "Warm Spring",
            }
        )

    # Moderate -> Warm Autumn for Deep/Medium,
    # Warm Spring for Medium-Light/Light.
    for depth_name, depth in depths:
        expected = (
            "Warm Autumn"
            if depth in ["Deep", "Medium"]
            else "Warm Spring"
        )

        cases.append(
            {
                "name": f"warm_moderate_{depth_name}",
                "temperature": "Warm",
                "saturation": "Moderate",
                "depth": depth,
                "expected": expected,
            }
        )

    # Muted -> Warm Autumn for every depth.
    for depth_name, depth in depths:
        cases.append(
            {
                "name": f"warm_muted_{depth_name}",
                "temperature": "Warm",
                "saturation": "Muted",
                "depth": depth,
                "expected": "Warm Autumn",
            }
        )

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
# Full Cool Season Matrix
# ============================================================

def test_cool_season_matrix(failures):
    print_section("FULL COOL SEASON MATRIX")

    depths = [
        ("deep", "Deep"),
        ("medium", "Medium"),
        ("medium_light", "Medium-Light"),
        ("light", "Light"),
    ]

    saturation_expectations = {
        "Clear": "Cool Winter",
        "Moderate": "Cool Summer",
        "Muted": "Cool Summer",
    }

    cases = []

    for saturation, expected in saturation_expectations.items():
        for depth_name, depth in depths:
            cases.append(
                {
                    "name": f"cool_{saturation.lower()}_{depth_name}",
                    "temperature": "Cool",
                    "saturation": saturation,
                    "depth": depth,
                    "expected": expected,
                }
            )

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
# Full Neutral Season Matrix
# ============================================================

def test_neutral_season_matrix(failures):
    print_section("FULL NEUTRAL SEASON MATRIX")

    depths = [
        ("deep", "Deep"),
        ("medium", "Medium"),
        ("medium_light", "Medium-Light"),
        ("light", "Light"),
    ]

    saturation_expectations = {
        "Clear": "Spring",
        "Moderate": "Neutral",
        "Muted": "Autumn",
    }

    cases = []

    for saturation, expected in saturation_expectations.items():
        for depth_name, depth in depths:
            cases.append(
                {
                    "name": f"neutral_{saturation.lower()}_{depth_name}",
                    "temperature": "Neutral",
                    "saturation": saturation,
                    "depth": depth,
                    "expected": expected,
                }
            )

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
# Explicit Season Boundaries
# ============================================================

def test_season_boundaries(failures):
    print_section("EXPLICIT SEASON BOUNDARIES")

    cases = [
        {
            "name": "warm_clear_boundary",
            "temperature": "Warm",
            "saturation": "Clear",
            "depth": "Medium-Light",
            "expected": "Warm Spring",
        },
        {
            # Medium depth + Moderate saturation intentionally
            # resolves to Warm Autumn in the current engine.
            "name": "warm_moderate_boundary",
            "temperature": "Warm",
            "saturation": "Moderate",
            "depth": "Medium",
            "expected": "Warm Autumn",
        },
        {
            "name": "warm_muted_boundary",
            "temperature": "Warm",
            "saturation": "Muted",
            "depth": "Medium-Light",
            "expected": "Warm Autumn",
        },
        {
            "name": "cool_clear_boundary",
            "temperature": "Cool",
            "saturation": "Clear",
            "depth": "Medium-Light",
            "expected": "Cool Winter",
        },
        {
            "name": "cool_moderate_boundary",
            "temperature": "Cool",
            "saturation": "Moderate",
            "depth": "Medium-Light",
            "expected": "Cool Summer",
        },
        {
            "name": "cool_muted_boundary",
            "temperature": "Cool",
            "saturation": "Muted",
            "depth": "Medium-Light",
            "expected": "Cool Summer",
        },
        {
            "name": "neutral_clear_boundary",
            "temperature": "Neutral",
            "saturation": "Clear",
            "depth": "Medium-Light",
            "expected": "Spring",
        },
        {
            "name": "neutral_moderate_boundary",
            "temperature": "Neutral",
            "saturation": "Moderate",
            "depth": "Medium-Light",
            "expected": "Neutral",
        },
        {
            "name": "neutral_muted_boundary",
            "temperature": "Neutral",
            "saturation": "Muted",
            "depth": "Medium-Light",
            "expected": "Autumn",
        },
        {
            "name": "warm_clear_deep_boundary",
            "temperature": "Warm",
            "saturation": "Clear",
            "depth": "Deep",
            "expected": "Warm Spring",
        },
        {
            "name": "warm_muted_deep_boundary",
            "temperature": "Warm",
            "saturation": "Muted",
            "depth": "Deep",
            "expected": "Warm Autumn",
        },
        {
            "name": "cool_clear_light_boundary",
            "temperature": "Cool",
            "saturation": "Clear",
            "depth": "Light",
            "expected": "Cool Winter",
        },
        {
            "name": "cool_muted_light_boundary",
            "temperature": "Cool",
            "saturation": "Muted",
            "depth": "Light",
            "expected": "Cool Summer",
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
    print_section("COMBINED COLOR PROFILES")

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
# Season Coverage
# ============================================================

def test_season_coverage(failures):
    print_section("SEASON COVERAGE")

    expected_labels = {
        "Autumn",
        "Cool Summer",
        "Cool Winter",
        "Neutral",
        "Spring",
        "Warm Autumn",
        "Warm Spring",
    }

    reachable_labels = set()

    temperatures = [
        "Warm",
        "Cool",
        "Neutral",
    ]

    saturations = [
        "Clear",
        "Moderate",
        "Muted",
    ]

    depths = [
        "Deep",
        "Medium",
        "Medium-Light",
        "Light",
    ]

    for temperature in temperatures:
        for saturation in saturations:
            for depth in depths:
                season = suggest_season(
                    temperature=temperature,
                    saturation=saturation,
                    skin_depth=depth,
                )

                reachable_labels.add(season)

    missing = expected_labels - reachable_labels

    if not missing:
        actual = ", ".join(sorted(reachable_labels))
        print(
            "[PASS] All expected season labels are reachable: "
            f"{actual}"
        )
        return True

    actual = ", ".join(sorted(reachable_labels))
    expected = ", ".join(sorted(expected_labels))

    failures.append(
        "Season coverage: "
        f"expected labels {expected}, "
        f"missing {', '.join(sorted(missing))}"
    )

    print(
        "[FAIL] Season coverage: "
        f"expected {expected}, got {actual}"
    )

    return False


# ============================================================
# Section Summary
# ============================================================

def print_section_summary(failures):
    print()
    print("=" * 60)
    print("SECTION SUMMARY")
    print("=" * 60)

    total_expected = 0
    total_failed = 0

    # Failure names are stored in a simple string format:
    # "<name>: expected <expected>, got <actual>"
    #
    # We classify failures by their unique test-name prefixes.
    failure_names = [
        failure.split(":", 1)[0]
        for failure in failures
    ]

    section_prefixes = {
        "Temperature classification": [
            "warm_light",
            "warm_medium",
            "warm_deep",
            "cool_light",
            "cool_medium",
            "cool_deep",
            "neutral_light",
            "neutral_medium",
            "neutral_deep",
        ],
        "Temperature boundaries": [
            "cool_edge_1",
            "cool_edge_2",
            "cool_edge_3",
            "cool_outside_a",
            "cool_outside_b",
            "warm_edge_1",
            "warm_edge_2",
            "warm_outside_a",
            "warm_outside_b",
            "cool_below_boundary",
            "neutral_between_cool_and_warm",
            "warm_above_boundary",
            "warm_a_below_threshold",
            "warm_b_below_threshold",
        ],
        "Temperature sweep": [
            "grid_cool_1",
            "grid_cool_2",
            "grid_cool_3",
            "grid_cool_4",
            "grid_neutral_1",
            "grid_neutral_2",
            "grid_neutral_3",
            "grid_neutral_4",
            "grid_neutral_5",
            "grid_neutral_6",
            "grid_warm_1",
            "grid_warm_2",
            "grid_warm_3",
            "grid_warm_4",
        ],
        "Skin depth classification": [
            "depth_34_9",
            "depth_35",
            "depth_49_9",
            "depth_50",
            "depth_67_9",
            "depth_68",
            "depth_80",
        ],
        "Skin depth boundaries": [
            "depth_34_99",
            "depth_35_exact",
            "depth_35_01",
            "depth_49_99",
            "depth_50_exact",
            "depth_50_01",
            "depth_67_99",
            "depth_68_exact",
            "depth_68_01",
        ],
        "Skin saturation classification": [
            "saturation_17_9",
            "saturation_18",
            "saturation_29_9",
            "saturation_30",
            "saturation_40",
        ],
        "Skin saturation boundaries": [
            "saturation_17_99",
            "saturation_18_exact",
            "saturation_18_01",
            "saturation_29_99",
            "saturation_30_exact",
            "saturation_30_01",
        ],
        "Contrast classification": [
            "contrast_low",
            "contrast_moderate",
            "contrast_high",
            "contrast_high_with_eyes",
        ],
        "Season matrix": [
            "warm_clear_light",
            "warm_clear_medium",
            "warm_muted_light",
            "warm_muted_medium",
            "warm_moderate_medium",
            "cool_clear_light",
            "cool_clear_deep",
            "cool_muted_light",
            "neutral_clear_light",
            "neutral_muted_light",
        ],
        "Warm season matrix": [
            "warm_clear_deep",
            "warm_clear_medium",
            "warm_clear_medium_light",
            "warm_clear_light",
            "warm_moderate_deep",
            "warm_moderate_medium",
            "warm_moderate_medium_light",
            "warm_moderate_light",
            "warm_muted_deep",
            "warm_muted_medium",
            "warm_muted_medium_light",
            "warm_muted_light",
        ],
        "Cool season matrix": [
            "cool_clear_deep",
            "cool_clear_medium",
            "cool_clear_medium_light",
            "cool_clear_light",
            "cool_moderate_deep",
            "cool_moderate_medium",
            "cool_moderate_medium_light",
            "cool_moderate_light",
            "cool_muted_deep",
            "cool_muted_medium",
            "cool_muted_medium_light",
            "cool_muted_light",
        ],
        "Neutral season matrix": [
            "neutral_clear_deep",
            "neutral_clear_medium",
            "neutral_clear_medium_light",
            "neutral_clear_light",
            "neutral_moderate_deep",
            "neutral_moderate_medium",
            "neutral_moderate_medium_light",
            "neutral_moderate_light",
            "neutral_muted_deep",
            "neutral_muted_medium",
            "neutral_muted_medium_light",
            "neutral_muted_light",
        ],
        "Season boundaries": [
            "warm_clear_boundary",
            "warm_moderate_boundary",
            "warm_muted_boundary",
            "cool_clear_boundary",
            "cool_moderate_boundary",
            "cool_muted_boundary",
            "neutral_clear_boundary",
            "neutral_moderate_boundary",
            "neutral_muted_boundary",
            "warm_clear_deep_boundary",
            "warm_muted_deep_boundary",
            "cool_clear_light_boundary",
            "cool_muted_light_boundary",
        ],
        "Combined profiles": [
            "warm_spring_profile",
            "warm_autumn_profile",
            "cool_winter_profile",
            "cool_summer_profile",
            "neutral_spring_profile",
            "neutral_autumn_profile",
        ],
    }

    for section, count in SECTION_COUNTS.items():
        if section == "Season coverage":
            section_failures = 1 if any(
                "Season coverage:" in failure
                for failure in failures
            ) else 0
        else:
            prefixes = section_prefixes[section]
            section_failures = sum(
                1
                for name in failure_names
                if name in prefixes
            )

        passed = count - section_failures

        print(
            f"{section:<38}"
            f"{passed}/{count}"
            f"{' PASS' if section_failures == 0 else ' FAIL'}"
        )

        total_expected += count
        total_failed += section_failures

    return total_expected, total_failed


# ============================================================
# Main
# ============================================================

def main():
    """Run the complete Tinayu color-profile evaluation."""

    failures = []

    print()
    print("=" * 60)
    print("TINAYU COLOR PROFILE EVALUATION v2")
    print("=" * 60)
    print()
    print("Evaluation mode: classifier validation only")
    print("No Tinayu engine values will be modified.")

    # Temperature
    test_temperature(failures)
    test_temperature_boundaries(failures)
    test_temperature_grid(failures)

    # Skin attributes
    test_skin_depth(failures)
    test_skin_depth_boundaries(failures)
    test_skin_saturation(failures)
    test_skin_saturation_boundaries(failures)

    # Contrast
    test_contrast(failures)

    # Season classification
    test_season_matrix(failures)
    test_warm_season_matrix(failures)
    test_cool_season_matrix(failures)
    test_neutral_season_matrix(failures)
    test_season_boundaries(failures)

    # Combined profiles
    test_combined_profiles(failures)

    # Reachability
    test_season_coverage(failures)

    # Summary
    total_checks, total_failures = print_section_summary(
        failures
    )

    passed_checks = total_checks - total_failures

    print()
    print("=" * 60)
    print("FINAL RESULT")
    print("=" * 60)
    print()

    print(
        f"Checks passed: {passed_checks}/{total_checks}"
    )

    print(
        f"Checks failed: {total_failures}/{total_checks}"
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
    raise SystemExit(main())
