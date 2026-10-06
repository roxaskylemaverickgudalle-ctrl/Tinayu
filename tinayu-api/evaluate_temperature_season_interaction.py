
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tinayu_engine import (
    suggest_season,
    temperature_score,
)


SAMPLES = {
    "Bright indoor": {
        "lab": [67.84, 16.0, 11.0],
        "saturation": "Moderate",
        "depth": "Light",
    },
    "Different camera": {
        "lab": [65.10, 13.0, 14.0],
        "saturation": "Moderate",
        "depth": "Light",
    },
    "Outdoor": {
        "lab": [60.78, 13.0, 18.0],
        "saturation": "Moderate",
        "depth": "Light",
    },
    "Slightly different angle": {
        "lab": [67.45, 15.0, 20.0],
        "saturation": "Moderate",
        "depth": "Light",
    },
    "Without makeup": {
        "lab": [70.59, 9.0, 16.0],
        "saturation": "Moderate",
        "depth": "Light",
    },
    "Normal indoor": {
        "lab": [58.04, 10.0, 18.0],
        "saturation": "Moderate",
        "depth": "Light",
    },
    "Makeover": {
        "lab": [64.71, 26.0, 34.0],
        "saturation": "Clear",
        "depth": "Light",
    },
}


PERTURBATIONS = [
    (0.0, 0.0),
    (-0.5, 0.0),
    (+0.5, 0.0),
    (-1.0, 0.0),
    (+1.0, 0.0),
    (-1.5, 0.0),
    (+1.5, 0.0),
    (-2.0, 0.0),
    (+2.0, 0.0),
    (0.0, -0.5),
    (0.0, +0.5),
    (0.0, -1.0),
    (0.0, +1.0),
    (0.0, -1.5),
    (0.0, +1.5),
    (0.0, -2.0),
    (0.0, +2.0),
    (-1.0, -1.0),
    (-1.0, +1.0),
    (+1.0, -1.0),
    (+1.0, +1.0),
    (-2.0, -2.0),
    (-2.0, +2.0),
    (+2.0, -2.0),
    (+2.0, +2.0),
]


def current_temperature(score):
    if score >= 0.55:
        return "Warm"

    if score <= 0.45:
        return "Cool"

    return "Neutral"


def temperature_band(score):
    if score <= 0.42:
        return "Strong Cool"

    if score < 0.45:
        return "Borderline Cool"

    if score < 0.55:
        return "Neutral"

    if score < 0.58:
        return "Borderline Warm"

    return "Strong Warm"


def conservative_temperature(score):
    """
    Experimental approach:
    Treat the entire 0.53–0.60 region as uncertain Neutral
    for purposes of season selection.

    Outside that region, preserve the current classifier.
    """
    if score < 0.53:
        return "Cool"

    if score <= 0.60:
        return "Neutral"

    return "Warm"


def warm_preserving_temperature(score):
    """
    Experimental approach:
    Preserve Warm classification at >= 0.55,
    but expose the borderline state separately.
    """
    if score >= 0.55:
        return "Warm"

    if score <= 0.45:
        return "Cool"

    return "Neutral"


def season_from_method(
    temperature,
    saturation,
    depth,
    strength,
):
    return suggest_season(
        temperature,
        saturation,
        depth,
        temperature_strength=strength,
    )


def main():
    print("=" * 105)
    print("TINAYU TEMPERATURE → SEASON INTERACTION DIAGNOSTIC")
    print("=" * 105)

    all_results = []

    for sample_name, sample in SAMPLES.items():
        L, base_a, base_b = sample["lab"]

        sample_results = []

        for delta_a, delta_b in PERTURBATIONS:
            a = base_a + delta_a
            b = base_b + delta_b

            score = temperature_score(a, b)

            current_temp = current_temperature(score)
            conservative_temp = conservative_temperature(score)
            warm_preserving_temp = warm_preserving_temperature(score)

            current_season = season_from_method(
                current_temp,
                sample["saturation"],
                sample["depth"],
                score,
            )

            conservative_season = season_from_method(
                conservative_temp,
                sample["saturation"],
                sample["depth"],
                score,
            )

            warm_preserving_season = season_from_method(
                warm_preserving_temp,
                sample["saturation"],
                sample["depth"],
                score,
            )

            sample_results.append(
                {
                    "delta_a": delta_a,
                    "delta_b": delta_b,
                    "a": float(a),
                    "b": float(b),
                    "temperature_score": float(score),
                    "temperature_band": temperature_band(score),
                    "current_temperature": current_temp,
                    "conservative_temperature": conservative_temp,
                    "warm_preserving_temperature": (
                        warm_preserving_temp
                    ),
                    "current_season": current_season,
                    "conservative_season": (
                        conservative_season
                    ),
                    "warm_preserving_season": (
                        warm_preserving_season
                    ),
                }
            )

        all_results.append(
            {
                "sample": sample_name,
                "lab": sample["lab"],
                "saturation": sample["saturation"],
                "depth": sample["depth"],
                "perturbations": sample_results,
            }
        )

    # ------------------------------------------------------------------
    # Print per-sample baseline behavior
    # ------------------------------------------------------------------

    print("\n" + "=" * 105)
    print("REAL SAMPLE BASELINES")
    print("=" * 105)

    for result in all_results:
        baseline = next(
            row
            for row in result["perturbations"]
            if row["delta_a"] == 0
            and row["delta_b"] == 0
        )

        print(
            f"\n{result['sample']}"
        )

        print(
            f"  score:               "
            f"{baseline['temperature_score']:.4f}"
        )

        print(
            f"  temperature band:    "
            f"{baseline['temperature_band']}"
        )

        print(
            f"  current:             "
            f"{baseline['current_temperature']}"
            f" → "
            f"{baseline['current_season']}"
        )

        print(
            f"  conservative:        "
            f"{baseline['conservative_temperature']}"
            f" → "
            f"{baseline['conservative_season']}"
        )

        print(
            f"  warm-preserving:     "
            f"{baseline['warm_preserving_temperature']}"
            f" → "
            f"{baseline['warm_preserving_season']}"
        )

    # ------------------------------------------------------------------
    # Compare season stability
    # ------------------------------------------------------------------

    print("\n" + "=" * 105)
    print("SEASON STABILITY UNDER PERTURBATIONS")
    print("=" * 105)

    global_stats = {
        "current": {
            "season_changes": 0,
            "total": 0,
        },
        "conservative": {
            "season_changes": 0,
            "total": 0,
        },
        "warm_preserving": {
            "season_changes": 0,
            "total": 0,
        },
    }

    for result in all_results:
        baseline = next(
            row
            for row in result["perturbations"]
            if row["delta_a"] == 0
            and row["delta_b"] == 0
        )

        methods = {
            "current": "current_season",
            "conservative": "conservative_season",
            "warm_preserving": "warm_preserving_season",
        }

        print(
            f"\n{result['sample']}"
        )

        for method_name, field in methods.items():
            baseline_season = baseline[field]

            changes = sum(
                row[field] != baseline_season
                for row in result["perturbations"]
            )

            total = len(result["perturbations"])

            global_stats[method_name]["season_changes"] += (
                changes
            )
            global_stats[method_name]["total"] += total

            print(
                f"  {method_name:<18} "
                f"{changes}/{total} changes "
                f"({changes / total * 100:.1f}%)"
            )

    # ------------------------------------------------------------------
    # Global comparison
    # ------------------------------------------------------------------

    print("\n" + "=" * 105)
    print("GLOBAL SEASON-STABILITY COMPARISON")
    print("=" * 105)

    for method_name, stats in global_stats.items():
        change_rate = (
            stats["season_changes"]
            / stats["total"]
        )

        print(
            f"{method_name:<20} "
            f"{stats['season_changes']:>3}/"
            f"{stats['total']} season changes "
            f"({change_rate * 100:.2f}%)"
        )

    # ------------------------------------------------------------------
    # Find cases where temperature changes but season does not.
    # ------------------------------------------------------------------

    print("\n" + "=" * 105)
    print("BORDERLINE TEMPERATURE → SEASON BEHAVIOR")
    print("=" * 105)

    for result in all_results:
        interesting = []

        for row in result["perturbations"]:
            if (
                row["temperature_band"]
                in {
                    "Borderline Warm",
                    "Neutral",
                    "Borderline Cool",
                }
            ):
                interesting.append(row)

        if not interesting:
            continue

        print(
            f"\n{result['sample']}"
        )

        season_counts = {}

        for row in interesting:
            season = row["current_season"]
            season_counts[season] = (
                season_counts.get(season, 0) + 1
            )

        print(
            "  Current seasons inside borderline region:"
        )

        for season, count in sorted(
            season_counts.items()
        ):
            print(
                f"    {season:<20} {count}"
            )

    # ------------------------------------------------------------------
    # Save JSON report
    # ------------------------------------------------------------------

    report = {
        "samples": all_results,
        "global_stats": global_stats,
    }

    output_path = Path(
        "temperature_season_interaction_report.json"
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            report,
            f,
            indent=2,
        )

    print(
        f"\nSaved report: {output_path}"
    )


if __name__ == "__main__":
    main()
