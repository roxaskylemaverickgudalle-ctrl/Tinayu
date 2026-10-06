
import json
import sys
from pathlib import Path

import numpy as np

# Allow importing tinayu_engine.py from this folder.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tinayu_engine import (
    classify_temperature_stable,
    temperature_score,
)


SAMPLES = {
    "Bright indoor": [67.84, 16.0, 11.0],
    "Different camera": [65.10, 13.0, 14.0],
    "Outdoor": [60.78, 13.0, 18.0],
    "Slightly different angle": [67.45, 15.0, 20.0],
    "Without makeup": [70.59, 9.0, 16.0],
    "Normal indoor": [58.04, 10.0, 18.0],
    "Makeover": [64.71, 26.0, 34.0],
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


def classify_strength(strength):
    if strength <= 0.42:
        return "Strong Cool"

    if strength < 0.45:
        return "Borderline Cool"

    if strength < 0.55:
        return "Neutral"

    if strength < 0.58:
        return "Borderline Warm"

    return "Strong Warm"


def run_sample(name, lab):
    L, base_a, base_b = lab

    baseline_strength = temperature_score(base_a, base_b)
    baseline_temperature = classify_temperature_stable(
        base_a,
        base_b,
    )

    results = []

    for delta_a, delta_b in PERTURBATIONS:
        test_a = base_a + delta_a
        test_b = base_b + delta_b

        strength = temperature_score(test_a, test_b)
        temperature = classify_temperature_stable(
            test_a,
            test_b,
        )

        results.append(
            {
                "delta_a": delta_a,
                "delta_b": delta_b,
                "a": float(test_a),
                "b": float(test_b),
                "strength": float(strength),
                "temperature": temperature,
                "band": classify_strength(strength),
                "classification_changed": (
                    temperature != baseline_temperature
                ),
            }
        )

    strengths = [r["strength"] for r in results]

    changed = [
        r for r in results
        if r["classification_changed"]
    ]

    # Isolate a-only and b-only perturbations.
    a_only = [
        r for r in results
        if r["delta_a"] != 0 and r["delta_b"] == 0
    ]

    b_only = [
        r for r in results
        if r["delta_a"] == 0 and r["delta_b"] != 0
    ]

    max_a_effect = max(
        abs(r["strength"] - baseline_strength)
        for r in a_only
    )

    max_b_effect = max(
        abs(r["strength"] - baseline_strength)
        for r in b_only
    )

    return {
        "sample": name,
        "baseline": {
            "L": float(L),
            "a": float(base_a),
            "b": float(base_b),
            "strength": float(baseline_strength),
            "temperature": baseline_temperature,
            "band": classify_strength(baseline_strength),
        },
        "perturbations": results,
        "summary": {
            "total_tests": len(results),
            "classification_flips": len(changed),
            "flip_rate": (
                len(changed) / len(results)
            ),
            "min_strength": float(min(strengths)),
            "max_strength": float(max(strengths)),
            "strength_range": float(
                max(strengths) - min(strengths)
            ),
            "max_a_only_effect": float(max_a_effect),
            "max_b_only_effect": float(max_b_effect),
        },
    }


def main():
    print("=" * 100)
    print("TINAYU TEMPERATURE PERTURBATION STRESS TEST")
    print("=" * 100)

    all_results = []

    for name, lab in SAMPLES.items():
        result = run_sample(name, lab)
        all_results.append(result)

        baseline = result["baseline"]
        summary = result["summary"]

        print(f"\n{name}")
        print("-" * 100)

        print(
            f"Baseline: "
            f"a={baseline['a']:.2f}, "
            f"b={baseline['b']:.2f}, "
            f"strength={baseline['strength']:.4f}, "
            f"{baseline['temperature']} "
            f"({baseline['band']})"
        )

        print(
            f"Strength range under perturbations: "
            f"{summary['min_strength']:.4f} "
            f"→ "
            f"{summary['max_strength']:.4f}"
        )

        print(
            f"Total tests: {summary['total_tests']}"
        )

        print(
            f"Classification flips: "
            f"{summary['classification_flips']} "
            f"({summary['flip_rate'] * 100:.1f}%)"
        )

        print(
            f"Max effect from ±a perturbation: "
            f"{summary['max_a_only_effect']:.4f}"
        )

        print(
            f"Max effect from ±b perturbation: "
            f"{summary['max_b_only_effect']:.4f}"
        )

        changed = [
            r
            for r in result["perturbations"]
            if r["classification_changed"]
        ]

        if changed:
            print("\nClassification flips:")
            for r in changed:
                print(
                    f"  Δa={r['delta_a']:+.1f}, "
                    f"Δb={r['delta_b']:+.1f} "
                    f"→ "
                    f"{r['strength']:.4f} "
                    f"{r['temperature']} "
                    f"({r['band']})"
                )
        else:
            print("\nNo classification flips.")

    # Global summary.
    total_tests = sum(
        r["summary"]["total_tests"]
        for r in all_results
    )

    total_flips = sum(
        r["summary"]["classification_flips"]
        for r in all_results
    )

    print("\n" + "=" * 100)
    print("GLOBAL SUMMARY")
    print("=" * 100)

    print(f"Samples tested:       {len(all_results)}")
    print(f"Perturbations:        {len(PERTURBATIONS)} per sample")
    print(f"Total tests:          {total_tests}")
    print(f"Total flips:          {total_flips}")
    print(
        f"Overall flip rate:    "
        f"{(total_flips / total_tests) * 100:.2f}%"
    )

    print("\nInterpretation guide:")
    print(
        "  0% flips      = very stable under these perturbations"
    )
    print(
        "  Low flips     = mostly stable; boundary cases exist"
    )
    print(
        "  High flips    = classifier may be too sensitive"
    )

    output = {
        "samples": all_results,
        "global_summary": {
            "sample_count": len(all_results),
            "perturbations_per_sample": len(PERTURBATIONS),
            "total_tests": total_tests,
            "total_flips": total_flips,
            "overall_flip_rate": (
                total_flips / total_tests
            ),
        },
    }

    output_path = Path(
        "temperature_perturbation_report.json"
    )

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(
            output,
            f,
            indent=2,
        )

    print(
        f"\nSaved report: {output_path}"
    )


if __name__ == "__main__":
    main()
