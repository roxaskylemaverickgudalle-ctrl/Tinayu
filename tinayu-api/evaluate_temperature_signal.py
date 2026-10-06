
import json
import sys
from pathlib import Path

import numpy as np

# Allow importing tinayu_engine.py from the same folder.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tinayu_engine import (
    classify_temperature_stable,
    suggest_season,
    temperature_score,
)


SAMPLES = {
    "Bright indoor": {
        "lab": [67.84, 16.0, 11.0],
    },
    "Different camera": {
        "lab": [65.10, 13.0, 14.0],
    },
    "Outdoor": {
        "lab": [60.78, 13.0, 18.0],
    },
    "Slightly different angle": {
        "lab": [67.45, 15.0, 20.0],
    },
    "Without makeup": {
        "lab": [70.59, 9.0, 16.0],
    },
    "Normal indoor": {
        "lab": [58.04, 10.0, 18.0],
    },
    "Makeover": {
        "lab": [64.71, 26.0, 34.0],
    },
}


def calculate_signal_parts(lab_a, lab_b):
    a = float(lab_a)
    b = float(lab_b)

    a_component = ((a - 8.0) / 8.0) * 0.40
    b_component = ((b - 14.0) / 8.0) * 0.60

    signal = a_component + b_component

    sigmoid = 1.0 / (1.0 + np.exp(-signal))

    return {
        "a_component": float(a_component),
        "b_component": float(b_component),
        "combined_signal": float(signal),
        "sigmoid_strength": float(sigmoid),
    }


def classify_band(strength):
    if strength <= 0.42:
        return "Strong Cool"
    if strength < 0.45:
        return "Borderline Cool"
    if strength < 0.55:
        return "Neutral"
    if strength < 0.58:
        return "Borderline Warm"
    return "Strong Warm"


def main():
    rows = []

    print("=" * 90)
    print("TINAYU TEMPERATURE SIGNAL DIAGNOSTIC")
    print("=" * 90)

    for name, sample in SAMPLES.items():
        L, a, b = sample["lab"]

        parts = calculate_signal_parts(a, b)

        engine_strength = temperature_score(a, b)
        temperature = classify_temperature_stable(a, b)
        band = classify_band(engine_strength)

        distance_055 = engine_strength - 0.55
        distance_058 = engine_strength - 0.58
        distance_045 = engine_strength - 0.45

        # Reconstruct the season using the same saturation/depth
        # information only where it is obvious from the diagnostic.
        # This is NOT replacing the production classifier.
        if name == "Makeover":
            saturation = "Clear"
        elif name == "Slightly different angle":
            saturation = "Moderate"
        elif name == "Outdoor":
            saturation = "Moderate"
        elif name == "Normal indoor":
            saturation = "Moderate"
        elif name == "Different camera":
            saturation = "Moderate"
        elif name == "Without makeup":
            saturation = "Moderate"
        else:
            saturation = "Moderate"

        season = suggest_season(
            temperature,
            saturation,
            "Light",
            temperature_strength=engine_strength,
        )

        row = {
            "sample": name,
            "L": float(L),
            "a": float(a),
            "b": float(b),
            "a_component": parts["a_component"],
            "b_component": parts["b_component"],
            "combined_signal": parts["combined_signal"],
            "temperature_strength": float(engine_strength),
            "temperature": temperature,
            "temperature_band": band,
            "distance_to_0.45": float(distance_045),
            "distance_to_0.55": float(distance_055),
            "distance_to_0.58": float(distance_058),
            "season": season,
        }

        rows.append(row)

        print(f"\n{name}")
        print("-" * 90)
        print(f"LAB:                 L={L:.2f}, a={a:.2f}, b={b:.2f}")
        print(f"a contribution:      {parts['a_component']:+.4f}")
        print(f"b contribution:      {parts['b_component']:+.4f}")
        print(f"combined signal:     {parts['combined_signal']:+.4f}")
        print(f"temperature score:   {engine_strength:.4f}")
        print(f"temperature:         {temperature}")
        print(f"temperature band:    {band}")
        print(f"distance to 0.45:    {distance_045:+.4f}")
        print(f"distance to 0.55:    {distance_055:+.4f}")
        print(f"distance to 0.58:    {distance_058:+.4f}")
        print(f"season:              {season}")

    strengths = [r["temperature_strength"] for r in rows]
    a_parts = [r["a_component"] for r in rows]
    b_parts = [r["b_component"] for r in rows]

    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)

    print(f"Mean temperature strength:   {np.mean(strengths):.4f}")
    print(f"Median temperature strength: {np.median(strengths):.4f}")
    print(f"Min temperature strength:    {np.min(strengths):.4f}")
    print(f"Max temperature strength:    {np.max(strengths):.4f}")

    print(f"\nMean a contribution:          {np.mean(a_parts):+.4f}")
    print(f"Mean b contribution:          {np.mean(b_parts):+.4f}")

    print("\nTemperature strength values:")
    for row in rows:
        print(
            f"  {row['sample']:<28} "
            f"{row['temperature_strength']:.4f} "
            f"{row['temperature']:<7} "
            f"{row['temperature_band']}"
        )

    print("\nBoundary cases:")
    for row in rows:
        strength = row["temperature_strength"]

        if 0.52 <= strength <= 0.60:
            print(
                f"  {row['sample']:<28} "
                f"{strength:.4f} "
                f"(near decision boundary)"
            )

    output_path = Path("temperature_signal_report.json")

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "samples": rows,
                "summary": {
                    "mean_strength": float(np.mean(strengths)),
                    "median_strength": float(np.median(strengths)),
                    "min_strength": float(np.min(strengths)),
                    "max_strength": float(np.max(strengths)),
                    "mean_a_component": float(np.mean(a_parts)),
                    "mean_b_component": float(np.mean(b_parts)),
                },
            },
            f,
            indent=2,
        )

    print(f"\nSaved report: {output_path}")


if __name__ == "__main__":
    main()
