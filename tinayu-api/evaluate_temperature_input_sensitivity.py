
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tinayu_engine import temperature_score


SAMPLES = {
    "Bright indoor": {
        "a": 16.0,
        "b": 11.0,
    },
    "Different camera": {
        "a": 13.0,
        "b": 14.0,
    },
    "Outdoor": {
        "a": 13.0,
        "b": 18.0,
    },
    "Slightly different angle": {
        "a": 15.0,
        "b": 20.0,
    },
    "Without makeup": {
        "a": 9.0,
        "b": 16.0,
    },
    "Normal indoor": {
        "a": 10.0,
        "b": 18.0,
    },
    "Makeover": {
        "a": 26.0,
        "b": 34.0,
    },
}


def classify(score):
    if score >= 0.55:
        return "Warm"

    if score <= 0.45:
        return "Cool"

    return "Neutral"


def print_header(title):
    print("\n" + "=" * 95)
    print(title)
    print("=" * 95)


def main():
    print_header("TINAYU TEMPERATURE INPUT SENSITIVITY")

    print(
        "\nThis test measures how much the temperature score moves "
        "when the extracted Lab a/b signal changes slightly."
    )

    print(
        "It does NOT modify the production engine.\n"
    )

    # ---------------------------------------------------------------
    # Baselines
    # ---------------------------------------------------------------

    print_header("BASELINE SIGNAL")

    for name, sample in SAMPLES.items():
        a = sample["a"]
        b = sample["b"]

        score = temperature_score(a, b)

        print(
            f"{name:<28} "
            f"a={a:>5.1f} "
            f"b={b:>5.1f} "
            f"score={score:.4f} "
            f"{classify(score)}"
        )

    # ---------------------------------------------------------------
    # One-channel sensitivity
    # ---------------------------------------------------------------

    print_header("SENSITIVITY TO LAB a")

    print(
        "\nEach sample gets a/b changes of ±0.5, ±1, ±1.5 and ±2 "
        "while b remains unchanged.\n"
    )

    for name, sample in SAMPLES.items():
        a = sample["a"]
        b = sample["b"]

        baseline = temperature_score(a, b)

        scores = []

        for delta in [-2, -1.5, -1, -0.5, 0.5, 1, 1.5, 2]:
            scores.append(
                temperature_score(a + delta, b)
            )

        minimum = min(scores)
        maximum = max(scores)

        print(
            f"{name:<28} "
            f"baseline={baseline:.4f} "
            f"range={minimum:.4f}-{maximum:.4f} "
            f"span={maximum - minimum:.4f}"
        )

    # ---------------------------------------------------------------
    # One-channel b sensitivity
    # ---------------------------------------------------------------

    print_header("SENSITIVITY TO LAB b")

    print(
        "\nEach sample gets a/b changes of ±0.5, ±1, ±1.5 and ±2 "
        "while a remains unchanged.\n"
    )

    for name, sample in SAMPLES.items():
        a = sample["a"]
        b = sample["b"]

        baseline = temperature_score(a, b)

        scores = []

        for delta in [-2, -1.5, -1, -0.5, 0.5, 1, 1.5, 2]:
            scores.append(
                temperature_score(a, b + delta)
            )

        minimum = min(scores)
        maximum = max(scores)

        print(
            f"{name:<28} "
            f"baseline={baseline:.4f} "
            f"range={minimum:.4f}-{maximum:.4f} "
            f"span={maximum - minimum:.4f}"
        )

    # ---------------------------------------------------------------
    # Directional movement
    # ---------------------------------------------------------------

    print_header("DIRECTIONAL RESPONSE")

    print(
        "\nFor each sample, +1 means increasing only one Lab channel "
        "by 1 unit.\n"
    )

    for name, sample in SAMPLES.items():
        a = sample["a"]
        b = sample["b"]

        baseline = temperature_score(a, b)

        plus_a = temperature_score(a + 1, b)
        minus_a = temperature_score(a - 1, b)

        plus_b = temperature_score(a, b + 1)
        minus_b = temperature_score(a, b - 1)

        print(f"\n{name}")

        print(
            f"  baseline: {baseline:.4f}"
        )

        print(
            f"  a -1:     {minus_a:.4f} "
            f"({minus_a - baseline:+.4f})"
        )

        print(
            f"  a +1:     {plus_a:.4f} "
            f"({plus_a - baseline:+.4f})"
        )

        print(
            f"  b -1:     {minus_b:.4f} "
            f"({minus_b - baseline:+.4f})"
        )

        print(
            f"  b +1:     {plus_b:.4f} "
            f"({plus_b - baseline:+.4f})"
        )

    # ---------------------------------------------------------------
    # Boundary distance
    # ---------------------------------------------------------------

    print_header("DISTANCE FROM TEMPERATURE BOUNDARIES")

    for name, sample in SAMPLES.items():
        score = temperature_score(
            sample["a"],
            sample["b"],
        )

        distance_055 = score - 0.55
        distance_058 = score - 0.58

        print(
            f"{name:<28} "
            f"score={score:.4f} "
            f"to .55={distance_055:+.4f} "
            f"to .58={distance_058:+.4f}"
        )

    # ---------------------------------------------------------------
    # Combined perturbation grid
    # ---------------------------------------------------------------

    print_header("COMBINED a/b PERTURBATION")

    print(
        "\nEach sample is tested with every combination of "
        "a/b changes from -2 to +2 in 0.5 increments.\n"
    )

    deltas = np.arange(-2.0, 2.01, 0.5)

    total_flips = 0
    total_cases = 0

    for name, sample in SAMPLES.items():
        baseline_score = temperature_score(
            sample["a"],
            sample["b"],
        )

        baseline_class = classify(baseline_score)

        scores = []

        for delta_a in deltas:
            for delta_b in deltas:
                score = temperature_score(
                    sample["a"] + delta_a,
                    sample["b"] + delta_b,
                )

                scores.append(score)

        minimum = min(scores)
        maximum = max(scores)

        flips = sum(
            classify(score) != baseline_class
            for score in scores
        )

        total_flips += flips
        total_cases += len(scores)

        print(
            f"{name:<28} "
            f"baseline={baseline_score:.4f} "
            f"range={minimum:.4f}-{maximum:.4f} "
            f"flips={flips}/{len(scores)} "
            f"({flips / len(scores) * 100:.1f}%)"
        )

    print(
        f"\nGLOBAL: "
        f"{total_flips}/{total_cases} "
        f"classification changes "
        f"({total_flips / total_cases * 100:.2f}%)"
    )

    print_header("INTERPRETATION GUIDE")

    print(
        "If borderline samples show large score movement from tiny "
        "a/b changes while strong-Warm samples remain stable, the "
        "problem is likely signal sensitivity rather than season "
        "thresholds."
    )

    print(
        "\nIf a and b respond smoothly and predictably, we should "
        "leave the temperature formula alone and investigate the "
        "actual skin-pixel extraction / lighting normalization."
    )

    print(
        "\nNo production code was changed by this evaluator."
    )


if __name__ == "__main__":
    main()
