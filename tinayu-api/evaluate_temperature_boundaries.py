
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tinayu_engine import temperature_score


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


def current_classifier(score):
    if score >= 0.55:
        return "Warm"
    if score <= 0.45:
        return "Cool"
    return "Neutral"


def existing_band(score):
    if score <= 0.42:
        return "Strong Cool"
    if score < 0.45:
        return "Borderline Cool"
    if score < 0.55:
        return "Neutral"
    if score < 0.58:
        return "Borderline Warm"
    return "Strong Warm"


def candidate_a(score):
    """
    Preserve the current temperature strength bands,
    but explicitly expose the borderline states.
    """
    if score <= 0.42:
        return "Strong Cool"
    if score < 0.45:
        return "Borderline Cool"
    if score < 0.55:
        return "Neutral"
    if score < 0.58:
        return "Borderline Warm"
    return "Strong Warm"


def candidate_b(score):
    """
    Wider uncertainty zone.

    The middle region is explicitly considered
    borderline rather than forcing it into Warm/Neutral.
    """
    if score < 0.53:
        return "Cool/Neutral"
    if score <= 0.60:
        return "Borderline"
    return "Warm"


def candidate_c(score):
    """
    Conservative warm interpretation.

    Scores below 0.58 are not treated as confidently warm.
    """
    if score < 0.45:
        return "Cool"
    if score < 0.58:
        return "Neutral/Borderline"
    return "Warm"


def collect_scores():
    rows = []

    for sample_name, lab in SAMPLES.items():
        _, base_a, base_b = lab

        for delta_a, delta_b in PERTURBATIONS:
            a = base_a + delta_a
            b = base_b + delta_b

            score = temperature_score(a, b)

            rows.append(
                {
                    "sample": sample_name,
                    "base_a": float(base_a),
                    "base_b": float(base_b),
                    "delta_a": float(delta_a),
                    "delta_b": float(delta_b),
                    "a": float(a),
                    "b": float(b),
                    "score": float(score),
                }
            )

    return rows


def summarize_scheme(rows, classifier):
    labels = [
        classifier(row["score"])
        for row in rows
    ]

    counts = {}

    for label in labels:
        counts[label] = counts.get(label, 0) + 1

    return {
        "counts": counts,
        "total": len(labels),
    }


def sample_summary(rows, classifier):
    output = {}

    for sample_name in SAMPLES:
        sample_rows = [
            row
            for row in rows
            if row["sample"] == sample_name
        ]

        labels = [
            classifier(row["score"])
            for row in sample_rows
        ]

        counts = {}

        for label in labels:
            counts[label] = counts.get(label, 0) + 1

        baseline_row = next(
            row
            for row in sample_rows
            if row["delta_a"] == 0
            and row["delta_b"] == 0
        )

        baseline_label = classifier(
            baseline_row["score"]
        )

        flips = sum(
            label != baseline_label
            for label in labels
        )

        output[sample_name] = {
            "baseline_score": baseline_row["score"],
            "baseline_label": baseline_label,
            "label_counts": counts,
            "classification_changes": flips,
            "change_rate": flips / len(labels),
        }

    return output


def main():
    print("=" * 100)
    print("TINAYU TEMPERATURE BOUNDARY EVALUATION")
    print("=" * 100)

    rows = collect_scores()

    schemes = {
        "CURRENT": current_classifier,
        "CANDIDATE_A_EXISTING_BANDS": candidate_a,
        "CANDIDATE_B_WIDER_BORDERLINE": candidate_b,
        "CANDIDATE_C_CONSERVATIVE_WARM": candidate_c,
    }

    all_results = {}

    for name, classifier in schemes.items():
        summary = summarize_scheme(
            rows,
            classifier,
        )

        per_sample = sample_summary(
            rows,
            classifier,
        )

        all_results[name] = {
            "summary": summary,
            "per_sample": per_sample,
        }

        print("\n" + "=" * 100)
        print(name)
        print("=" * 100)

        print("\nGlobal label distribution:")

        for label, count in sorted(
            summary["counts"].items()
        ):
            percentage = (
                count / summary["total"]
            ) * 100

            print(
                f"  {label:<22} "
                f"{count:>3} "
                f"({percentage:>5.1f}%)"
            )

        print("\nPer-sample baseline + perturbation behavior:")

        for sample_name, data in per_sample.items():
            print(
                f"\n  {sample_name}"
            )

            print(
                f"    baseline: "
                f"{data['baseline_score']:.4f} "
                f"→ "
                f"{data['baseline_label']}"
            )

            print(
                f"    changes:  "
                f"{data['classification_changes']}/"
                f"{len(PERTURBATIONS)} "
                f"({data['change_rate'] * 100:.1f}%)"
            )

            counts_text = ", ".join(
                f"{label}={count}"
                for label, count
                in sorted(
                    data["label_counts"].items()
                )
            )

            print(
                f"    labels:   {counts_text}"
            )

    # Direct comparison of baseline real samples.
    print("\n" + "=" * 100)
    print("REAL SAMPLE BASELINES")
    print("=" * 100)

    for sample_name, lab in SAMPLES.items():
        _, a, b = lab

        score = temperature_score(a, b)

        print(
            f"{sample_name:<28} "
            f"score={score:.4f} | "
            f"current={current_classifier(score):<8} | "
            f"bands={existing_band(score):<16} | "
            f"wide={candidate_b(score):<16} | "
            f"conservative={candidate_c(score)}"
        )

    # Focus specifically on the uncertain score region.
    print("\n" + "=" * 100)
    print("UNCERTAINTY REGION ANALYSIS")
    print("=" * 100)

    for lower, upper in [
        (0.53, 0.58),
        (0.53, 0.60),
        (0.55, 0.58),
    ]:
        selected = [
            row
            for row in rows
            if lower <= row["score"] <= upper
        ]

        print(
            f"Scores {lower:.2f}–{upper:.2f}: "
            f"{len(selected)} / {len(rows)} "
            f"({len(selected) / len(rows) * 100:.1f}%)"
        )

    # Save report.
    output = {
        "sample_count": len(SAMPLES),
        "perturbations_per_sample": len(PERTURBATIONS),
        "total_measurements": len(rows),
        "schemes": all_results,
        "raw_measurements": rows,
    }

    output_path = Path(
        "temperature_boundary_report.json"
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as f:
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
