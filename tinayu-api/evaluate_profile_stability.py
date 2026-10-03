import json
import statistics
import sys
from pathlib import Path

from tinayu_engine import analyze_image


SUPPORTED = {".jpg", ".jpeg", ".png", ".webp"}


def load_result(path: Path):
    with path.open("rb") as file:
        return analyze_image(file.read())


def mean(values):
    return round(statistics.mean(values), 2) if values else None


def median(values):
    return round(statistics.median(values), 2) if values else None


def recommendation_names(result, category):
    return [
        item.get("name")
        for item in result.get("recommendations", {}).get(category, [])
        if item.get("name")
    ]


def overlap_score(first, second):
    a = set(first)
    b = set(second)

    if not a and not b:
        return 1.0

    if not a or not b:
        return 0.0

    return len(a & b) / len(a | b)


def main():
    directory = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        "validation_samples"
    )

    if not directory.exists():
        print(f"ERROR: Directory not found: {directory}")
        print()
        print("Create it and place your validation photos inside:")
        print(f"  {directory / 'front_daylight.jpg'}")
        print(f"  {directory / 'front_indoor.jpg'}")
        print(f"  {directory / 'slightly_different_angle.jpg'}")
        return 1

    files = sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED
    )

    if len(files) < 2:
        print("ERROR: At least 2 validation images are required.")
        return 1

    print()
    print("============================================================")
    print("TINAYU PROFILE STABILITY EVALUATION")
    print("============================================================")
    print()
    print("This test measures consistency across photos of the same")
    print("person. It is NOT a ground-truth accuracy test.")
    print()
    print(f"Directory: {directory.resolve()}")
    print(f"Images:    {len(files)}")
    print()

    successful = []
    failures = []

    for index, path in enumerate(files, start=1):
        print(f"[{index}/{len(files)}] {path.name}")

        try:
            result = load_result(path)

            if not result.get("success"):
                failures.append({
                    "file": path.name,
                    "error": result.get("error", "Unknown analysis failure"),
                })
                print("  FAIL:", result.get("error", "Unknown analysis failure"))
                continue

            successful.append((path.name, result))

            profile = result.get("profile", {})
            heuristics = profile.get("heuristics", {})
            skin = profile.get("skin", {})
            quality = result.get("quality", {})

            print(
                "  season:",
                heuristics.get("suggested_season"),
                "| temp:",
                heuristics.get("temperature"),
                "| saturation:",
                heuristics.get("skin_saturation"),
                "| contrast:",
                heuristics.get("contrast_level"),
            )
            print(
                "  quality:",
                quality.get("image_confidence"),
                "| skin pixels:",
                quality.get("skin_pixels_analyzed"),
            )

        except Exception as error:
            failures.append({
                "file": path.name,
                "error": f"{type(error).__name__}: {error}",
            })
            print(f"  ERROR: {type(error).__name__}: {error}")

    print()
    print("============================================================")
    print("PIPELINE SUCCESS")
    print("============================================================")
    print()

    success_rate = len(successful) / len(files) * 100

    print(f"Successful: {len(successful)}/{len(files)} ({success_rate:.1f}%)")
    print(f"Failed:     {len(failures)}")

    if not successful:
        print()
        print("No successful analyses available for stability testing.")
        return 1

    # ---------------------------------------------------------
    # PROFILE STABILITY
    # ---------------------------------------------------------

    seasons = [
        result.get("profile", {})
        .get("heuristics", {})
        .get("suggested_season")
        for _, result in successful
    ]

    temperatures = [
        result.get("profile", {})
        .get("heuristics", {})
        .get("temperature")
        for _, result in successful
    ]

    saturations = [
        result.get("profile", {})
        .get("heuristics", {})
        .get("skin_saturation")
        for _, result in successful
    ]

    contrasts = [
        result.get("profile", {})
        .get("heuristics", {})
        .get("contrast_level")
        for _, result in successful
    ]

    season_counts = {
        value: seasons.count(value)
        for value in sorted(set(seasons))
        if value is not None
    }

    print()
    print("============================================================")
    print("PROFILE CONSISTENCY")
    print("============================================================")
    print()

    print("Seasons:", season_counts)
    print("Temperature:", sorted(set(temperatures)))
    print("Saturation:", sorted(set(saturations)))
    print("Contrast:", sorted(set(contrasts)))

    dominant_season = max(
        season_counts,
        key=season_counts.get
    ) if season_counts else None

    dominant_count = (
        season_counts.get(dominant_season, 0)
        if dominant_season
        else 0
    )

    season_consistency = (
        dominant_count / len(successful) * 100
        if successful
        else 0
    )

    print(
        f"Dominant season: {dominant_season} "
        f"({season_consistency:.1f}% of successful analyses)"
    )

    # ---------------------------------------------------------
    # SKIN COLOR STABILITY
    # ---------------------------------------------------------

    lightness = []
    a_values = []
    b_values = []
    hue_values = []
    chroma_values = []

    for _, result in successful:
        skin = result.get("profile", {}).get("skin", {})

        lab = skin.get("lab", {})

        if isinstance(lab.get("L"), (int, float)):
            lightness.append(float(lab["L"]))

        if isinstance(lab.get("a"), (int, float)):
            a_values.append(float(lab["a"]))

        if isinstance(lab.get("b"), (int, float)):
            b_values.append(float(lab["b"]))

        if isinstance(skin.get("hue"), (int, float)):
            hue_values.append(float(skin["hue"]))

        if isinstance(skin.get("chroma"), (int, float)):
            chroma_values.append(float(skin["chroma"]))

    print()
    print("============================================================")
    print("NORMALIZED SKIN STABILITY")
    print("============================================================")
    print()

    print("LAB L   mean:", mean(lightness), "median:", median(lightness))
    print("LAB a   mean:", mean(a_values), "median:", median(a_values))
    print("LAB b   mean:", mean(b_values), "median:", median(b_values))
    print("Hue     mean:", mean(hue_values), "median:", median(hue_values))
    print("Chroma  mean:", mean(chroma_values), "median:", median(chroma_values))

    if lightness:
        print("L range:", round(max(lightness) - min(lightness), 2))

    if a_values:
        print("a range:", round(max(a_values) - min(a_values), 2))

    if b_values:
        print("b range:", round(max(b_values) - min(b_values), 2))

    if hue_values:
        print("Hue range:", round(max(hue_values) - min(hue_values), 2))

    if chroma_values:
        print("Chroma range:", round(max(chroma_values) - min(chroma_values), 2))

    # ---------------------------------------------------------
    # RECOMMENDATION STABILITY
    # ---------------------------------------------------------

    categories = ["clothing", "makeup", "accents"]

    print()
    print("============================================================")
    print("RECOMMENDATION CONSISTENCY")
    print("============================================================")
    print()

    pair_scores = {category: [] for category in categories}

    for i in range(len(successful)):
        for j in range(i + 1, len(successful)):
            first_name, first_result = successful[i]
            second_name, second_result = successful[j]

            print(f"{first_name} <-> {second_name}")

            for category in categories:
                first = recommendation_names(first_result, category)
                second = recommendation_names(second_result, category)

                score = overlap_score(first, second)
                pair_scores[category].append(score)

                print(
                    f"  {category:8s}: "
                    f"{score * 100:.1f}% top-list overlap"
                )

    print()
    for category in categories:
        values = pair_scores[category]

        if values:
            print(
                f"{category.capitalize():8s} average overlap: "
                f"{statistics.mean(values) * 100:.1f}%"
            )

    # ---------------------------------------------------------
    # NORMALIZATION
    # ---------------------------------------------------------

    normalization_values = [
        result.get("quality", {}).get("normalization_applied")
        for _, result in successful
    ]

    normalization_rate = (
        sum(value is True for value in normalization_values)
        / len(successful)
        * 100
    )

    print()
    print("============================================================")
    print("NORMALIZATION")
    print("============================================================")
    print()

    print(
        f"Normalization applied: "
        f"{sum(value is True for value in normalization_values)}"
        f"/{len(successful)} "
        f"({normalization_rate:.1f}%)"
    )

    # ---------------------------------------------------------
    # JSON REPORT
    # ---------------------------------------------------------

    report = {
        "image_count": len(files),
        "successful": len(successful),
        "failed": len(failures),
        "success_rate": round(success_rate, 2),
        "dominant_season": dominant_season,
        "season_consistency": round(season_consistency, 2),
        "season_counts": season_counts,
        "normalization_rate": round(normalization_rate, 2),
        "skin_lab": {
            "L_mean": mean(lightness),
            "L_median": median(lightness),
            "a_mean": mean(a_values),
            "a_median": median(a_values),
            "b_mean": mean(b_values),
            "b_median": median(b_values),
        },
        "recommendation_overlap": {
            category: round(
                statistics.mean(values) * 100,
                2
            ) if values else None
            for category, values in pair_scores.items()
        },
        "failures": failures,
    }

    output_path = directory / "tinayu_stability_report.json"

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    print()
    print("============================================================")
    print("REPORT")
    print("============================================================")
    print()
    print(f"Saved: {output_path}")
    print()
    print("Important:")
    print("- This evaluates stability, not personal-color ground truth.")
    print("- A stable season does not prove the season is objectively correct.")
    print("- Use FairFace separately for CV robustness.")
    print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
