from tinayu_engine import (
    score_color,
    get_color_features,
    build_recommendation,
)


PALETTE = [
    ("Warm Beige", "#C8A27A"),
    ("Peach", "#F2A07B"),
    ("Camel", "#C19A6B"),
    ("Dusty Rose", "#C08081"),
    ("Mustard", "#C2A83E"),
    ("Warm Brown", "#8B5A2B"),
    ("Brown", "#8B5E3C"),
    ("Olive", "#808000"),
    ("Cool Blue", "#5B7DB1"),
    ("Cool Pink", "#C080A8"),
    ("Lavender", "#9B8FC4"),
    ("Navy", "#263A5A"),
    ("Emerald", "#2E8B57"),
    ("Soft Gray", "#A8A8A8"),
    ("Black", "#111111"),
]


PROFILES = {
    "Warm Spring": {
        "skin": {
            "rgb": [220, 170, 130],
            "lab": {"L": 70, "a": 15, "b": 25},
        },
        "heuristics": {
            "suggested_season": "Warm Spring",
            "contrast_level": "High",
        },
    },

    "Cool Summer": {
        "skin": {
            "rgb": [205, 175, 170],
            "lab": {"L": 72, "a": 8, "b": 5},
        },
        "heuristics": {
            "suggested_season": "Cool Summer",
            "contrast_level": "Low",
        },
    },

    "Warm Autumn": {
        "skin": {
            "rgb": [185, 145, 115],
            "lab": {"L": 62, "a": 14, "b": 24},
        },
        "heuristics": {
            "suggested_season": "Warm Autumn",
            "contrast_level": "Medium",
        },
    },

    "Cool Winter": {
        "skin": {
            "rgb": [190, 175, 180],
            "lab": {"L": 70, "a": 5, "b": -2},
        },
        "heuristics": {
            "suggested_season": "Cool Winter",
            "contrast_level": "High",
        },
    },
}


def get_hue_family_from_features(features):
    """
    Convert a color's hue into a broad hue family for
    recommendation diversity analysis.

    This is evaluator-side metadata only.
    It does not modify Tinayu's recommendation engine.
    """

    hue = float(features["hue"]) % 360

    if 0 <= hue < 15:
        return "Red"

    if 15 <= hue < 45:
        return "Orange"

    if 45 <= hue < 75:
        return "Yellow"

    if 75 <= hue < 165:
        return "Green"

    if 165 <= hue < 195:
        return "Cyan"

    if 195 <= hue < 255:
        return "Blue"

    if 255 <= hue < 285:
        return "Purple"

    if 285 <= hue < 345:
        return "Pink"

    return "Red"


def get_family_for_color(color_name, hex_color):
    """
    Special-case brown shades so the evaluator matches
    Tinayu's recommendation hue-family convention.

    This is only used for diversity reporting.
    """

    features = get_color_features(hex_color)

    hue = float(features["hue"]) % 360
    lightness = float(features["lab"][0])

    if 20 <= hue < 45 and lightness < 55:
        return "Brown"

    return get_hue_family_from_features(features)


def build_evaluation_metadata():
    """
    Precompute evaluator-side metadata for the palette.
    """

    metadata = {}

    for color_name, hex_color in PALETTE:
        features = get_color_features(hex_color)

        metadata[color_name] = {
            "family": get_family_for_color(
                color_name,
                hex_color,
            ),
            "hue": features["hue"],
            "chroma": features["chroma"],
        }

    return metadata


PALETTE_METADATA = build_evaluation_metadata()


def evaluate_profile(name, profile):
    print("\n" + "=" * 72)
    print(name)
    print("=" * 72)

    scored = []

    for color_name, hex_color in PALETTE:
        score = score_color(
            color_name,
            hex_color,
            profile,
            "clothing",
        )

        features = get_color_features(hex_color)

        scored.append({
            "name": color_name,
            "hex": hex_color,
            "score": score,
            "hue": features["hue"],
            "chroma": features["chroma"],
            "family": PALETTE_METADATA[color_name]["family"],
        })

    scored.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    print("\nUnderlying compatibility ranking:")
    print("-" * 72)

    for index, item in enumerate(scored, start=1):
        print(
            f"{index:2}. "
            f"{item['name']:<14} "
            f"{item['hex']}  "
            f"{item['score']:.4f}"
        )

    recommendations = build_recommendation(
        PALETTE,
        profile,
        "clothing",
        limit=5,
    )

    print("\nFinal recommendations:")
    print("-" * 72)

    for index, item in enumerate(recommendations, start=1):
        score = item.get("score", 0)

        print(
            f"{index}. "
            f"{item['name']:<14} "
            f"{item['hex']}  "
            f"{score}/100"
        )


def analyze_diversity():
    print()
    print("=" * 72)
    print("RECOMMENDATION DIVERSITY STRESS TEST")
    print("=" * 72)

    for profile_name, profile in PROFILES.items():
        recommendations = build_recommendation(
            PALETTE,
            profile,
            "clothing",
            limit=10,
        )

        enriched = []

        for item in recommendations:
            color_name = item["name"]
            hex_color = item["hex"]

            metadata = PALETTE_METADATA.get(
                color_name,
                {},
            )

            family = metadata.get(
                "family",
                "Unknown",
            )

            score = item.get(
                "score",
                item.get("compatibility", 0),
            )

            enriched.append({
                **item,
                "family": family,
                "score": score,
            })

        families = [
            item["family"]
            for item in enriched
        ]

        unique_families = len(set(families))

        print(f"\n{profile_name}")
        print("-" * 72)

        print(
            f"Unique hue families in top 10: "
            f"{unique_families}/{len(families)}"
        )

        print("Family distribution:")

        counts = {}

        for family in families:
            counts[family] = counts.get(
                family,
                0,
            ) + 1

        for family, count in sorted(
            counts.items(),
            key=lambda x: (-x[1], x[0]),
        ):
            print(
                f"  {family:<10} {count}"
            )

        print("Top 10:")

        for index, item in enumerate(
            enriched,
            start=1,
        ):
            print(
                f" {index:>2}. "
                f"{item['name']:<15} "
                f"{item['family']:<10} "
                f"{item['score']:.1f}/100"
            )

        if unique_families <= 2:
            print(
                "DIVERSITY FLAG: "
                "High family concentration"
            )

        elif unique_families <= 3:
            print(
                "DIVERSITY FLAG: "
                "Moderate family concentration"
            )

        else:
            print(
                "DIVERSITY FLAG: "
                "Healthy family spread"
            )


def main():
    print(
        "TINAYU RECOMMENDATION ENGINE — "
        "CONTROLLED EVALUATION"
    )

    print(
        "Baseline test. No engine modifications."
    )

    for name, profile in PROFILES.items():
        evaluate_profile(
            name,
            profile,
        )


if __name__ == "__main__":
    main()
    analyze_diversity()