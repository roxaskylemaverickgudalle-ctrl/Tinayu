from tinayu_engine import (
    get_color_features,
    get_season_profile,
    classify_color_temperature,
    get_hue_family,
    range_score,
    category_compatibility,
    score_color,
)


COLORS = [
    ("Warm Beige", "#C8A27A"),
    ("Camel", "#C19A6B"),
    ("Brown", "#8B5E3C"),
    ("Warm Brown", "#8B5A2B"),
    ("Peach", "#F2A07B"),
    ("Lavender", "#9B8FC4"),
    ("Cool Blue", "#5B7DB1"),
    ("Navy", "#263A5A"),
    ("Cool Pink", "#C080A8"),
    ("Emerald", "#2E8B57"),
]


SEASONS = [
    "Warm Spring",
    "Warm Autumn",
    "Cool Summer",
    "Cool Winter",
]


def make_profile(season):
    return {
        "skin": {
            "rgb": [200, 160, 130],
            "lab": {
                "L": 65,
                "a": 12,
                "b": 18,
            },
        },
        "heuristics": {
            "suggested_season": season,
            "contrast_level": "High",
        },
    }


def main():
    print("=" * 90)
    print("TINAYU RECOMMENDATION COMPONENT DIAGNOSTICS")
    print("=" * 90)

    for color_name, hex_color in COLORS:

        features = get_color_features(hex_color)

        temperature = classify_color_temperature(features)
        family = get_hue_family(features["hue"])
        chroma = float(features["chroma"])
        lightness = float(features["lab"][0])

        print("\n" + "-" * 90)
        print(
            f"{color_name} {hex_color} | "
            f"temperature={temperature} | "
            f"family={family} | "
            f"chroma={chroma:.1f} | "
            f"lightness={lightness:.1f}"
        )
        print("-" * 90)

        for season in SEASONS:

            profile = get_season_profile(season)

            temp_score = profile["temperature"].get(
                temperature,
                0.50,
            )

            chroma_score = range_score(
                chroma,
                profile["chroma_range"][0],
                profile["chroma_range"][1],
            )

            lightness_score = range_score(
                lightness,
                profile["lightness_range"][0],
                profile["lightness_range"][1],
            )

            family_score = profile[
                "preferred_families"
            ].get(
                family,
                0.45,
            )

            season_score = (
                temp_score * 0.30
                + chroma_score * 0.25
                + lightness_score * 0.20
                + family_score * 0.25
            )

            category_score = category_compatibility(
                features,
                "clothing",
                season,
            )

            final_score = score_color(
                color_name,
                hex_color,
                make_profile(season),
                "clothing",
            )

            print(
                f"{season:<13} "
                f"season={season_score:.4f}  "
                f"temp={temp_score:.2f}  "
                f"chroma={chroma_score:.2f}  "
                f"light={lightness_score:.2f}  "
                f"family={family_score:.2f}  "
                f"category={category_score:.4f}  "
                f"FINAL={final_score:.4f}"
            )


if __name__ == "__main__":
    main()
