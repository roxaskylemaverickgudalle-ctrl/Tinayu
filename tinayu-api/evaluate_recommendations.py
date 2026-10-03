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
        print(
            f"{index}. "
            f"{item['name']:<14} "
            f"{item['hex']}  "
            f"{item['score']}/100"
        )




def analyze_diversity():
    print()
    print('=' * 72)
    print('RECOMMENDATION DIVERSITY STRESS TEST')
    print('=' * 72)

    for profile_name, profile in PROFILES.items():
        recommendations = build_recommendation(PALETTE, profile, 'clothing', limit=10)

        families = [item.get('family', 'Unknown') for item in recommendations]
        unique_families = len(set(families))

        print(f'\n{profile_name}')
        print('-' * 72)
        print(f'Unique hue families in top 10: {unique_families}/{len(families)}')
        print('Family distribution:')

        counts = {}
        for family in families:
            counts[family] = counts.get(family, 0) + 1
        for family, count in sorted(counts.items(), key=lambda x: (-x[1], x[0])):
            print(f'  {family:<10} {count}')

        print('Top 10:')
        for i, item in enumerate(recommendations, 1):
            print(f" {i:>2}. {item['name']:<15} {item['family']:<10} {item['compatibility']:.1f}/100")

        if unique_families <= 2:
            print('DIVERSITY FLAG: High family concentration')
        elif unique_families <= 3:
            print('DIVERSITY FLAG: Moderate family concentration')
        else:
            print('DIVERSITY FLAG: Healthy family spread')

def main():
    print("TINAYU RECOMMENDATION ENGINE — CONTROLLED EVALUATION")
    print("Baseline test. No engine modifications.")

    for name, profile in PROFILES.items():
        evaluate_profile(name, profile)


if __name__ == "__main__":
    main()
    analyze_diversity()
