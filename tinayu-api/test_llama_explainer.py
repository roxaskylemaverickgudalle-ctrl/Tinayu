from llama_explainer import generate_explanation


profile = {
    "skin": {
        "hue": 22.0,
        "chroma": 30.81
    },
    "heuristics": {
        "skin_category": "Light",
        "temperature": "Warm",
        "temperature_strength": 0.79,
        "temperature_strength_band": "Strong Warm",
        "suggested_season": "Warm Spring",
        "skin_depth": "Light",
        "skin_saturation": "Clear",
        "contrast_level": "High"
    }
}


recommendations = {
    "clothing": [
        {"name": "Camel", "score": 95.0},
        {"name": "Warm Beige", "score": 94.9},
        {"name": "Dusty Rose", "score": 93.2},
        {"name": "Peach", "score": 93.1},
        {"name": "Mustard", "score": 92.8}
    ],
    "makeup": [
        {"name": "Warm Nude", "score": 95.0},
        {"name": "Dusty Rose", "score": 93.1},
        {"name": "Peach", "score": 92.6}
    ],
    "accents": [
        {"name": "Warm Beige", "score": 95.0},
        {"name": "Muted Gold", "score": 90.6},
        {"name": "Antique Gold", "score": 90.1}
    ]
}


print()
print("========== TINAYU LLAMA EXPLANATION ==========")
print()

result = generate_explanation(
    profile,
    recommendations
)

print(result)

print()
print("==============================================")
