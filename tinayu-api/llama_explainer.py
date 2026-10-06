import json
import os
import re

from dotenv import load_dotenv
from huggingface_hub import InferenceClient


load_dotenv()

LLAMA_ENABLED = os.getenv("LLAMA_ENABLED", "true").lower() == "true"

LLAMA_MODEL = os.getenv(
    "LLAMA_MODEL",
    "meta-llama/Llama-3.1-8B-Instruct"
)


def build_explanation_data(profile, recommendations):
    heuristics = profile.get("heuristics", {})
    skin = profile.get("skin", {})

    return {
        "season": heuristics.get("suggested_season"),
        "temperature": heuristics.get("temperature"),
        "temperature_strength_band": heuristics.get(
            "temperature_strength_band"
        ),
        "skin_depth": heuristics.get("skin_depth"),
        "skin_saturation": heuristics.get("skin_saturation"),
        "contrast_level": heuristics.get("contrast_level"),
        "hue": skin.get("hue"),
        "chroma": skin.get("chroma"),
        "clothing": [
            item.get("name")
            for item in recommendations.get("clothing", [])[:5]
        ],
        "makeup": [
            item.get("name")
            for item in recommendations.get("makeup", [])[:5]
        ],
        "accents": [
            item.get("name")
            for item in recommendations.get("accents", [])[:5]
        ],
    }


def build_explanation_prompt(profile, recommendations):
    data = build_explanation_data(
        profile,
        recommendations
    )

    return f"""
You are Tinayu's explanation writer.

Tinayu has already determined the color analysis.
Your job is ONLY to turn the supplied facts into three short sentences.

FACTS FROM TINAYU:
{json.dumps(data, indent=2)}

Return ONLY valid JSON:

{{
  "profile": "...",
  "why": "...",
  "color_direction": "..."
}}

STRICT RULES:

1. Use ONLY facts explicitly present in the Tinayu data.
2. Never invent facts.
3. Never mention physical characteristics.
4. Never mention color preferences.
5. Never mention attractiveness or appearance.
6. Never say a color is flattering, complementary, enhancing, or suitable for someone's skin.
7. Never introduce "muted" when skin_saturation is "Clear".
8. Never change "Clear" into "Muted".
9. Never change "Warm" into "Cool" or "Neutral".
10. Preserve the supplied season exactly.
11. Preserve the supplied temperature exactly.
12. Preserve the supplied skin depth exactly.
13. Preserve the supplied skin saturation exactly.
14. Preserve the supplied contrast level exactly.
15. Use the supplied recommendation names only.
16. Use normal spaces between every word.
17. Do not combine words.
18. Do not use markdown.
19. Do not add explanations outside the JSON.
20. Keep every field to one short sentence.

Use these meanings:

- profile: state the supplied season, temperature, skin depth, skin saturation, and contrast.
- why: explain the color direction using the supplied temperature, saturation, contrast, and season.
- color_direction: describe the supplied recommended colors using only their names.

Example structure:

{{
  "profile": "The profile is Warm Spring with a Warm temperature, Light skin depth, Clear skin saturation, and High contrast.",
  "why": "The Warm temperature and Clear saturation indicate a warm and vibrant color direction.",
  "color_direction": "The recommended direction includes warm colors such as Camel, Warm Beige, Dusty Rose, Peach, and Mustard."
}}
""".strip()


def clean_text(text):
    text = str(text)

    replacements = {
        "individualwith": "individual with",
        "profilewith": "profile with",
        "seasonwith": "season with",

        "strongemphasis": "strong emphasis",
        "strongemphasis": "strong emphasis",
        "ofcontrast": "of contrast",

        "Warmand": "Warm and",
        "warmand": "warm and",
        "Clearand": "Clear and",
        "clearand": "clear and",
        "Brightand": "Bright and",
        "brightand": "bright and",
        "Richand": "Rich and",
        "richand": "rich and",
        "Vibrantand": "Vibrant and",
        "vibrantand": "vibrant and",

        "WarmSpring": "Warm Spring",
        "CoolSummer": "Cool Summer",
        "CoolWinter": "Cool Winter",
        "WarmAutumn": "Warm Autumn",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.!?])", r"\1", text)

    return text.strip()


def parse_explanation(raw_text):
    raw_text = raw_text.strip()

    raw_text = re.sub(
        r"^```(?:json)?\s*",
        "",
        raw_text,
        flags=re.IGNORECASE
    )

    raw_text = re.sub(
        r"\s*```$",
        "",
        raw_text
    )

    try:
        parsed = json.loads(raw_text)

        if isinstance(parsed, dict):
            result = {
                "profile": clean_text(
                    parsed.get("profile", "")
                ),
                "why": clean_text(
                    parsed.get("why", "")
                ),
                "color_direction": clean_text(
                    parsed.get("color_direction", "")
                ),
            }

            if all(result.values()):
                return result

    except json.JSONDecodeError:
        pass

    sections = {
        "profile": "",
        "why": "",
        "color_direction": "",
    }

    current = None

    for line in raw_text.splitlines():
        line = line.strip()

        if not line:
            continue

        normalized = line.lower().rstrip(":").strip()

        if normalized == "profile":
            current = "profile"
            continue

        if normalized == "why":
            current = "why"
            continue

        if normalized in {
            "color direction",
            "color_direction",
        }:
            current = "color_direction"
            continue

        if current:
            sections[current] += " " + line

    sections = {
        key: clean_text(value)
        for key, value in sections.items()
    }

    if all(sections.values()):
        return sections

    raise ValueError(
        "Llama returned an unsupported explanation format."
    )


def generate_explanation(profile, recommendations):
    if not LLAMA_ENABLED:
        return None

    token = os.getenv("HF_TOKEN")

    if not token:
        raise RuntimeError(
            "HF_TOKEN is not configured."
        )

    prompt = build_explanation_prompt(
        profile,
        recommendations
    )

    client = InferenceClient(
        api_key=token
    )

    response = client.chat_completion(
        model=LLAMA_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are the Tinayu explanation component. "
                    "Follow the supplied data exactly. "
                    "Return only the requested JSON."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        max_tokens=220,
        temperature=0.1,
    )

    raw_text = response.choices[0].message.content

    return parse_explanation(raw_text)