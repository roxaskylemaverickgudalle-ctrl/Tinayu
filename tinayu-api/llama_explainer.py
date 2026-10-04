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
You are the explanation layer for Tinayu, a personal color analysis application.

Your job is ONLY to explain the supplied analysis results.

Use ONLY the information provided below.
Do not invent measurements, colors, traits, diagnoses, or recommendations.

Keep the explanation concise, neutral, and easy to understand.

Return ONLY valid JSON using exactly this structure:

{{
  "profile": "one concise sentence describing the supplied profile",
  "why": "one concise sentence explaining the main color direction using the supplied temperature, saturation, contrast, or season",
  "color_direction": "one concise sentence describing the recommended color direction using supplied recommendation names"
}}

Rules:
- Do not mention AI, models, prompts, or algorithms.
- Do not claim scientific certainty.
- Do not describe attractiveness or physical appearance.
- Do not make claims about personality.
- Do not add recommendations that are not supplied.
- Do not use markdown.
- Do not use bullet points.
- Keep each field to one sentence.
- Use normal spacing between words.
- Keep the wording simple and professional.

Supplied Tinayu analysis:

{json.dumps(data, indent=2)}
""".strip()


def clean_text(text):
    text = str(text)

    # Repair common spacing artifacts produced by the model.
    replacements = {
        "strongemphasis": "strong emphasis",
        "strong emphasis": "strong emphasis",
        "ofcontrast": "of contrast",
        "of contrast": "of contrast",
        "Warmand": "Warm and",
        "warmand": "warm and",
        "Clearand": "Clear and",
        "clearand": "clear and",
        "Brightand": "Bright and",
        "brightand": "bright and",
        "Richand": "Rich and",
        "richand": "rich and",
        "WarmSpring": "Warm Spring",
        "CoolSummer": "Cool Summer",
        "CoolWinter": "Cool Winter",
        "WarmAutumn": "Warm Autumn",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    # Repair repeated whitespace.
    text = re.sub(r"\s+", " ", text)

    # Repair spaces before punctuation.
    text = re.sub(r"\s+([,.!?])", r"\1", text)

    return text.strip()


def parse_explanation(raw_text):
    raw_text = raw_text.strip()

    # Remove accidental markdown code fences.
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

    # First attempt: valid JSON.
    try:
        parsed = json.loads(raw_text)

        if isinstance(parsed, dict):
            result = {
                "profile": clean_text(parsed.get("profile", "")),
                "why": clean_text(parsed.get("why", "")),
                "color_direction": clean_text(
                    parsed.get("color_direction", "")
                ),
            }

            if all(result.values()):
                return result

    except json.JSONDecodeError:
        pass

    # Fallback: recover common heading-based output.
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
    """
    Generate a concise structured explanation using
    Meta Llama through Hugging Face Inference Providers.
    """

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
                    "You are a concise explanation component "
                    "for the Tinayu application. "
                    "Follow the requested JSON format exactly."
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