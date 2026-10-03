
from pathlib import Path
import json
import traceback

from tinayu_engine import analyze_image


BASE_DIR = Path(__file__).resolve().parent
IMAGE_PATH = BASE_DIR / "test.jpg"


def print_section(title):
    print(f"\n========== {title} ==========\n")


def main():
    print_section("TINAYU ANALYSIS")

    # ---------------------------------------------------------
    # CHECK TEST IMAGE
    # ---------------------------------------------------------

    if not IMAGE_PATH.exists():
        print("ERROR: Test image not found:")
        print(IMAGE_PATH)
        return

    print(f"Test image: {IMAGE_PATH}")

    # ---------------------------------------------------------
    # READ IMAGE
    # ---------------------------------------------------------

    try:
        with open(IMAGE_PATH, "rb") as file:
            image_bytes = file.read()

    except Exception as error:
        print_section("ERROR READING IMAGE")
        print(f"{type(error).__name__}: {error}")
        return

    print(f"Image bytes: {len(image_bytes):,}")

    # ---------------------------------------------------------
    # RUN TINAYU ENGINE
    # ---------------------------------------------------------

    try:
        result = analyze_image(image_bytes)

    except Exception as error:
        print_section("ANALYSIS ERROR")

        print(f"{type(error).__name__}: {error}")

        traceback.print_exc()

        return

    # ---------------------------------------------------------
    # BASIC RESULT
    # ---------------------------------------------------------

    print_section("RESULT")

    print("Success:", result.get("success"))

    if not result.get("success"):
        print("\nAnalysis failed.")

        print_section("FULL RESPONSE")

        print(
            json.dumps(
                result,
                indent=2,
                default=str
            )
        )

        return

    # ---------------------------------------------------------
    # EXTRACT RESULT SECTIONS
    # ---------------------------------------------------------

    image = result.get("image", {})
    face = result.get("face", {})
    colors = result.get("colors", {})
    profile = result.get("profile", {})
    quality = result.get("quality", {})
    normalization = result.get("normalization", {})
    recommendations = result.get("recommendations", {})

    # ---------------------------------------------------------
    # IMAGE
    # ---------------------------------------------------------

    print_section("IMAGE")

    print("Width:", image.get("width"))
    print("Height:", image.get("height"))

    # ---------------------------------------------------------
    # FACE
    # ---------------------------------------------------------

    print_section("FACE")

    print("Face confidence:", face.get("confidence"))
    print("Landmarks:", face.get("landmarks"))

    # ---------------------------------------------------------
    # COLORS
    # ---------------------------------------------------------

    print_section("COLORS")

    # These keys match the actual response returned by
    # tinayu_engine.py.
    print("Raw skin RGB:", colors.get("skin_raw_rgb"))
    print("Normalized skin RGB:", colors.get("skin_normalized_rgb"))
    print("Skin LAB:", colors.get("skin_lab"))
    print("Hair RGB:", colors.get("hair_rgb"))
    print("Eyes RGB:", colors.get("eye_rgb"))

    # ---------------------------------------------------------
    # PROFILE
    # ---------------------------------------------------------

    print_section("PROFILE")

    print(
        json.dumps(
            profile,
            indent=2,
            default=str
        )
    )

    # ---------------------------------------------------------
    # QUALITY
    # ---------------------------------------------------------

    print_section("QUALITY")

    print(
        json.dumps(
            quality,
            indent=2,
            default=str
        )
    )

    # ---------------------------------------------------------
    # NORMALIZATION
    # ---------------------------------------------------------

    print_section("NORMALIZATION")

    print(
        json.dumps(
            normalization,
            indent=2,
            default=str
        )
    )

    # ---------------------------------------------------------
    # CLOTHING
    # ---------------------------------------------------------

    print_section("CLOTHING")

    clothing = recommendations.get("clothing", [])

    if clothing:
        for index, color in enumerate(clothing, start=1):
            print(
                f"{index}. "
                f"{color.get('name')} "
                f"{color.get('hex')} "
                f"Score: {color.get('score')}"
            )
    else:
        print("No clothing recommendations returned.")

    # ---------------------------------------------------------
    # MAKEUP
    # ---------------------------------------------------------

    print_section("MAKEUP")

    makeup = recommendations.get("makeup", [])

    if makeup:
        for index, color in enumerate(makeup, start=1):
            print(
                f"{index}. "
                f"{color.get('name')} "
                f"{color.get('hex')} "
                f"Score: {color.get('score')}"
            )
    else:
        print("No makeup recommendations returned.")

    # ---------------------------------------------------------
    # ACCENTS
    # ---------------------------------------------------------

    print_section("ACCENTS")

    accents = recommendations.get("accents", [])

    if accents:
        for index, color in enumerate(accents, start=1):
            print(
                f"{index}. "
                f"{color.get('name')} "
                f"{color.get('hex')} "
                f"Score: {color.get('score')}"
            )
    else:
        print("No accent recommendations returned.")

    # ---------------------------------------------------------
    # RECOMMENDATIONS JSON
    # ---------------------------------------------------------

    print_section("RECOMMENDATIONS JSON")

    print(
        json.dumps(
            recommendations,
            indent=2,
            default=str
        )
    )

    # ---------------------------------------------------------
    # FULL RESULT
    # ---------------------------------------------------------

    print_section("FULL RESULT")

    print(
        json.dumps(
            result,
            indent=2,
            default=str
        )
    )

    print("\n==========================================\n")


if __name__ == "__main__":
    main()

