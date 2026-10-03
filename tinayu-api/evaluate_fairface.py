import io
import json
import time
from pathlib import Path

from datasets import load_dataset
from PIL import Image

from tinayu_engine import analyze_image


# ============================================================
# TINAYU — FAIRFACE 25-IMAGE BENCHMARK
# ============================================================

SAMPLE_SIZE = 100
SEED = 42

OUTPUT_DIR = Path("fairface_evaluation")
IMAGE_DIR = OUTPUT_DIR / "images"

RESULTS_FILE = OUTPUT_DIR / "fairface_results.json"
SUMMARY_FILE = OUTPUT_DIR / "fairface_summary.json"


def safe_get(data, *keys, default=None):
    """Safely retrieve nested dictionary values."""
    current = data

    for key in keys:
        if not isinstance(current, dict):
            return default

        current = current.get(key)

        if current is None:
            return default

    return current


def save_image(image, path):
    """Save a PIL image to disk."""
    image.save(path, format="JPEG", quality=95)


def main():
    print("=" * 60)
    print("TINAYU — FAIRFACE 25-IMAGE BENCHMARK")
    print("=" * 60)

    OUTPUT_DIR.mkdir(exist_ok=True)
    IMAGE_DIR.mkdir(exist_ok=True)

    print("\n[1/4] Loading FairFace validation dataset...")

    dataset = load_dataset(
        "HuggingFaceM4/FairFace",
        "0.25",
        split="validation",
    )

    print(f"Dataset loaded: {len(dataset):,} images")

    print(f"\nSelecting {SAMPLE_SIZE} images...")

    sample = dataset.shuffle(seed=SEED).select(
        range(min(SAMPLE_SIZE, len(dataset)))
    )

    print(f"Selected: {len(sample)} images")

    results = []

    total_start = time.perf_counter()

    successful = 0
    failed = 0

    face_detected = 0
    hair_detected = 0
    eyes_detected = 0
    normalization_applied = 0

    processing_times = []

    print("\n[2/4] Running Tinayu analysis...")
    print("-" * 60)

    for index, item in enumerate(sample):
        image_start = time.perf_counter()

        image_path = IMAGE_DIR / f"fairface_{index + 1:03d}.jpg"

        try:
            image = item["image"]

            if not isinstance(image, Image.Image):
                image = Image.fromarray(image)

            image = image.convert("RGB")

            save_image(image, image_path)

            # Convert PIL image to bytes for Tinayu.
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=95)

            image_bytes = buffer.getvalue()

            # Run the actual Tinayu pipeline.
            result = analyze_image(image_bytes)

            elapsed = time.perf_counter() - image_start
            processing_times.append(elapsed)

            success = bool(
                result.get("success", False)
                if isinstance(result, dict)
                else False
            )

            quality = result.get("quality", {}) if isinstance(result, dict) else {}
            face = result.get("face", {}) if isinstance(result, dict) else {}
            profile = result.get("profile", {}) if isinstance(result, dict) else {}
            recommendations = (
                result.get("recommendations", {})
                if isinstance(result, dict)
                else {}
            )

            has_face = bool(face.get("confidence") is not None)
            has_hair = bool(quality.get("hair_detected", False))
            has_eyes = bool(quality.get("eyes_detected", False))
            normalized = bool(quality.get("normalization_applied", False))

            if success:
                successful += 1
            else:
                failed += 1

            if has_face:
                face_detected += 1

            if has_hair:
                hair_detected += 1

            if has_eyes:
                eyes_detected += 1

            if normalized:
                normalization_applied += 1

            result_record = {
                "index": index + 1,
                "image_file": str(image_path),
                "processing_time_seconds": round(elapsed, 4),
                "success": success,

                "face": {
                    "confidence": face.get("confidence"),
                    "landmarks": face.get("landmarks"),
                    "detected": has_face,
                },

                "quality": {
                    "image_confidence": quality.get("image_confidence"),
                    "hair_detected": has_hair,
                    "eyes_detected": has_eyes,
                    "normalization_applied": normalized,
                    "skin_pixels_analyzed": quality.get(
                        "skin_pixels_analyzed"
                    ),
                },

                "profile": {
                    "skin_category": safe_get(
                        profile,
                        "heuristics",
                        "skin_category",
                    ),
                    "temperature": safe_get(
                        profile,
                        "heuristics",
                        "temperature",
                    ),
                    "suggested_season": safe_get(
                        profile,
                        "heuristics",
                        "suggested_season",
                    ),
                    "skin_depth": safe_get(
                        profile,
                        "heuristics",
                        "skin_depth",
                    ),
                    "skin_saturation": safe_get(
                        profile,
                        "heuristics",
                        "skin_saturation",
                    ),
                    "contrast_level": safe_get(
                        profile,
                        "heuristics",
                        "contrast_level",
                    ),
                },

                "recommendations": {
                    "clothing_count": len(
                        recommendations.get("clothing", [])
                    ),
                    "makeup_count": len(
                        recommendations.get("makeup", [])
                    ),
                    "accents_count": len(
                        recommendations.get("accents", [])
                    ),
                },

                "error": None,
            }

            results.append(result_record)

            print(
                f"[{index + 1:02d}/{len(sample):02d}] "
                f"{'OK' if success else 'FAIL'} | "
                f"{elapsed:.2f}s | "
                f"season={result_record['profile']['suggested_season']} | "
                f"face={has_face} | "
                f"hair={has_hair} | "
                f"eyes={has_eyes}"
            )

        except Exception as exc:
            elapsed = time.perf_counter() - image_start
            processing_times.append(elapsed)

            failed += 1

            error_record = {
                "index": index + 1,
                "image_file": str(image_path),
                "processing_time_seconds": round(elapsed, 4),
                "success": False,
                "face": {},
                "quality": {},
                "profile": {},
                "recommendations": {},
                "error": f"{type(exc).__name__}: {exc}",
            }

            results.append(error_record)

            print(
                f"[{index + 1:02d}/{len(sample):02d}] "
                f"ERROR | {elapsed:.2f}s | {type(exc).__name__}: {exc}"
            )

    total_elapsed = time.perf_counter() - total_start

    average_time = (
        sum(processing_times) / len(processing_times)
        if processing_times
        else 0
    )

    summary = {
        "dataset": "FairFace",
        "split": "validation",
        "sample_size": len(sample),
        "seed": SEED,

        "runtime": {
            "total_seconds": round(total_elapsed, 3),
            "average_seconds_per_image": round(average_time, 3),
            "estimated_images_per_minute": round(
                60 / average_time,
                2
            )
            if average_time > 0
            else 0,
        },

        "pipeline": {
            "successful": successful,
            "failed": failed,
            "success_rate_percent": round(
                successful / len(sample) * 100,
                2,
            )
            if sample
            else 0,

            "face_detected": face_detected,
            "face_detection_rate_percent": round(
                face_detected / len(sample) * 100,
                2,
            )
            if sample
            else 0,

            "hair_detected": hair_detected,
            "hair_detection_rate_percent": round(
                hair_detected / len(sample) * 100,
                2,
            )
            if sample
            else 0,

            "eyes_detected": eyes_detected,
            "eye_detection_rate_percent": round(
                eyes_detected / len(sample) * 100,
                2,
            )
            if sample
            else 0,

            "normalization_applied": normalization_applied,
            "normalization_rate_percent": round(
                normalization_applied / len(sample) * 100,
                2,
            )
            if sample
            else 0,
        },
    }

    print("\n[3/4] Saving results...")

    RESULTS_FILE.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    SUMMARY_FILE.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print(f"Results: {RESULTS_FILE}")
    print(f"Summary: {SUMMARY_FILE}")

    print("\n[4/4] BENCHMARK SUMMARY")
    print("=" * 60)

    print(f"Images processed:       {len(sample)}")
    print(f"Successful:             {successful}")
    print(f"Failed:                 {failed}")
    print(f"Success rate:           {summary['pipeline']['success_rate_percent']}%")

    print()
    print(f"Face detection:         {summary['pipeline']['face_detection_rate_percent']}%")
    print(f"Hair detection:         {summary['pipeline']['hair_detection_rate_percent']}%")
    print(f"Eye detection:          {summary['pipeline']['eye_detection_rate_percent']}%")
    print(f"Normalization:          {summary['pipeline']['normalization_rate_percent']}%")

    print()
    print(f"Total runtime:          {total_elapsed:.2f} seconds")
    print(f"Average per image:      {average_time:.2f} seconds")

    if average_time > 0:
        print(
            f"Processing speed:       "
            f"{summary['runtime']['estimated_images_per_minute']:.2f} images/min"
        )

    print("=" * 60)
    print("\n25-image benchmark complete.")


if __name__ == "__main__":
    main()