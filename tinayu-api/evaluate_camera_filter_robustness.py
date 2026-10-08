import json
import sys
from pathlib import Path

import cv2
import numpy as np

from tinayu_engine import analyze_image

SUPPORTED = {".jpg", ".jpeg", ".png", ".webp"}

def encode_image(image):
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise ValueError("Could not encode image.")
    return encoded.tobytes()

def make_variants(image):
    variants = {
        "brightness_up": cv2.convertScaleAbs(image, alpha=1.0, beta=15),
        "brightness_down": cv2.convertScaleAbs(image, alpha=1.0, beta=-15),
        "contrast_up": cv2.convertScaleAbs(image, alpha=1.08, beta=0),
        "contrast_down": cv2.convertScaleAbs(image, alpha=0.92, beta=0),
    }

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.10, 0, 255)
    variants["saturation_up"] = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 0.90, 0, 255)
    variants["saturation_down"] = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 70])
    if ok:
        recompressed = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if recompressed is not None:
            variants["compression"] = recompressed

    return variants

def profile(result):
    p = result.get("profile", {})
    h = p.get("heuristics", {})
    return {
        "temperature": h.get("temperature"),
        "temperature_strength": h.get("temperature_strength"),
        "season": h.get("suggested_season"),
        "skin_category": h.get("skin_category"),
        "skin_saturation": h.get("skin_saturation"),
        "skin_depth": h.get("skin_depth"),
        "contrast": h.get("contrast"),
    }

def compare(base, variant):
    fields = ["temperature", "season", "skin_category", "skin_saturation", "skin_depth", "contrast"]
    changes = {f: base.get(f) != variant.get(f) for f in fields}
    return {
        "changed": any(changes.values()),
        "field_changes": changes,
        "temperature_strength_delta": round(
            float(variant["temperature_strength"]) - float(base["temperature_strength"]), 4
        ),
    }

def main():
    directory = Path(sys.argv[1] if len(sys.argv) > 1 else "fairface_evaluation/images")
    if not directory.exists():
        print(f"ERROR: Directory not found: {directory}")
        return 1

    files = sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED)
    if not files:
        print("ERROR: No supported images found.")
        return 1

    print("=" * 64)
    print("TINAYU CAMERA / FILTER ROBUSTNESS EXPERIMENT")
    print("=" * 64)
    print("Production engine is NOT modified.")
    print(f"Images: {len(files)}")
    print()

    results = []
    failures = []

    for i, path in enumerate(files, 1):
        try:
            image = cv2.imread(str(path))
            if image is None:
                raise ValueError("Could not decode image.")

            base_result = analyze_image(encode_image(image))
            if not base_result.get("success", False):
                raise ValueError(base_result.get("error", "Baseline analysis failed."))

            base = profile(base_result)
            variants = make_variants(image)
            variant_results = {}

            for name, variant in variants.items():
                result = analyze_image(encode_image(variant))
                if not result.get("success", False):
                    variant_results[name] = {"success": False, "error": result.get("error", "Analysis failed.")}
                    continue

                vp = profile(result)
                variant_results[name] = {
                    "success": True,
                    "profile": vp,
                    "comparison": compare(base, vp),
                }

            results.append({"file": path.name, "baseline": base, "variants": variant_results})

            changed = sum(
                1 for v in variant_results.values()
                if v.get("success") and v["comparison"]["changed"]
            )
            print(f"[{i}/{len(files)}] {path.name} | {base['temperature']} {base['season']} | changed_variants={changed}")

        except Exception as e:
            failures.append({"file": path.name, "error": str(e)})
            print(f"[{i}/{len(files)}] {path.name} | ERROR: {e}")

    names = [
        "brightness_up", "brightness_down", "contrast_up", "contrast_down",
        "saturation_up", "saturation_down", "compression"
    ]

    summary = {}
    for name in names:
        tested = changed = 0
        deltas = []

        for item in results:
            v = item["variants"].get(name)
            if not v or not v.get("success"):
                continue
            tested += 1
            if v["comparison"]["changed"]:
                changed += 1
            deltas.append(abs(v["comparison"]["temperature_strength_delta"]))

        summary[name] = {
            "tested": tested,
            "profile_changes": changed,
            "profile_change_rate": round(changed / tested, 4) if tested else None,
            "mean_abs_temperature_strength_delta": round(float(np.mean(deltas)), 4) if deltas else None,
        }

    report = {
        "experiment": "Camera / filter robustness",
        "engine_modified": False,
        "image_count": len(files),
        "successful_baselines": len(results),
        "failed_baselines": len(failures),
        "perturbations": names,
        "summary": summary,
        "results": results,
        "failures": failures,
    }

    output = Path("validation_samples") / "camera_filter_robustness_report.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print()
    print("=" * 64)
    print("CAMERA / FILTER ROBUSTNESS SUMMARY")
    print("=" * 64)
    for name, data in summary.items():
        print(f"{name:18} | tested={data['tested']:3} | changes={data['profile_changes']:3} | rate={data['profile_change_rate']}")

    print()
    print(f"Saved report: {output}")
    print("EXPERIMENT COMPLETE")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
