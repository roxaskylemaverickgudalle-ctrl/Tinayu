import json
import sys
from pathlib import Path

import cv2
import numpy as np

from tinayu_engine import (
    decode_image,
    detect_landmarks,
    landmarks_to_pixels,
    polygon_mask,
    normalize_skin_profile,
    rgb_to_lab,
    temperature_score,
    classify_temperature_stable,
    classify_skin_category,
    classify_skin_saturation,
    suggest_season,
    analyze_image,
)

LEFT = [50, 101, 118, 119, 100, 47]
RIGHT = [280, 330, 347, 348, 329, 277]


def cheek_pixels(image_rgb, landmarks):
    pts = landmarks_to_pixels(landmarks, image_rgb.shape)
    masks = []
    for indices in (LEFT, RIGHT):
        masks.append(polygon_mask(image_rgb.shape, pts[indices]))
    mask = cv2.bitwise_or(masks[0], masks[1])
    return image_rgb[mask > 0]


def robust_rgb(pixels):
    if len(pixels) < 20:
        return None, 0, len(pixels)

    lab = np.array([rgb_to_lab(p) for p in pixels], dtype=float)
    med = np.median(lab, axis=0)
    mad = np.median(np.abs(lab - med), axis=0)
    scale = np.maximum(1.4826 * mad, 1.0)
    z = np.abs(lab - med) / scale
    keep = np.all(z <= 2.5, axis=1)

    kept = pixels[keep]
    if len(kept) < max(30, int(len(pixels) * 0.35)):
        keep = np.all(z <= 3.0, axis=1)
        kept = pixels[keep]

    if len(kept) == 0:
        return None, 0, len(pixels)

    return np.mean(kept, axis=0).astype(int).tolist(), len(kept), len(pixels)


def robust_profile(image_bytes):
    image_rgb = decode_image(image_bytes)
    landmarks = detect_landmarks(image_rgb)
    pixels = cheek_pixels(image_rgb, landmarks)
    skin_rgb, retained, total = robust_rgb(pixels)
    if skin_rgb is None:
        raise ValueError("Not enough robust cheek pixels")

    normalized = normalize_skin_profile(skin_rgb)
    lab = normalized["lab"]
    strength = temperature_score(lab[1], lab[2])
    temperature = classify_temperature_stable(lab[1], lab[2])
    depth = classify_skin_category(lab[0])
    saturation = classify_skin_saturation(normalized["chroma"])
    season = suggest_season(temperature, saturation, depth, strength)

    return {
        "temperature_strength": round(float(strength), 4),
        "temperature": temperature,
        "season": season,
        "retained_pixels": retained,
        "total_pixels": total,
        "retention": round(retained / total, 4) if total else 0,
    }


def main():
    folder = Path(sys.argv[1] if len(sys.argv) > 1 else "validation_samples")
    images = sorted(p for p in folder.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
    results = []

    print("=" * 60)
    print("TINAYU ROBUST SKIN STABILITY EVALUATION")
    print("=" * 60)
    print(f"Directory: {folder.resolve()}")
    print(f"Images:    {len(images)}\n")

    for i, path in enumerate(images, 1):
        try:
            data = path.read_bytes()
            current = analyze_image(data)
            robust = robust_profile(data)
            cur = current["profile"]["heuristics"]
            row = {
                "file": path.name,
                "current_strength": float(cur["temperature_strength"]),
                "robust_strength": robust["temperature_strength"],
                "current_temperature": cur["temperature"],
                "robust_temperature": robust["temperature"],
                "current_season": cur["suggested_season"],
                "robust_season": robust["season"],
                "retained_pixels": robust["retained_pixels"],
                "total_pixels": robust["total_pixels"],
                "retention": robust["retention"],
            }
            results.append(row)
            delta = row["robust_strength"] - row["current_strength"]
            changed = row["current_season"] != row["robust_season"]
            print(f"[{i}/{len(images)}] {path.name}")
            print(f"  current: {row['current_strength']:.3f} {row['current_temperature']} -> {row['current_season']}")
            print(f"  robust:  {row['robust_strength']:.3f} {row['robust_temperature']} -> {row['robust_season']}")
            print(f"  delta: {delta:+.3f} | retained: {row['retained_pixels']}/{row['total_pixels']} ({row['retention']:.0%})" + (" | SEASON CHANGED" if changed else ""))
        except Exception as e:
            print(f"[{i}/{len(images)}] {path.name} -> FAILED: {e}")

    if not results:
        print("\nNo successful analyses.")
        return

    current = np.array([r["current_strength"] for r in results])
    robust = np.array([r["robust_strength"] for r in results])
    season_changes = sum(r["current_season"] != r["robust_season"] for r in results)
    temp_changes = sum(r["current_temperature"] != r["robust_temperature"] for r in results)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Successful: {len(results)}/{len(images)}")
    print(f"Mean strength:   {current.mean():.3f} -> {robust.mean():.3f} ({(robust-current).mean():+.3f})")
    print(f"Median strength: {np.median(current):.3f} -> {np.median(robust):.3f}")
    print(f"Temperature changes: {temp_changes}/{len(results)}")
    print(f"Season changes:      {season_changes}/{len(results)}")
    print(f"Mean retention:     {np.mean([r['retention'] for r in results]):.1%}")

    report = {
        "successful": len(results),
        "total": len(images),
        "mean_current_strength": round(float(current.mean()), 4),
        "mean_robust_strength": round(float(robust.mean()), 4),
        "median_current_strength": round(float(np.median(current)), 4),
        "median_robust_strength": round(float(np.median(robust)), 4),
        "mean_strength_delta": round(float((robust-current).mean()), 4),
        "temperature_changes": temp_changes,
        "season_changes": season_changes,
        "mean_retention": round(float(np.mean([r["retention"] for r in results])), 4),
        "results": results,
    }
    out = folder / "robust_stability_report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
