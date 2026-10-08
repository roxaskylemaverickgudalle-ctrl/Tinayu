import json
import sys
from pathlib import Path

import cv2
import numpy as np

from tinayu_engine import analyze_image

SUPPORTED = {".jpg", ".jpeg", ".png", ".webp"}

PERTURBATIONS = [
    "brightness_up",
    "brightness_down",
    "contrast_up",
    "contrast_down",
    "saturation_up",
    "saturation_down",
    "compression",
]


def encode_image(image, quality=95):
    ok, data = cv2.imencode(
        ".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality]
    )
    if not ok:
        raise ValueError("Could not encode image.")
    return data.tobytes()


def make_variants(image):
    out = {
        "brightness_up": cv2.convertScaleAbs(
            image, alpha=1.0, beta=15
        ),
        "brightness_down": cv2.convertScaleAbs(
            image, alpha=1.0, beta=-15
        ),
        "contrast_up": cv2.convertScaleAbs(
            image, alpha=1.08, beta=0
        ),
        "contrast_down": cv2.convertScaleAbs(
            image, alpha=0.92, beta=0
        ),
    }

    hsv = cv2.cvtColor(
        image, cv2.COLOR_BGR2HSV
    ).astype(np.float32)

    hsv[:, :, 1] = np.clip(
        hsv[:, :, 1] * 1.10, 0, 255
    )
    out["saturation_up"] = cv2.cvtColor(
        hsv.astype(np.uint8),
        cv2.COLOR_HSV2BGR,
    )

    hsv = cv2.cvtColor(
        image, cv2.COLOR_BGR2HSV
    ).astype(np.float32)

    hsv[:, :, 1] = np.clip(
        hsv[:, :, 1] * 0.90, 0, 255
    )
    out["saturation_down"] = cv2.cvtColor(
        hsv.astype(np.uint8),
        cv2.COLOR_HSV2BGR,
    )

    ok, data = cv2.imencode(
        ".jpg",
        image,
        [cv2.IMWRITE_JPEG_QUALITY, 70],
    )

    if ok:
        decoded = cv2.imdecode(
            data,
            cv2.IMREAD_COLOR,
        )
        if decoded is not None:
            out["compression"] = decoded

    return out


def get_profile(result):
    heuristics = (
        result.get("profile", {})
        .get("heuristics", {})
    )

    strength = heuristics.get(
        "temperature_strength"
    )

    return {
        "temperature": heuristics.get(
            "temperature"
        ),
        "temperature_strength": (
            float(strength)
            if strength is not None
            else None
        ),
        "season": heuristics.get(
            "suggested_season"
        ),
        "skin_category": heuristics.get(
            "skin_category"
        ),
        "skin_saturation": heuristics.get(
            "skin_saturation"
        ),
        "skin_depth": heuristics.get(
            "skin_depth"
        ),
    }


def temperature_band(strength):
    if strength is None:
        return "UNKNOWN"

    if strength < 0.45:
        return "COOL"

    if strength <= 0.55:
        return "BOUNDARY"

    if strength < 0.58:
        return "WARM-EDGE"

    return "STRONG-WARM"


def analyze(image):
    result = analyze_image(
        encode_image(image)
    )

    if not result.get("success", False):
        return None, result.get(
            "error",
            "Analysis failed.",
        )

    return get_profile(result), None


def main():
    directory = Path(
        sys.argv[1]
        if len(sys.argv) > 1
        else "fairface_evaluation/images"
    )

    if not directory.exists():
        print(
            f"ERROR: Directory not found: "
            f"{directory}"
        )
        return 1

    files = sorted(
        p
        for p in directory.iterdir()
        if p.is_file()
        and p.suffix.lower() in SUPPORTED
    )

    if not files:
        print("ERROR: No supported images found.")
        return 1

    print("=" * 70)
    print("TINAYU TEMPERATURE BOUNDARY ISOLATION")
    print("=" * 70)
    print("Production engine is NOT modified.")
    print(f"Images: {len(files)}")
    print()

    results = []
    failures = []

    for index, path in enumerate(files, 1):
        try:
            image = cv2.imread(str(path))

            if image is None:
                raise ValueError(
                    "Could not decode image."
                )

            baseline, error = analyze(image)

            if baseline is None:
                failures.append({
                    "file": path.name,
                    "stage": "baseline",
                    "error": error,
                })

                print(
                    f"[{index}/{len(files)}] "
                    f"{path.name} | BASELINE ERROR"
                )
                continue

            item = {
                "file": path.name,
                "baseline": {
                    **baseline,
                    "band": temperature_band(
                        baseline[
                            "temperature_strength"
                        ]
                    ),
                },
                "variants": {},
            }

            for name, transformed in (
                make_variants(image).items()
            ):
                current, error = analyze(
                    transformed
                )

                if current is None:
                    item["variants"][name] = {
                        "success": False,
                        "error": error,
                    }
                    continue

                delta = (
                    current[
                        "temperature_strength"
                    ]
                    - baseline[
                        "temperature_strength"
                    ]
                )

                item["variants"][name] = {
                    "success": True,
                    "temperature": current[
                        "temperature"
                    ],
                    "temperature_strength": current[
                        "temperature_strength"
                    ],
                    "season": current["season"],
                    "band": temperature_band(
                        current[
                            "temperature_strength"
                        ]
                    ),
                    "temperature_strength_delta": round(
                        delta,
                        4,
                    ),
                    "absolute_strength_delta": round(
                        abs(delta),
                        4,
                    ),
                    "temperature_flip": (
                        current["temperature"]
                        != baseline["temperature"]
                    ),
                    "season_flip": (
                        current["season"]
                        != baseline["season"]
                    ),
                    "boundary_crossed": (
                        temperature_band(
                            current[
                                "temperature_strength"
                            ]
                        )
                        != temperature_band(
                            baseline[
                                "temperature_strength"
                            ]
                        )
                    ),
                }

            results.append(item)

            temp_flips = sum(
                value.get(
                    "temperature_flip",
                    False,
                )
                for value in item[
                    "variants"
                ].values()
            )

            season_flips = sum(
                value.get(
                    "season_flip",
                    False,
                )
                for value in item[
                    "variants"
                ].values()
            )

            print(
                f"[{index}/{len(files)}] "
                f"{path.name} | "
                f"{baseline['temperature']} "
                f"{baseline['temperature_strength']:.3f} "
                f"{temperature_band(baseline['temperature_strength'])} "
                f"| temp_flips={temp_flips} "
                f"season_flips={season_flips}"
            )

        except Exception as error:
            failures.append({
                "file": path.name,
                "stage": "processing",
                "error": str(error),
            })

            print(
                f"[{index}/{len(files)}] "
                f"{path.name} | ERROR: {error}"
            )

    summary = {}

    for name in PERTURBATIONS:
        rows = [
            item["variants"][name]
            for item in results
            if name in item["variants"]
            and item["variants"][name].get(
                "success"
            )
        ]

        if not rows:
            continue

        deltas = np.array(
            [
                row[
                    "temperature_strength_delta"
                ]
                for row in rows
            ],
            dtype=float,
        )

        absolute = np.abs(deltas)

        temp_flips = sum(
            row["temperature_flip"]
            for row in rows
        )

        season_flips = sum(
            row["season_flip"]
            for row in rows
        )

        boundary_crossings = sum(
            row["boundary_crossed"]
            for row in rows
        )

        summary[name] = {
            "tested": len(rows),
            "mean_delta": round(
                float(deltas.mean()),
                4,
            ),
            "mean_abs_delta": round(
                float(absolute.mean()),
                4,
            ),
            "median_abs_delta": round(
                float(np.median(absolute)),
                4,
            ),
            "max_abs_delta": round(
                float(absolute.max()),
                4,
            ),
            "temperature_flips": temp_flips,
            "temperature_flip_rate": round(
                temp_flips / len(rows),
                4,
            ),
            "season_flips": season_flips,
            "season_flip_rate": round(
                season_flips / len(rows),
                4,
            ),
            "boundary_crossings": boundary_crossings,
            "boundary_crossing_rate": round(
                boundary_crossings / len(rows),
                4,
            ),
        }

    baseline_bands = {}

    for item in results:
        current_band = item[
            "baseline"
        ]["band"]

        baseline_bands[current_band] = (
            baseline_bands.get(
                current_band,
                0,
            )
            + 1
        )

    sensitivity = []

    for item in results:
        absolute_deltas = []
        temp_flips = 0
        season_flips = 0

        for value in item[
            "variants"
        ].values():

            if not value.get("success"):
                continue

            absolute_deltas.append(
                value[
                    "absolute_strength_delta"
                ]
            )

            temp_flips += int(
                value["temperature_flip"]
            )

            season_flips += int(
                value["season_flip"]
            )

        if absolute_deltas:
            sensitivity.append({
                "file": item["file"],
                "baseline_strength": item[
                    "baseline"
                ]["temperature_strength"],
                "baseline_band": item[
                    "baseline"
                ]["band"],
                "mean_abs_delta": round(
                    float(
                        np.mean(
                            absolute_deltas
                        )
                    ),
                    4,
                ),
                "max_abs_delta": round(
                    float(
                        np.max(
                            absolute_deltas
                        )
                    ),
                    4,
                ),
                "temperature_flips": temp_flips,
                "season_flips": season_flips,
            })

    sensitivity.sort(
        key=lambda row: (
            row["temperature_flips"],
            row["mean_abs_delta"],
        ),
        reverse=True,
    )

    report = {
        "experiment": (
            "Temperature boundary isolation"
        ),
        "engine_modified": False,
        "image_count": len(files),
        "successful_baselines": len(
            results
        ),
        "failed_baselines": len([
            failure
            for failure in failures
            if failure["stage"]
            == "baseline"
        ]),
        "baseline_band_distribution": (
            baseline_bands
        ),
        "summary": summary,
        "most_sensitive_images": (
            sensitivity[:20]
        ),
        "results": results,
        "failures": failures,
    }

    output = (
        Path("validation_samples")
        / "temperature_boundary_isolation_report.json"
    )

    output.parent.mkdir(
        exist_ok=True
    )

    output.write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 70)
    print("TEMPERATURE BOUNDARY SUMMARY")
    print("=" * 70)

    print("Baseline bands:")

    for key in (
        "COOL",
        "BOUNDARY",
        "WARM-EDGE",
        "STRONG-WARM",
    ):
        print(
            f"  {key:12}: "
            f"{baseline_bands.get(key, 0)}"
        )

    print()

    for name, data in summary.items():
        print(
            f"{name:18} | "
            f"abs_delta="
            f"{data['mean_abs_delta']:.4f} | "
            f"temp_flip="
            f"{data['temperature_flip_rate']:.1%} | "
            f"season_flip="
            f"{data['season_flip_rate']:.1%} | "
            f"boundary="
            f"{data['boundary_crossing_rate']:.1%}"
        )

    print()
    print(f"Saved: {output}")
    print("EXPERIMENT COMPLETE")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
