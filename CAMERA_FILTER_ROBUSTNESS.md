# Camera & Filter Robustness Benchmark

## Purpose

This benchmark evaluates how stable Tinayu's existing analysis pipeline is under mild camera, exposure, color, and compression changes.

The benchmark was run against the FairFace evaluation image set. The production engine was **not modified** during the experiment.

## Perturbations Tested

Each image was analyzed under these controlled variations:

- Brightness up (+15)
- Brightness down (-15)
- Contrast up (1.08×)
- Contrast down (0.92×)
- Saturation up (+10%)
- Saturation down (-10%)
- JPEG compression (quality 70)

The baseline analysis was compared with each perturbed version.

## Results

| Perturbation | Images Tested | Profile Changes |
|---|---:|---:|
| Brightness up | 87 | 27 (31.0%) |
| Brightness down | 87 | 21 (24.1%) |
| Contrast up | 87 | 21 (24.1%) |
| Contrast down | 87 | 24 (27.6%) |
| Saturation up | 87 | 27 (31.0%) |
| Saturation down | 87 | 31 (35.6%) |
| JPEG compression | 87 | 5 (5.7%) |

The baseline engine successfully analyzed 87 of the 100 benchmark images. The remaining 13 images failed the existing face/landmark analysis pipeline and were not included in perturbation comparisons.

## Interpretation

The benchmark shows that JPEG compression has relatively little effect on the resulting profile, while exposure, contrast, and especially saturation changes can cause larger profile differences.

This does **not** mean the production classifier should immediately be changed. The results are evidence that color/exposure conditions can affect analysis, but changing production thresholds or replacing the current skin-region extraction without stronger validation could introduce new errors.

## Decision

For this benchmark:

1. Keep the current production temperature and season logic unchanged.
2. Keep the current skin extraction and normalization pipeline unchanged.
3. Treat robustness improvements as a validation/research task rather than an immediate production change.
4. Use additional controlled benchmarks before introducing a new production robustness method.

The benchmark therefore serves as a regression/reference point for future Tinayu improvements.

## Related Validation

A follow-up temperature-boundary isolation experiment was also performed to determine whether perturbation instability was concentrated around temperature boundaries. That experiment found that saturation changes were the strongest source of temperature instability, while compression remained comparatively stable.

The production classifier was intentionally left unchanged after the experiment.
