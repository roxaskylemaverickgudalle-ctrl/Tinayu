# Temperature Boundary Isolation Findings

## Purpose

This experiment isolates whether Tinayu's temperature and season instability under image perturbations is primarily caused by samples near the temperature decision boundaries.

The production engine was **not modified** during this experiment.

## Baseline

The benchmark used the FairFace evaluation image set.

The baseline temperature bands were:

- Cool: 16
- Boundary: 16
- Warm-edge: 1
- Strong-warm: 54

Only images successfully processed by the existing face and landmark pipeline were included in perturbation comparisons.

## Perturbation Results

| Perturbation | Mean Temperature Delta | Temperature Flip | Season Flip | Boundary Rate |
|---|---:|---:|---:|---:|
| Brightness up | 0.0159 | 4.6% | 13.8% | 5.8% |
| Brightness down | 0.0125 | 3.5% | 12.6% | 3.5% |
| Contrast up | 0.0215 | 6.9% | 11.5% | 9.2% |
| Contrast down | 0.0242 | 4.6% | 12.6% | 6.9% |
| Saturation up | 0.0397 | 10.3% | 18.4% | 11.5% |
| Saturation down | 0.0427 | 8.1% | 20.7% | 12.6% |
| JPEG compression | 0.0051 | 2.3% | 3.5% | 2.3% |

## Findings

The results show that saturation perturbations produced the largest temperature movement and the highest season-change rates in this benchmark.

JPEG compression produced comparatively small changes.

The existence of boundary samples explains some instability, but the perturbation results do not justify rewriting the production temperature thresholds by themselves.

## Engineering Decision

Keep the current production temperature and season logic unchanged.

Specifically:

1. Do not rewrite the temperature boundaries based on this benchmark alone.
2. Do not replace the production skin-region extraction based on this benchmark.
3. Treat boundary cases as an uncertainty/robustness concern rather than automatically forcing them into another temperature category.
4. Use larger validation sets before introducing production changes.

## Relationship to Camera/Filter Testing

The earlier camera/filter robustness benchmark showed that saturation and exposure-related changes can affect profiles more than JPEG compression.

This temperature-boundary experiment provides a more focused result: **saturation is the strongest tested perturbation for temperature movement in the current validation set**.

These experiments support keeping the deterministic production classifier stable while collecting more evidence for future robustness improvements.
