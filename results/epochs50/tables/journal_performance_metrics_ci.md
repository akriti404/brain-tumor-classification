# Updated test-set performance metrics

Both models were trained with a 50-epoch maximum (seed 42; validation early stopping) and evaluated on the same audited official-test cohort of 1,298 train-disjoint, deduplicated images. Values are point estimate (95% CI), shown as percentages.

| Metric | Classical CNN | Hybrid CNN-VQC |
|---|---:|---:|
| Accuracy | 92.99% (91.60-94.30%) | 93.53% (92.22-94.76%) |
| Macro precision | 92.91% (91.50-94.24%) | 93.47% (92.16-94.76%) |
| Macro recall | 92.88% (91.46-94.19%) | 93.43% (92.10-94.69%) |
| Macro F1 | 92.87% (91.44-94.19%) | 93.44% (92.11-94.71%) |
| Weighted F1 | 92.96% (91.54-94.26%) | 93.51% (92.19-94.76%) |
| Macro OvR ROC AUC | 98.80% (98.38-99.15%) | 98.65% (98.22-99.04%) |

## Per-class precision, recall, and F1

| Class | Model | Precision (95% CI) | Recall (95% CI) | F1 (95% CI) | Support |
|---|---|---:|---:|---:|---:|
| Glioma | Classical CNN | 94.82% (92.33-97.08%) | 90.71% (87.31-93.81%) | 92.72% (90.54-94.70%) | 323 |
| Glioma | Hybrid CNN-VQC | 94.04% (91.61-96.45%) | 92.88% (90.09-95.67%) | 93.46% (91.53-95.34%) | 323 |
| Meningioma | Classical CNN | 87.10% (83.60-90.40%) | 85.99% (82.17-89.81%) | 86.54% (83.65-89.13%) | 314 |
| Meningioma | Hybrid CNN-VQC | 88.35% (84.98-91.70%) | 86.94% (83.12-90.45%) | 87.64% (85.02-90.18%) | 314 |
| Notumor | Classical CNN | 95.02% (92.62-97.16%) | 96.83% (94.92-98.73%) | 95.91% (94.23-97.35%) | 315 |
| Notumor | Hybrid CNN-VQC | 96.81% (94.94-98.40%) | 96.19% (93.97-98.10%) | 96.50% (95.02-97.78%) | 315 |
| Pituitary | Classical CNN | 94.69% (92.43-96.85%) | 97.98% (96.53-99.42%) | 96.31% (94.90-97.58%) | 346 |
| Pituitary | Hybrid CNN-VQC | 94.68% (92.41-96.84%) | 97.69% (95.95-99.13%) | 96.16% (94.72-97.55%) | 346 |

## Methods and reporting notes

- Confidence intervals are 95% percentile intervals from 5,000 class-stratified bootstrap resamples of test images, using seed 20261020. Each resample preserves the observed support for each class (glioma 323, meningioma 314, no-tumor 315, pituitary 346).
- These intervals quantify test-cohort sampling uncertainty conditional on the fitted seed-42 checkpoints. They do not capture training-seed variability, model-selection uncertainty, or external-dataset uncertainty.
- Per-class precision, recall, and F1 use one-vs-rest definitions; macro metrics are unweighted averages across the four classes. Weighted F1 weights class F1 by class support. ROC AUC is macro-averaged one-vs-rest AUC using predicted class probabilities.
- The models share exactly the same test images. The table provides descriptive model-specific intervals; overlapping/non-overlapping intervals are not a paired significance test. Avoid claiming statistical superiority from this single-seed comparison.
- The point estimates were checked against the saved epoch-50 evaluation JSON files before writing this report.

Machine-readable estimates and intervals: `journal_performance_metrics_ci.csv`.
