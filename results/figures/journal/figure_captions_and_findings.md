# Journal figures: captions and interpretation

All model figures use seed-42 runs with a 50-epoch maximum and validation-accuracy early stopping. The primary comparison uses 1,298 unique, train-disjoint images from the separate official Kaggle test folder, evaluated identically for both models. Earlier runs with a 30-epoch maximum are not pooled into this comparison.

## Figures and findings

1. **Dataset split counts** (`fig01_dataset_split_counts`): the project training dataset has 4,006 train, 858 validation, and 859 internal holdout images. The separate official test folder contains 1,430 files; the audited evaluation cohort contains 1,298. Class-wise counts are in `results/epochs50/tables/journal_dataset_split_counts.csv`.
2. **Training curves** (`fig02_training_curves_seed42`): best validation accuracy was 94.9% for classical and 94.1% for hybrid. They completed 50 and 37 epochs, respectively. The curves show optimization progress, not test-set superiority.
3. **Primary performance** (`fig03_primary_comparison`): on the matched cohort, classical accuracy/macro F1 were 93.0%/92.9%; hybrid accuracy/macro F1 were 93.5%/93.4%. Only seed 42 was retrained under the 50-epoch protocol, so across-seed uncertainty is not estimable and is not shown.
4. **Row-normalized confusion matrices** (`fig04_confusion_matrices_seed42`): rows show recall within the true class; each cell reports count and row percentage. Meningioma-to-glioma errors were 14/314 (4.5%) for classical and 18/314 (5.7%) for hybrid. These are single-seed descriptive results.
5. **Per-class metrics** (`fig05_per_class_metrics_seed42`): compare precision, recall, and F1 across all four classes to reveal class-specific trade-offs, including glioma/meningioma confusions that aggregate metrics can conceal.
6. **One-vs-rest ROC curves** (`fig06_roc_curves_seed42`): curves show class-wise discrimination over thresholds on the audited cohort. Macro OvR AUC was 0.988 for classical and 0.987 for hybrid. Hybrid class-wise AUCs (glioma, meningioma, no-tumor, pituitary) were glioma: 0.987; meningioma: 0.970; notumor: 0.995; pituitary: 0.994; AUC does not replace thresholded metrics or calibration analysis.
7. **Efficiency comparison** (`fig07_efficiency_seed42`): parameter counts were 2,265,340 for classical and 2,265,492 for hybrid. Training took 47.96 and 40.16 minutes; evaluation averaged 4.82 and 5.18 ms/image, respectively. These are single-run measurements; evaluation time includes test data loading/transforms.
8. **Architecture schematic** (`fig08_architecture_comparison`): both models share the MobileNetV2 feature extractor and reducer. The hybrid path adds a VQC; its measured expectations are combined with reduced classical features before classification.
9. **Paired explainability panels**: four separate class-specific images use identical audited test images for both models, with up to four cases per class: high-confidence hybrid-correct examples and one error example. They use the epoch-50 checkpoints and prioritize glioma/meningioma confusions. The dataset has no tumor masks or boxes, so these are qualitative illustrations rather than validated lesion localization; hybrid Grad-CAM reflects CNN-backbone gradients, not the VQC's internal computation.

## Statistical and reporting cautions

- Each model has one seed under the 50-epoch protocol. Figure 3 therefore has no across-seed standard deviation or confidence interval; do not infer statistical significance.
- Metric confidence intervals in `results/epochs50/tables/journal_performance_metrics_bootstrap_ci.csv` are 95% percentile bootstrap intervals from 10,000 test-image resamples stratified by true class. They quantify finite-test-cohort sampling uncertainty conditional on each trained seed-42 model; they do not capture training-seed variability.
- The raw official test folder contains 1,430 files; the primary comparison uses the 1,298-image audited, unique, train-disjoint cohort.
- The internal holdout documents the configured project split; reported model comparison uses the separate official Kaggle test cohort.
