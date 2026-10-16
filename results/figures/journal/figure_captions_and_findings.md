# Journal figures: captions and interpretation

Primary comparison uses the official Kaggle test set after excluding 122 exact train–test overlaps and 10 repeated test images (1,298 unique, train-disjoint images). The seed-42 model comparison uses per-image predictions filtered to this identical cohort.

## Figures and findings

1. **Dataset split counts** (`fig01_dataset_split_counts`): the Kaggle training directory contains 5,723 images, split into 4,006 training, 858 validation, and 859 internal holdout images. The separate official test folder contains 1,430 files; the audited evaluation cohort contains 1,298. Class counts reveal the test cohort is not perfectly balanced.
2. **Training curves** (`fig02_training_curves_seed42`): both seed-42 models reduce training and validation loss and improve accuracy. Best validation accuracy was 93.2% for the classical model and 93.1% for the hybrid. This shows optimization progress, not test-set superiority.
3. **Primary performance and seed variation** (`fig03_primary_comparison`): on the matched cohort, seed-42 accuracy/macro F1 were 90.7%/90.4% for classical and 91.4%/91.2% for hybrid. Across hybrid seeds, accuracy was 90.5% ± 0.9%, and macro F1 was 90.4% ± 0.9% (mean ± sample SD, n=3). Classical has only one seed; no classical uncertainty interval or significance claim is supported.
4. **Row-normalized confusion matrices** (`fig04_confusion_matrices_seed42`): rows show recall distribution by true class and annotations show counts and row percentages. Meningioma-to-glioma errors decreased from 31/314 (9.9%) for classical to 16/314 (5.1%) for hybrid; glioma-to-meningioma errors were similar (36/323 vs. 35/323). These are single-seed descriptive results.
5. **Per-class metrics** (`fig05_per_class_metrics_seed42`): hybrid raised meningioma recall from 78.7% to 83.8% and F1 from 81.1% to 83.9%, while no-tumor recall was lower (95.6% vs. 97.8%). This illustrates why the overall macro score should be interpreted alongside class-specific trade-offs.
6. **One-vs-rest ROC curves** (`fig06_roc_curves_seed42`): curves show class-wise discrimination over thresholds. Hybrid AUC was 0.986 for glioma, 0.959 for meningioma, 0.995 for no-tumor, and 0.993 for pituitary; all class curves indicate strong ranking performance, but AUC does not replace thresholded metrics or calibration.
7. **Efficiency comparison** (`fig07_efficiency_seed42`): parameter counts are nearly identical (2.265M each), while seed-42 hybrid accuracy is 91.4% vs. 90.7% for classical. Hybrid training took 26.74 min vs. 20.38 min; evaluation averaged 5.53 vs. 4.74 ms/image. Timings are single-run and include data loading/transforms during the test loop; report as machine-specific measurements, not general hardware performance.
8. **Architecture schematic** (`fig08_architecture_comparison`): both models share the MobileNetV2 feature extractor and reducer. The classical head is replaced in the hybrid path by a VQC, whose measured expectations are concatenated with the reduced classical features before classification.
9. **Explainability panel not generated**: existing Grad-CAM outputs are from other model/split runs, not paired explanations on this audited official test cohort. Reusing them here would imply a comparison the artifacts do not support. Generate fresh paired examples on this test cohort before including an explainability figure.

## Statistical and reporting cautions

- Hybrid error bars in Figure 3 are sample SD across three training seeds, not a confidence interval. Classical currently has one seed.
- The historical 1,430-image results include exact train/test duplicates. The primary figures use the 1,298-image audited cohort instead.
- Do not claim statistically significant superiority from these figures. Complete classical runs for seeds 1 and 7, then perform a paired analysis using retained per-image predictions.
- The internal holdout is shown to document the configured train/validation/test split, but the primary model metrics use the separate official Kaggle test folder.
