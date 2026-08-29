# Member 3 Experimental Validation

This directory contains the experimental validation framework for systematically evaluating CNN-VQC vs GNN-VQC representations across multiple axes: quantum resources, NISQ noise, explainability, cross-dataset generalization, and statistical validation.

## Overview

The experimental framework is designed to provide rigorous statistical validation of the controlled comparative study between classical (CNN) and graph-based (GNN) representations within hybrid quantum-classical brain MRI classification.

## Directory Structure

```
experiments/
├── README.md                          # This file
├── multi_seed_runner.py               # Multi-seed validation framework
├── resource_ablations.py              # Quantum resource ablation experiments
├── noise_experiments.py               # NISQ noise model experiments
├── explainability_cnn.py              # CNN explainability (Grad-CAM)
├── explainability_gnn.py              # GNN explainability (GNNExplainer)
├── cross_dataset.py                  # Cross-dataset generalization
├── statistical_analysis.py           # Statistical analysis framework
└── run_member3_experiments.py        # Unified experiment orchestrator
```

## Usage

### Quick Start - Run All Experiments

```bash
# Run complete experimental suite
python experiments/run_member3_experiments.py --all

# Run specific phase
python experiments/run_member3_experiments.py --phase multi_seed
python experiments/run_member3_experiments.py --phase resource_ablations
python experiments/run_member3_experiments.py --phase noise
python experiments/run_member3_experiments.py --phase explainability
python experiments/run_member3_experiments.py --phase cross_dataset
python experiments/run_member3_experiments.py --phase statistical_analysis
```

### Individual Experiment Scripts

#### 1. Multi-Seed Validation

Run training and evaluation across multiple random seeds for statistical robustness.

```bash
python experiments/multi_seed_runner.py \
    --model hybrid \
    --representation cnn \
    --config configs/config.yaml \
    --seeds 42 1 7
```

**Output:**
- `results/logs/multi_seed_*.json` - Aggregated multi-seed results
- `results/tables/multi_seed_results.csv` - Statistical summary table

#### 2. Resource Ablations

Systematically evaluate impact of quantum resources (qubits, layers, re-uploading).

```bash
python experiments/resource_ablations.py \
    --config configs/config.yaml \
    --models hybrid \
    --representations cnn gnn \
    --qubits 2 4 6 \
    --layers 1 2 4 \
    --reuploading both \
    --seeds 42 1 7
```

**Output:**
- `results/logs/ablation_*.json` - Detailed ablation results
- `results/tables/resource_ablations.csv` - Ablation summary table

#### 3. Noise Experiments

Evaluate NISQ device noise robustness with different noise types and probabilities.

```bash
python experiments/noise_experiments.py \
    --config configs/config.yaml \
    --models hybrid \
    --representations cnn gnn \
    --noise_types ideal bit_flip phase_flip depolarizing \
    --noise_probs 0.0 0.01 0.05 0.1 \
    --seeds 42 1 7
```

**Output:**
- `results/logs/noise_*.json` - Detailed noise results
- `results/tables/noise_experiments.csv` - Noise summary table

#### 4. CNN Explainability

Generate Grad-CAM visualizations for CNN-VQC model predictions.

```bash
python experiments/explainability_cnn.py \
    --config configs/config.yaml \
    --model hybrid \
    --seed 42 \
    --num_samples 10 \
    --method gradcam \
    --output_dir results/explainability/cnn
```

**Output:**
- `results/explainability/cnn/exp_*.png` - Grad-CAM heatmaps
- `results/explainability/cnn/explainability_*.json` - Explanation metadata

#### 5. GNN Explainability

Generate GNNExplainer visualizations for GNN-VQC model predictions.

```bash
python experiments/explainability_gnn.py \
    --config configs/config.yaml \
    --model hybrid \
    --seed 42 \
    --num_samples 10 \
    --output_dir results/explainability/gnn
```

**Output:**
- `results/explainability/gnn/exp_*.json` - Graph explanations
- `results/explainability/gnn/feature_importance_analysis.json` - Feature analysis

#### 6. Cross-Dataset Generalization

Evaluate model robustness across different MRI datasets.

```bash
python experiments/cross_dataset.py \
    --config configs/config.yaml \
    --models hybrid \
    --representations cnn gnn \
    --dataset_pairs "data/source:data/target" \
    --seeds 42 1 7
```

**Output:**
- `results/logs/cross_dataset_*.json` - Cross-dataset results
- `results/tables/cross_dataset_results.csv` - Generalization summary

#### 7. Statistical Analysis

Perform comprehensive statistical analysis on experimental results.

```bash
python experiments/statistical_analysis.py \
    --config configs/config.yaml \
    --results_dir results \
    --output_dir results/statistical_analysis \
    --metric accuracy
```

**Output:**
- `results/statistical_analysis/statistical_analysis_report.json` - Complete statistical report

## Configuration

### Experimental Settings in `configs/config.yaml`

```yaml
# Quantum resource ablation settings
ablation_experiments:
  qubit_counts: [2, 4, 6]
  layer_counts: [1, 2, 4]
  reuploading_options: [true, false]
  models: ["hybrid"]
  representations: ["cnn", "gnn"]

# Noise experiment settings
noise_experiments:
  noise_types: ["ideal", "bit_flip", "phase_flip", "depolarizing"]
  noise_probabilities: [0.0, 0.01, 0.05, 0.1]
  models: ["hybrid"]
  representations: ["cnn", "gnn"]

# Explainability settings
explainability:
  cnn:
    method: "gradcam"
    num_samples: 10
  gnn:
    num_samples: 10
    epochs: 100
    lr: 0.01

# Cross-dataset settings
cross_dataset:
  dataset_pairs: []
  source_datasets: []
  target_datasets: []

# Statistical analysis settings
statistical_analysis:
  primary_metric: "accuracy"
  confidence_level: 0.95
  metrics_to_analyze: ["accuracy", "f1_macro", "precision_macro", "recall_macro"]
```

## Results Organization

```
results/
├── logs/
│   ├── multi_seed_*.json              # Multi-seed aggregated results
│   ├── ablation_*.json                # Resource ablation results
│   ├── noise_*.json                   # Noise experiment results
│   ├── cross_dataset_*.json           # Cross-dataset results
│   └── member3_experiments_*.json     # Orchestrator logs
├── tables/
│   ├── multi_seed_results.csv         # Multi-seed summary
│   ├── resource_ablations.csv         # Ablation summary
│   ├── noise_experiments.csv          # Noise summary
│   └── cross_dataset_results.csv      # Cross-dataset summary
├── figures/
│   ├── 06_resource_ablations.png      # Ablation visualizations
│   ├── 07_noise_experiments.png       # Noise visualizations
│   └── 08_statistical_analysis.png    # Statistical visualizations
├── explainability/
│   ├── cnn/
│   │   ├── exp_*.png                  # Grad-CAM heatmaps
│   │   └── explainability_*.json      # CNN explanations
│   └── gnn/
│       ├── exp_*.json                 # Graph explanations
│       └── feature_importance_analysis.json
└── statistical_analysis/
    └── statistical_analysis_report.json  # Complete statistical report
```

## Key Metrics Tracked

### Performance Metrics
- Accuracy, Precision, Recall, F1-score (macro/weighted)
- ROC-AUC, Specificity
- Training time, Inference time

### Resource Metrics
- Total parameters, Quantum parameters
- Qubit count, Circuit depth
- Parameter efficiency (accuracy per parameter)

### Statistical Metrics
- Mean ± Standard deviation across seeds
- 95% Confidence intervals
- p-values from hypothesis tests
- Effect sizes (Cohen's d)

### Noise Robustness Metrics
- Performance degradation per noise type
- Noise probability impact curves
- Robustness coefficients

### Explainability Metrics
- Feature importance rankings
- Activation localization precision
- Explanation consistency

### Generalization Metrics
- Domain shift impact
- Cross-dataset performance gaps
- Generalization coefficients

## Dependencies

All experimental scripts require the base project dependencies plus:

- `scipy>=1.11` - Statistical analysis functions
- `grad-cam>=1.5` - CNN explainability (already in requirements.txt)
- `torch-geometric>=2.3` - GNN explainability (already in requirements.txt)

## Computational Requirements

### Resource Ablations
- **Estimated time**: ~2-4 hours per model-representation pair
- **Parallelization**: Can run different ablation configurations in parallel

### Noise Experiments
- **Estimated time**: ~3-5 hours per model-representation pair
- **Note**: Noise simulations may be computationally expensive

### Explainability
- **Estimated time**: ~10-30 minutes per model
- **GPU**: Recommended for faster explainability generation

### Cross-Dataset
- **Estimated time**: Depends on dataset sizes
- **Storage**: Requires multiple datasets

## Troubleshooting

### Common Issues

1. **Checkpoint not found**
   - Ensure model is trained before running evaluation/explainability
   - Check checkpoint path in results directory

2. **Noise model errors**
   - Verify PennyLane version supports noise operations
   - Check noise probability values are in valid range [0, 1]

3. **Explainability failures**
   - Ensure target layers are found in model architecture
   - Check PyTorch Geometric version for GNNExplainer compatibility

4. **Memory issues**
   - Reduce batch size in config
   - Process fewer samples in explainability scripts
   - Use smaller datasets for initial testing
