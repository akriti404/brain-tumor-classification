# Parameter-Efficient Hybrid Quantum-Classical Brain Tumor MRI Classification

Research-grade, reproducible codebase for:
**"A Parameter-Efficient and Explainable Hybrid Quantum Machine Learning Framework
for Multi-Class Brain Tumor Classification from MRI Images"**

This is the **core-pieces milestone**: data pipeline, classical baselines, and the
proposed hybrid CNN+VQC model. Noise/NISQ evaluation, cross-dataset generalization,
full ablation sweeps, explainability, and statistical-significance testing are
scaffolded in the architecture but not yet implemented — see
["What's implemented vs. scaffolded"](#whats-implemented-vs-scaffolded).

## Read this first: sandbox limitations and research integrity

This code was built and smoke-tested in a sandboxed environment with:
- **No GPU** (`torch.cuda.is_available() == False`, verified) — everything ran on CPU.
- **No access to Kaggle, Mendeley, or figshare**, where the real public brain-tumor
  MRI datasets are hosted. The sandbox's network allowlist covers package registries
  (PyPI, npm, crates) and github.com code, not data-hosting sites. Every public
  brain-tumor-MRI GitHub repo I could inspect ships *code*, not the actual images
  (the images are downloaded from Kaggle at runtime) — confirmed by cloning several
  and checking their contents.
- **No access to `download.pytorch.org`**, so torchvision's pretrained ImageNet
  weights (`pretrained: true` in the config) could not be downloaded here either
  (verified: HTTP 403). `configs/config.yaml` ships with `pretrained: false` for
  that reason. **On real hardware with normal internet access, set it back to
  `true`** — the architecture is designed around transfer learning from a
  pretrained MobileNetV2/ResNet18 backbone, per the spec.

Because of this, **all numbers currently in `results/` were produced by training on
a small procedurally-generated synthetic placeholder dataset** (see
`data/synthetic_data_generator.py`), for a handful of epochs, purely to prove the
pipeline runs correctly end-to-end (data loading → splitting → augmentation →
baseline models → the actual PennyLane quantum circuit → training loop → metrics →
results table → figures). **These numbers are not a scientific finding, do not
reflect real MRI data, and must not be cited or reported as results in your paper.**
This follows the spec's own research-integrity requirement (Section 16): don't
fabricate or imply results that weren't actually produced under the claimed
conditions.

**To get real results:** download a real dataset (e.g. Kaggle's
`masoudnickparvar/brain-tumor-mri-dataset`, 4 classes: glioma, meningioma,
pituitary, notumor — the one most of the reviewed 2026 papers use) onto a machine
with GPU + full internet access, arrange it as `data/raw/<class_name>/*.jpg`,
set `pretrained: true`, raise `training.epochs`, and everything else runs
unmodified.

## Architecture

```
MRI Image
  -> Preprocessing (resize, normalize, augment)
  -> Lightweight classical feature extractor (MobileNetV2 or ResNet18, pretrained)
  -> Dimensionality reduction (MLP -> tanh, output width = n_qubits)
  -> Quantum feature encoding (angle encoding, scaled to [-pi, pi])
  -> Parameter-efficient data-re-uploading VQC (configurable qubits/layers/entanglement)
  -> Classical classification head
  -> Multi-class prediction
```

### What's novel here (vs. a generic CNN+VQC)

The quantum component (`models/quantum.py`) combines three choices specifically to
target the "parameter efficiency" and "limited qubit/depth analysis" gaps your
literature review identified (Slide 6 of `Review_1_final.pptx`):

1. **Reduced-qubit representation** — the classical dimensionality-reduction head
   compresses the backbone's 1280-d (MobileNetV2) or 512-d (ResNet18) feature vector
   down to exactly `n_qubits` values, so qubit count is a free experimental knob
   independent of backbone choice.
2. **Data re-uploading** — features are re-encoded at every variational layer
   (not just once at the start), which the re-uploading literature shows lets a
   small number of qubits approximate more complex decision boundaries than a
   single-encoding circuit of the same width — directly relevant to keeping qubit
   count low (2/4/6/8 as required) without sacrificing expressivity.
3. **Configurable entanglement topology** (circular / linear / full) — so the
   qubit-count vs. circuit-depth vs. accuracy trade-off study the spec asks for
   (Section 6) can isolate the entanglement strategy as its own variable.

This is a combination, not a copy of any single reviewed paper — it's closest in
spirit to Rahman & Kim (2025, parameter-efficient QML) and Rahman et al.'s
QuReBrain (2026, data re-uploading), but neither paper couples re-uploading with a
qubit-count-independent classical bottleneck the way this does. This claim should
be checked against the full papers (only abstracts/summaries were available in
your review deck) before it goes in a thesis — I have not read the original PDFs.

## Repository layout

```
configs/config.yaml       All hyperparameters — nothing is hard-coded in the scripts.
data/
  dataset_inspector.py     Auto-discovers classes/counts/corrupted/duplicate images.
  dataset.py                Patient-level (falls back to stratified) split, transforms,
                             augmentation, WeightedRandomSampler for class imbalance.
  synthetic_data_generator.py  SANDBOX-ONLY placeholder data generator (see above).
models/
  classical.py              SimpleCNN, ResNet18, MobileNetV2 baselines; the shared
                             feature-extractor + dim-reduction head; classical-only
                             ablation variant of the proposed architecture.
  quantum.py                 The parameter-efficient, data-re-uploading VQC (PennyLane).
  hybrid.py                  Assembles the full proposed CNN+VQC model.
utils/
  reproducibility.py        Seeding, device selection.
  param_count.py             Parameter/gate/depth accounting for the efficiency study.
  metrics.py                  Accuracy/precision/recall/F1/specificity/ROC-AUC/confusion matrix.
visualization/plots.py       Class distribution, sample images, training curves,
                               baseline comparison, accuracy-vs-parameters.
train.py                     Trains one model (baseline or hybrid) per the config.
evaluate.py                  Full test-set metrics + appends a row to the results table.
results/
  logs/          Per-run JSON: full training history + parameter report.
  tables/experiment_results.csv   Master experiment-tracking table (spec Section 15).
  figures/        Generated plots.
  checkpoints/    Saved model weights.
```

## Running it

```bash
pip install -r requirements.txt

# 1. Get data. Either point configs/config.yaml -> data.root at a real
#    ImageFolder-structured dataset, or let the synthetic fallback generate
#    a placeholder set automatically (data.synthetic_fallback: true).

# 2. Inspect the dataset (auto class discovery, corruption/duplicate check).
python -m data.dataset_inspector --root data/raw

# 3. Train + evaluate any model. Results append to results/tables/experiment_results.csv.
python evaluate.py --model simple_cnn
python evaluate.py --model resnet18
python evaluate.py --model mobilenet_v2
python evaluate.py --model classical_proposed
python evaluate.py --model hybrid

# 4. Generate figures.
python -m visualization.plots
```

Every hyperparameter (qubits, layers, entanglement, encoding, epochs, backbone
choice, etc.) lives in `configs/config.yaml` — sweep any of them by editing the
config or passing a different config file.

## Parameter efficiency (real numbers, from the smoke test)

From `results/tables/experiment_results.csv`, this is the actual parameter
accounting the pipeline produces (not accuracy — see the integrity note above —
but the parameter/qubit/depth counts themselves are correct regardless of dataset):

| Model | Total params | Quantum params | Qubits | Circuit depth |
|---|---|---|---|---|
| simple_cnn | 24,068 | 0 | 0 | 0 |
| resnet18 | 11,178,564 | 0 | 0 | 0 |
| mobilenet_v2 | 2,228,996 | 0 | 0 | 0 |
| classical_proposed | 2,265,340 | 0 | 0 | 0 |
| **hybrid (proposed)** | 2,244,520 | **8** | 4 | 4 |

Note the hybrid model's *quantum* subcircuit uses only 8 trainable parameters
(2 layers × 4 qubits) regardless of image resolution or backbone width — this is
the parameter-efficiency claim the spec asks to substantiate (Section 6). The
total parameter count is currently dominated by the unfrozen MobileNetV2 backbone;
setting `classical_backbone.freeze_backbone: true` would isolate the quantum
component's contribution more cleanly for the ablation study.

## What's implemented vs. scaffolded

**Implemented and smoke-tested (this milestone):**
- Automatic dataset inspection (classes, counts, corrupted/duplicate images)
- Patient-level (or stratified fallback) train/val/test splitting, augmentation,
  class-imbalance handling
- 4 classical baselines + the classical-only ablation variant
- The proposed hybrid CNN+VQC model with a real, working PennyLane circuit
  (verified: batched forward pass and gradient backprop both function correctly)
- Full metrics suite (accuracy, precision, recall, F1 macro/weighted, specificity,
  ROC-AUC OvR, confusion matrix)
- Parameter/qubit/gate/depth accounting
- Experiment-tracking CSV, training-curve/comparison/parameter-efficiency figures

**Scaffolded (config fields and structure exist) but not yet built:**
- Noise/NISQ simulator comparison (Section 8) — `quantum.device_name` can point at
  a noisy PennyLane device, but the noise-sweep experiment script isn't written yet
- Cross-dataset generalization (Section 9) — needs a second dataset
- Full ablation sweep across qubit counts/depths/encodings (Section 11) — the model
  and config support this per-run; a sweep-orchestration script isn't written yet
- Multi-seed statistical aggregation + significance testing (Sections 10, 12) —
  `utils/metrics.py::aggregate_over_seeds` exists; the multi-seed runner doesn't yet
- Explainability: Grad-CAM on the classical backbone, expectation-value/ablation
  analysis on the quantum component (Section 7)
- Remaining visualizations (confusion matrix, ROC curves, noise-robustness,
  ablation, Grad-CAM) (Section 13)

Say the word and I'll build out any of these next.
