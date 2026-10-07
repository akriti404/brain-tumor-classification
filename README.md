# Brain Tumor Classification — Hybrid Quantum-Classical MRI Framework

A controlled comparative framework for evaluating **classical (CNN)**, **graph-based (GNN)**, and a novel **cross-attention CNN+GNN fusion** representation within a hybrid quantum-classical pipeline for brain MRI tumor classification, using a shared **Variational Quantum Classifier (VQC)** implemented in PennyLane.

```
                    ┌── CNN ──────────────────────────┐
                    │                                  │
MRI → Preprocessing ┤                                  ├→ Cross-Attention Fusion → VQC → Classifier
                    │                                  │
                    └── Superpixels → GNN ─────────────┘
```

Unlike treating CNN-VQC and GNN-VQC as two independent, non-interacting pipelines (the common pattern in prior hybrid-QML literature), this project's core contribution is letting the two representations exchange information **before** the quantum layer, via learned cross-attention — the graph's superpixel-node embeddings attend over the CNN's spatial feature map, and a gated fusion mechanism learns how much to trust each source per sample. The VQC itself also uses a multi-observable (X/Y/Z) readout rather than single-basis measurement, extracting 3x more classical information from the same circuit at no extra qubit or depth cost.

---

## Current results

Test set (891 held-out images, patient-level split), seed 42 — all models below trained and evaluated under the same pipeline, directly comparable:

| Model | Accuracy | F1-macro | Notes |
|---|---|---|---|
| `gnn_hybrid` (GNN → VQC, standalone) | ~40% | ~31% | weak — from-scratch GCN lacks a pretrained backbone's signal; kept as a baseline for the fusion ablation, excluded from the ensemble |
| `classical_proposed` (CNN → classical head, no VQC) | 92.4% | 92.1% | strongest single model |
| `hybrid` (CNN → VQC) | 91.1% | 90.9% | |
| **`fusion_hybrid` (CNN+GNN cross-attention → multi-observable VQC)** | **91.9%** | **90.6%** | novel architecture, single model |
| `classical_proposed` + TTA | 93.2% | 92.9% | horizontal-flip test-time augmentation |
| `fusion_hybrid` + TTA | 93.0% | 92.9% | TTA rebuilds the SLIC graph on the flipped image, not just the flipped CNN input |
| **Ensemble: `classical_proposed` + `fusion_hybrid` (TTA'd, soft-vote)** | **93.4%** | **93.2%** | **best result** |

**Honest assessment**: the fusion architecture measurably outperforms the plain (non-fused) CNN-VQC baseline, and ensembling it with the strongest classical ablation pushes further still — but it does not yet beat the classical-only (no-VQC) baseline as a standalone model. Error analysis (Grad-CAM) identified a specific, data-provenance-linked bottleneck behind most of the remaining error — not a vague "needs more tuning" situation. See [Known limitations](#known-limitations) below.

---

## Architecture

### Shared VQC

All three representation branches (CNN, GNN, fusion) feed a shared quantum layer (`models/quantum.py`) that:
1. Scales reduced classical features by π
2. Applies RY angle encoding with data re-uploading at every variational layer
3. Uses circular/linear/full entanglement (configurable)
4. Measures either single-basis (`readout: "z"`, `n_qubits` outputs) or multi-basis (`readout: "xyz"`, `3·n_qubits` outputs — used by the fusion model) expectation values

### CNN branch (`models/hybrid.py`)
MobileNetV2 or ResNet18 backbone → linear reduction to `n_qubits`, `tanh`-bounded → VQC → classifier head that sees `concat(reduced_features, quantum_output)` (residual path).

### GNN branch (`models/gnn_hybrid.py`)
SLIC superpixel segmentation (`models/graph.py`) → region-adjacency graph → 2-layer GCN with mean/max pooling → same reduction/VQC/residual-classifier pattern as the CNN branch.

### Fusion branch (`models/fusion.py`) — the novel contribution
- MobileNetV2's *spatial* feature map (not pooled) provides CNN tokens.
- A from-scratch GCN provides *per-node* (per-superpixel) embeddings, not pooled.
- **Cross-attention**: each sample's graph nodes (queries) attend over that *same* sample's CNN spatial tokens (keys/values), computed per-sample to respect batch boundaries exactly.
- **Gated fusion**: a learned sigmoid gate blends the CNN-global vector with the attention-pooled graph vector, rather than fixed concatenation.
- Feeds the shared VQC with multi-observable (`"xyz"`) readout.

---

## Repository layout

```
data/
  raw/                           # MRI images (not committed; see Dataset setup)
  dataset.py                     # Preprocessing, splits, transforms, CropToContent (margin-removal, off by default)
  fusion_dataset.py              # (image, graph, label) dataloaders for the fusion branch
  dataset_inspector.py           # Dataset sanity-check / stats CLI
  synthetic_data_generator.py    # Placeholder data generator (sandboxed/offline testing only)
models/
  classical.py                   # Simple CNN, ResNet18, MobileNetV2, classical-only proposed head
  quantum.py                     # Shared PennyLane VQC (single- or multi-observable readout)
  hybrid.py                      # CNN → VQC branch
  gnn_hybrid.py                  # GNN → VQC branch
  graph.py                       # SLIC superpixel graph construction
  fusion.py                      # Cross-attention CNN+GNN → VQC branch (novel contribution)
utils/                           # Seeding, metrics, parameter accounting
experiments/
  ensemble.py                    # Soft-vote ensemble over trained checkpoints
  tta_evaluate.py                # Horizontal-flip test-time augmentation
  ensemble_from_tta.py           # Combines saved TTA predictions (instant, no retraining)
  explainability_errors.py       # Error-targeted Grad-CAM (glioma<->meningioma misclassifications)
  explainability_cnn.py          # Grad-CAM, general sampling
  explainability_gnn.py          # GNNExplainer-based GNN explanations
  resource_ablations.py          # Qubit/layer/re-uploading ablation sweep
  noise_experiments.py           # NISQ noise-robustness sweep
  multi_seed_runner.py           # Multi-seed statistical validation
  cross_dataset.py               # Cross-dataset generalization harness
  statistical_analysis.py        # Hypothesis testing / confidence intervals / effect sizes
tests/                           # pytest unit tests (see Testing below)
visualization/plots.py           # Figure generation from results
train.py                         # Training entry point (--model, --representation)
evaluate.py                      # Evaluation entry point
configs/config.yaml              # Single master config
requirements.txt
```

> **Note on noise experiments**: `noise_experiments.py` requires `device_name: "default.mixed"` in `configs/config.yaml` for non-ideal noise runs — `lightning.qubit` (the default, fast backend) does not support PennyLane noise channels. Switch the config before running that script.

---

## Setup

### 1. Environment
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1      # Windows PowerShell; use .venv/bin/activate on Linux/Mac
python -m pip install -r requirements.txt
python -m pip install pennylane-lightning   # required for the fast quantum backend (~3x speedup vs default.qubit)
```

> **Quantum backend note**: the default backend is `lightning.qubit` with `adjoint` differentiation (~3.3x faster than `default.qubit`). It is a state-vector simulator and does **not** support PennyLane noise channels. For noise experiments, set `quantum.device_name: "default.mixed"` in `configs/config.yaml` before running `experiments/noise_experiments.py`.

### 2. Dataset
Point `configs/config.yaml`'s `data.root` at an ImageFolder-layout directory (`root/<class_name>/*.jpg`), four classes: `glioma`, `meningioma`, `notumor`, `pituitary`. The Kaggle **Brain Tumor MRI Dataset** (`masoudnickparvar/brain-tumor-mri-dataset`) is what this project was developed and evaluated against:
```powershell
python -m data.download_kaggle_dataset --out data/raw
```
If `data/raw` is empty and `data.synthetic_fallback: true` is set, a synthetic placeholder dataset is generated automatically — useful for smoke-testing the pipeline, **not** for any reported result.

**Dataset provenance note**: this dataset merges three source datasets (figshare, SARTAJ, Br35H) with inconsistent native image resolutions and framing conventions. This was found, via Grad-CAM error analysis, to contribute to the project's main remaining error source (glioma↔meningioma confusion) — see [Known limitations](#known-limitations).

### 3. Verify the dataset
```powershell
python -m data.dataset_inspector --root data/raw
```

### 4. Run the tests
```powershell
python -m pytest tests/ -v
```

---

## Reproducing results

```powershell
# Train the three models used in the best (ensemble) result
python train.py --model classical_proposed --config configs/config.yaml
python train.py --model hybrid --representation cnn --config configs/config.yaml
python train.py --model hybrid --representation fusion --config configs/config.yaml

# Evaluate individually
python evaluate.py --model classical_proposed --config configs/config.yaml
python evaluate.py --model hybrid --representation cnn --config configs/config.yaml
python evaluate.py --model hybrid --representation fusion --config configs/config.yaml

# Test-time augmentation
python experiments/tta_evaluate.py --model classical_proposed --representation cnn --config configs/config.yaml
python experiments/tta_evaluate.py --model hybrid --representation fusion --config configs/config.yaml

# Ensemble the TTA'd predictions (instant)
python experiments/ensemble_from_tta.py --config configs/config.yaml \
    --models classical_proposed hybrid --representations cnn fusion
```

GNN-only branch (weak standalone, kept for the fusion ablation comparison):
```powershell
python train.py --model hybrid --representation gnn --config configs/config.yaml
```

### Error analysis / explainability
```powershell
python experiments/explainability_errors.py --model classical_proposed --representation cnn --config configs/config.yaml
python experiments/explainability_errors.py --model hybrid --representation fusion --config configs/config.yaml
```
Finds and visualizes (Grad-CAM) every test-set misclassification within the glioma↔meningioma confusion pair specifically, rather than a random sample.

### Generate figures
```powershell
python -m visualization.plots
```

> Files inside `models/`, `data/`, and `utils/` are imported automatically — don't run them directly. Each `evaluate.py` call must follow its matching `train.py` call (it loads that model's checkpoint). All scripts that write to `results/tables/` now include a `git_commit` column (short SHA) so result rows from different code versions are distinguishable.

---

## Known limitations

- **Error is concentrated, not diffuse.** `notumor` and `pituitary` are both >98% recall; essentially all remaining error (best ensemble result, 93.4% overall) is the glioma↔meningioma confusion pair.
- **Grad-CAM analysis found this confusion is partly a dataset-provenance artifact**, not purely a modeling limitation — misclassified cases frequently show model attention on skull/scalp/background regions rather than brain tissue, consistent with the dataset's multi-source origin (different framing/resolution conventions per tumor class). A margin-removal preprocessing fix was implemented and tested (`data/dataset.py`'s `CropToContent`, toggled via `data.margin_crop`) but **regressed accuracy on every model tested** and is disabled by default — a negative result, kept in the codebase and documented rather than discarded.
- **The GNN-only branch is weak** (~40% accuracy) — a from-scratch 2-layer GCN on hand-crafted SLIC features doesn't carry much signal without the fusion architecture's access to CNN features.
- **The fusion model's novelty improves on the non-fused CNN-VQC baseline but not on the classical-only (no-VQC) ablation** as a standalone model — the quantum component's net contribution, isolated, is not yet a clear win; ensembling and TTA (classical techniques, not novel) account for most of the gain over the classical baseline in the current best result.

---

## Research framing

> A controlled comparative framework for evaluating classical, graph-based, and cross-attention-fused representations within hybrid quantum-classical brain MRI classification, with a multi-observable VQC readout, validated through test-time augmentation, ensembling, and error-targeted explainability analysis.

Planned/scaffolded but not yet executed at the time of writing: full qubit/layer/re-uploading ablation sweep, NISQ noise-robustness study, multi-seed statistical validation, and cross-dataset generalization testing — the scripts exist (`experiments/`) but need the fusion-representation support described above before running against the current best architecture.
