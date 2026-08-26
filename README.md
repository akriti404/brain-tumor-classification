# Brain Tumor Classification — Hybrid Quantum-Classical MRI Framework

A controlled comparative framework for evaluating **classical (CNN)** and **graph-based (GNN)**
representations within a hybrid quantum-classical pipeline for brain MRI tumor classification,
using a shared **Variational Quantum Classifier (VQC)** implemented in PennyLane.

This is **not** a single novel CNN-GNN-VQC architecture. It is two independent, directly
comparable pipelines — CNN-VQC and GNN-VQC — that share the same quantum classifier and
evaluation harness, enabling a controlled comparison of classical vs. graph-based representations
under identical quantum-resource and noise conditions.

```
                    ┌── CNN → Dimensional Reduction ──┐
MRI → Preprocessing ┤                                 ├→ VQC → Classifier
                    └── Superpixels → GNN → Reduction ─┘
```

---

## Project status

| Component | Status |
|---|---|
| Dataset pipeline & preprocessing | ✅ Implemented |
| Classical baselines (Simple CNN, ResNet18, MobileNetV2, proposed classical model) | ✅ Implemented |
| CNN → VQC hybrid branch | ✅ Implemented |
| GNN → VQC hybrid branch | ✅ Implemented |
| Shared PennyLane VQC (RY encoding, data re-uploading) | ✅ Implemented, shared unmodified across both branches |
| Training / evaluation harness (seeds, splits, metrics, logging) | ✅ Implemented, consistent across CNN and GNN branches |
| Visualization / figure generation | ✅ Implemented |
| Qubit / layer / re-uploading ablations | ⬜ Not yet run |
| Noise experiments (bit-flip, phase-flip, depolarizing) | ⬜ Not yet run |
| Multi-seed statistical validation | ⬜ Not yet run |
| Explainability (Grad-CAM / GNNExplainer) | ⬜ Not yet run |
| Cross-dataset generalization | ⬜ Not yet run |

---

## Architecture

### Shared CNN-to-VQC / GNN-to-VQC contract

Both representation branches are required to produce output matching this exact interface before
entering the shared quantum layer:

- Output shape: `(B, n_qubits)`
- Value range: bounded to `[-1, 1]` via `tanh`
- `n_qubits` is read from shared config (default `4`) — changing it affects both branches identically

The shared VQC (`quantum.py`) then:
1. Scales inputs by `π`
2. Applies RY angle encoding
3. Re-uploads features at every variational layer
4. Returns `(B, n_qubits)` expectation values to the shared classifier head (`hybrid.py`)

### CNN branch

- Input: `(B, 3, 96, 96)` RGB tensors, ImageNet-normalized (`dataset.py`)
- Backbone: MobileNetV2 (`(B, 1280)`) or ResNet18 (`(B, 512)`) (`classical.py`)
- Reducer: linear projection → `(B, n_qubits)`, `tanh`-bounded

### GNN branch

- Input: the same preprocessed `(B, 3, 96, 96)` tensors, de-normalized back to display-space RGB
- Graph construction (`mri_to_graph()`): SLIC superpixel segmentation → region-adjacency graph
  - Node features: mean intensity, texture/std stats, centroid `(x, y)`, region area
  - Edges: spatial adjacency between superpixels, optionally weighted by intensity similarity
- Encoder: 2–3 layer GCN with global mean/max pooling → graph-level embedding
- Reducer: independently-weighted linear projection (mirrors the CNN reducer's pattern) →
  `(B, n_qubits)`, `tanh`-bounded
- Feeds into the same, unmodified VQC

---

## Repository layout

```
data/
  raw/                          # MRI images (synthetic generator available for testing)
  synthetic_data_generator.py   # Generates placeholder data if raw/ is empty
  dataset_inspector.py          # Dataset sanity-check / stats CLI
dataset.py                      # Preprocessing, normalization, DataLoader
models/
  classical.py                  # Simple CNN, ResNet18, MobileNetV2, proposed classical model
  quantum.py                    # Shared PennyLane VQC (VariationalQuantumLayer)
  hybrid.py                     # CNN-VQC hybrid model, shared classifier head
  graph.py                      # mri_to_graph() — SLIC + region-adjacency graph construction
  gnn_hybrid.py                 # GNN encoder + reducer + GNN-VQC hybrid model
utils/                          # Shared helpers (seeding, metrics, logging, checkpoints)
visualization/
  plots.py                      # Figure generation from evaluation results
train.py                        # Training entry point (--model, --representation flags)
evaluate.py                     # Evaluation entry point
requirements.txt
```

---

## Setup

Run from the project root, in order.

### 1. Activate the environment
```powershell
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies
```powershell
pip install -r requirements.txt
```

### 3. Generate synthetic data (only if `data/raw` is empty)
```powershell
python data/synthetic_data_generator.py
```

### 4. Inspect the dataset
```powershell
python -m data.dataset_inspector --root data/raw
```

---

## Running experiments

### CNN-VQC branch
```powershell
python train.py --model hybrid --representation cnn
python evaluate.py --model hybrid --representation cnn
```

### GNN-VQC branch
```powershell
python train.py --model hybrid --representation gnn
python evaluate.py --model hybrid --representation gnn
```

### Classical baselines (optional)
```powershell
python train.py --model simple_cnn
python evaluate.py --model simple_cnn

python train.py --model resnet18
python evaluate.py --model resnet18

python train.py --model mobilenet_v2
python evaluate.py --model mobilenet_v2

python train.py --model classical_proposed
python evaluate.py --model classical_proposed
```

### Generate figures (after evaluations)
```powershell
python -m visualization.plots
```

> Files inside `models/`, `dataset.py`, and `utils/` are imported automatically — do not run them
> directly. Each `evaluate.py` call must follow its matching `train.py` call.

## Experimental axes (planned)

```
Representation
    ├── CNN
    └── GNN
          ↓
Quantum model
    ├── VQC
    └── Quantum Kernel (if feasible)
          ↓
Resource
    ├── 2 / 4 / 6 qubits
    ├── 1 / 2 / 4 layers
    └── Re-uploading ON / OFF
          ↓
Noise
    ├── Ideal
    ├── Bit-flip
    ├── Phase-flip
    └── Depolarizing
          ↓
Evaluation
    ├── Performance
    ├── Resource efficiency
    ├── Explainability
    └── Cross-dataset generalization
```

## Research framing

This project studies:

> A controlled comparative framework for evaluating classical and graph-based representations
> within hybrid quantum-classical brain MRI classification, while systematically quantifying the
> effects of quantum resources, NISQ noise, explainability, and cross-dataset generalization.

The current implementation (CNN-VQC + GNN-VQC on a shared VQC) is the **foundation** for this
study. The ablation, noise, explainability, and cross-dataset experiments listed above are the
next phase, not yet run.
