"""
Proposed model: MRI -> lightweight classical feature extractor ->
dimensionality reduction -> quantum feature encoding + parameter-efficient
data-re-uploading VQC -> classical classification head -> prediction.
"""
import torch
import torch.nn as nn

from models.classical import LightweightFeatureExtractor, DimensionalityReducer
from models.quantum import VariationalQuantumLayer


class HybridQCNN(nn.Module):
    def __init__(self, n_classes: int, backbone_arch: str, pretrained: bool, freeze_backbone: bool,
                 n_qubits: int, n_layers: int, entanglement: str, data_reuploading: bool,
                 diff_method: str, device_name: str, noise_type: str = "ideal", noise_prob: float = 0.0):
        super().__init__()
        self.extractor = LightweightFeatureExtractor(backbone_arch, pretrained, freeze_backbone)
        self.reducer = DimensionalityReducer(self.extractor.out_dim, n_qubits)
        self.qlayer = VariationalQuantumLayer(
            n_qubits=n_qubits, n_layers=n_layers, entanglement=entanglement,
            data_reuploading=data_reuploading, diff_method=diff_method, device_name=device_name,
            noise_type=noise_type, noise_prob=noise_prob,
        )
        self.classifier = nn.Sequential(
            nn.Linear(n_qubits * 2, max(n_qubits * 2, 8)),
            nn.ReLU(),
            nn.Linear(max(n_qubits * 2, 8), n_classes),
        )

    def forward(self, x):
        feats = self.extractor(x)
        reduced = self.reducer(feats)          # (B, n_qubits), bounded in [-1, 1]
        q_out = self.qlayer(reduced)            # (B, n_qubits), expectation values
        return self.classifier(torch.cat((reduced, q_out), dim=1))

    def get_intermediate(self, x):
        """Returns (reduced_features, quantum_expectations, logits) for analysis/explainability."""
        feats = self.extractor(x)
        reduced = self.reducer(feats)
        q_out = self.qlayer(reduced)
        logits = self.classifier(torch.cat((reduced, q_out), dim=1))
        return reduced, q_out, logits


def build_model_from_config(cfg: dict, n_classes: int) -> HybridQCNN:
    cb = cfg["classical_backbone"]
    q = cfg["quantum"]
    return HybridQCNN(
        n_classes=n_classes,
        backbone_arch=cb["architecture"],
        pretrained=cb["pretrained"],
        freeze_backbone=cb["freeze_backbone"],
        n_qubits=q["n_qubits"],
        n_layers=q["n_layers"],
        entanglement=q["entanglement"],
        data_reuploading=q["data_reuploading"],
        diff_method=q["diff_method"],
        device_name=q["device_name"],
        noise_type=q.get("noise_type", "ideal"),
        noise_prob=q.get("noise_prob", 0.0),
    )
