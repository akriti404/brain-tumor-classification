"""Independent GNN -> reducer -> shared VQC classification branch."""
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_max_pool, global_mean_pool

from models.classical import DimensionalityReducer
from models.quantum import VariationalQuantumLayer


class GNNEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 32, num_layers: int = 2):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")
        self.convs = nn.ModuleList()
        for layer in range(num_layers):
            self.convs.append(GCNConv(in_dim if layer == 0 else hidden_dim, hidden_dim))
        self.activation = nn.ReLU()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for conv in self.convs:
            x = self.activation(conv(x, edge_index))
        return torch.cat([global_mean_pool(x, data.batch), global_max_pool(x, data.batch)], dim=1)


class HybridGNNVQC(nn.Module):
    def __init__(self, n_classes: int, node_feature_dim: int, hidden_dim: int,
                 num_layers: int, n_qubits: int, n_quantum_layers: int,
                 entanglement: str, data_reuploading: bool, diff_method: str, device_name: str):
        super().__init__()
        self.encoder = GNNEncoder(node_feature_dim, hidden_dim, num_layers)
        self.reducer = DimensionalityReducer(hidden_dim * 2, n_qubits)
        self.qlayer = VariationalQuantumLayer(
            n_qubits, n_quantum_layers, entanglement, data_reuploading, diff_method, device_name,
        )
        classifier_dim = max(n_qubits * 2, 8)
        self.classifier = nn.Sequential(nn.Linear(n_qubits, classifier_dim), nn.ReLU(), nn.Linear(classifier_dim, n_classes))

    def forward(self, data):
        embedding = self.encoder(data)
        reduced = self.reducer(embedding)
        return self.classifier(self.qlayer(reduced))


def build_gnn_model_from_config(cfg: dict, n_classes: int) -> HybridGNNVQC:
    graph_cfg = cfg.get("graph", {})
    q = cfg["quantum"]
    return HybridGNNVQC(
        n_classes=n_classes, node_feature_dim=7,
        hidden_dim=graph_cfg.get("hidden_dim", 32), num_layers=graph_cfg.get("num_layers", 2),
        n_qubits=q["n_qubits"], n_quantum_layers=q["n_layers"], entanglement=q["entanglement"],
        data_reuploading=q["data_reuploading"], diff_method=q["diff_method"], device_name=q["device_name"],
    )