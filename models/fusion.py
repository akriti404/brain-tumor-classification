"""
Step 2: Cross-Attention CNN+GNN Fusion feeding a multi-observable VQC.

This is the novel-contribution model. Unlike models/hybrid.py (CNN-only ->
VQC) and models/gnn_hybrid.py (GNN-only -> VQC), which never share
information with each other, this model lets the two representations
actually talk before the quantum layer:

  1. CNN backbone (MobileNetV2.features) produces a SPATIAL feature map
     (not just a pooled vector) -- "what does each region look like".
  2. A GCN produces PER-NODE (per-superpixel) embeddings, not pooled --
     "what are the distinct regions, and how are they connected".
  3. Cross-attention: each graph node (query) attends over that SAME
     image's CNN spatial tokens (key/value), per-sample, respecting PyG's
     batch boundaries so node i of image A never attends to image B's
     feature map.
  4. Gated fusion combines the CNN's own global-pooled vector with the
     attention-pooled graph-informed vector via a learned sigmoid gate
     (not a fixed concat) -- the model can learn to weight CNN vs. GNN
     evidence per-sample.
  5. The fused embedding -> reducer -> VQC with X/Y/Z readout (models/
     quantum.py, readout="xyz") -> residual-concat classifier, same
     pattern established in Step 1 for the CNN and GNN branches.

Place this at: models/fusion.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from models.classical import DimensionalityReducer
from models.quantum import VariationalQuantumLayer


class GNNNodeEncoder(nn.Module):
    """
    Like models.gnn_hybrid.GNNEncoder but returns PER-NODE embeddings
    (no pooling) -- needed so cross-attention has a token per superpixel
    rather than a single pooled graph vector. Kept separate from
    gnn_hybrid.GNNEncoder so the already-validated (Step 1) GNN-only branch
    is never touched by this change.
    """

    def __init__(self, in_dim: int, hidden_dim: int = 32, num_layers: int = 2):
        super().__init__()
        from torch_geometric.nn import GCNConv
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")
        self.convs = nn.ModuleList()
        for layer in range(num_layers):
            self.convs.append(GCNConv(in_dim if layer == 0 else hidden_dim, hidden_dim))
        self.activation = nn.ReLU()

    def forward(self, x, edge_index):
        for conv in self.convs:
            x = self.activation(conv(x, edge_index))
        return x  # (total_nodes_in_batch, hidden_dim)


class CrossAttentionFusion(nn.Module):
    """
    Per-sample cross-attention: graph node embeddings (query) attend over
    that sample's CNN spatial tokens (key/value). Operates per-sample
    (looped over the batch) so graph boundaries from PyG's `data.batch`
    are respected exactly -- no cross-sample leakage. Batch sizes here
    (~32) make the loop cheap relative to the CNN/GNN/VQC compute anyway.
    """

    def __init__(self, cnn_channels: int, gnn_hidden_dim: int, d_model: int = 64, n_heads: int = 4):
        super().__init__()
        self.d_model = d_model
        self.cnn_proj = nn.Linear(cnn_channels, d_model)
        self.node_proj = nn.Linear(gnn_hidden_dim, d_model)
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, cnn_feature_map: torch.Tensor, node_embeddings: torch.Tensor, node_batch_idx: torch.Tensor):
        """
        cnn_feature_map: (B, C, H, W)
        node_embeddings: (total_nodes, gnn_hidden_dim) -- all graphs in the batch, concatenated
        node_batch_idx:  (total_nodes,) -- which sample (0..B-1) each node belongs to (PyG's data.batch)
        Returns: (B, d_model) -- mean+max pooled, attention-refined, per-sample graph vector -> (B, 2*d_model)
        """
        B, C, H, W = cnn_feature_map.shape
        tokens = cnn_feature_map.flatten(2).transpose(1, 2)  # (B, H*W, C)
        tokens = self.cnn_proj(tokens)                        # (B, H*W, d_model)

        pooled_per_sample = []
        for b in range(B):
            node_mask = node_batch_idx == b
            sample_nodes = node_embeddings[node_mask]          # (n_b, gnn_hidden_dim)
            if sample_nodes.shape[0] == 0:
                # Degenerate case (shouldn't happen with SLIC, but guard anyway):
                # fall back to a zero query so shapes stay consistent.
                pooled_per_sample.append(torch.zeros(2 * self.d_model, device=cnn_feature_map.device))
                continue
            query = self.node_proj(sample_nodes).unsqueeze(0)  # (1, n_b, d_model)
            kv = tokens[b:b + 1]                                # (1, H*W, d_model)
            attended, _ = self.attn(query, kv, kv)              # (1, n_b, d_model)
            attended = self.norm(attended.squeeze(0) + query.squeeze(0))  # residual + norm, (n_b, d_model)
            mean_pool = attended.mean(dim=0)
            max_pool = attended.max(dim=0).values
            pooled_per_sample.append(torch.cat([mean_pool, max_pool], dim=0))  # (2*d_model,)

        return torch.stack(pooled_per_sample, dim=0)  # (B, 2*d_model)


class GatedFusion(nn.Module):
    """Learned sigmoid gate blending CNN-global and graph-informed vectors, not a fixed concat."""

    def __init__(self, cnn_dim: int, graph_dim: int, d_fusion: int = 64):
        super().__init__()
        self.cnn_proj = nn.Linear(cnn_dim, d_fusion)
        self.graph_proj = nn.Linear(graph_dim, d_fusion)
        self.gate = nn.Sequential(nn.Linear(d_fusion * 2, d_fusion), nn.Sigmoid())

    def forward(self, cnn_vec: torch.Tensor, graph_vec: torch.Tensor) -> torch.Tensor:
        cnn_p = self.cnn_proj(cnn_vec)
        graph_p = self.graph_proj(graph_vec)
        g = self.gate(torch.cat([cnn_p, graph_p], dim=1))
        return g * cnn_p + (1 - g) * graph_p  # (B, d_fusion)


class CNNGNNFusionVQC(nn.Module):
    def __init__(self, n_classes: int, backbone_arch: str, pretrained: bool,
                 node_feature_dim: int, gnn_hidden_dim: int, gnn_num_layers: int,
                 d_model: int, n_attention_heads: int, d_fusion: int,
                 n_qubits: int, n_layers: int, entanglement: str, data_reuploading: bool,
                 diff_method: str, device_name: str, readout: str = "xyz",
                 noise_type: str = "ideal", noise_prob: float = 0.0):
        super().__init__()
        if backbone_arch != "mobilenet_v2":
            raise ValueError("CNNGNNFusionVQC currently only supports backbone_arch='mobilenet_v2' "
                              "(fixed 1280-channel spatial feature map assumption)")
        weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        self.cnn_backbone = models.mobilenet_v2(weights=weights).features  # -> (B, 1280, H, W)
        self.cnn_channels = 1280
        self.cnn_pool = nn.AdaptiveAvgPool2d(1)

        self.gnn_encoder = GNNNodeEncoder(node_feature_dim, gnn_hidden_dim, gnn_num_layers)
        self.cross_attention = CrossAttentionFusion(self.cnn_channels, gnn_hidden_dim, d_model, n_attention_heads)
        self.fusion = GatedFusion(self.cnn_channels, d_model * 2, d_fusion)

        self.reducer = DimensionalityReducer(d_fusion, n_qubits)
        self.qlayer = VariationalQuantumLayer(
            n_qubits=n_qubits, n_layers=n_layers, entanglement=entanglement,
            data_reuploading=data_reuploading, diff_method=diff_method, device_name=device_name,
            noise_type=noise_type, noise_prob=noise_prob, readout=readout,
        )
        classifier_in = n_qubits + self.qlayer.output_dim  # reduced (residual) + quantum readout
        classifier_dim = max(classifier_in, 8)
        self.classifier = nn.Sequential(
            nn.Linear(classifier_in, classifier_dim), nn.ReLU(), nn.Linear(classifier_dim, n_classes)
        )

    def set_backbone_trainable(self, flag: bool):
        for p in self.cnn_backbone.parameters():
            p.requires_grad = flag

    def forward(self, images: torch.Tensor, graph_batch):
        cnn_map = self.cnn_backbone(images)                          # (B, 1280, H, W)
        cnn_global = torch.flatten(self.cnn_pool(cnn_map), 1)        # (B, 1280)

        node_embeddings = self.gnn_encoder(graph_batch.x, graph_batch.edge_index)  # (total_nodes, hidden)
        graph_vec = self.cross_attention(cnn_map, node_embeddings, graph_batch.batch)  # (B, 2*d_model)

        fused = self.fusion(cnn_global, graph_vec)                   # (B, d_fusion)
        reduced = self.reducer(fused)                                 # (B, n_qubits), tanh-bounded
        q_out = self.qlayer(reduced)                                  # (B, qlayer.output_dim)
        return self.classifier(torch.cat((reduced, q_out), dim=1))

    def get_intermediate(self, images, graph_batch):
        cnn_map = self.cnn_backbone(images)
        cnn_global = torch.flatten(self.cnn_pool(cnn_map), 1)
        node_embeddings = self.gnn_encoder(graph_batch.x, graph_batch.edge_index)
        graph_vec = self.cross_attention(cnn_map, node_embeddings, graph_batch.batch)
        fused = self.fusion(cnn_global, graph_vec)
        reduced = self.reducer(fused)
        q_out = self.qlayer(reduced)
        logits = self.classifier(torch.cat((reduced, q_out), dim=1))
        return reduced, q_out, logits


def build_fusion_model_from_config(cfg: dict, n_classes: int) -> CNNGNNFusionVQC:
    cb = cfg["classical_backbone"]
    graph_cfg = cfg.get("graph", {})
    q = cfg["quantum"]
    fusion_cfg = cfg.get("fusion", {})
    return CNNGNNFusionVQC(
        n_classes=n_classes,
        backbone_arch=cb["architecture"],
        pretrained=cb["pretrained"],
        node_feature_dim=7,
        gnn_hidden_dim=graph_cfg.get("hidden_dim", 32),
        gnn_num_layers=graph_cfg.get("num_layers", 2),
        d_model=fusion_cfg.get("d_model", 64),
        n_attention_heads=fusion_cfg.get("n_attention_heads", 4),
        d_fusion=fusion_cfg.get("d_fusion", 64),
        n_qubits=q["n_qubits"],
        n_layers=q["n_layers"],
        entanglement=q["entanglement"],
        data_reuploading=q["data_reuploading"],
        diff_method=q["diff_method"],
        device_name=q["device_name"],
        readout=fusion_cfg.get("readout", "xyz"),
        noise_type=q.get("noise_type", "ideal"),
        noise_prob=q.get("noise_prob", 0.0),
    )
