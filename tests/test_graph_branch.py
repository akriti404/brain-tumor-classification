import torch
from torch_geometric.data import Batch

from models.graph import mri_to_graph
from models.gnn_hybrid import GNNEncoder


def test_mri_to_graph_has_standardized_features_and_edges():
    image = torch.zeros(3, 96, 96)
    image[:, 20:45, 20:45] = 1.0
    graph = mri_to_graph(image, n_segments=16, seed=42)

    assert graph.x.shape[1] == 7
    assert graph.edge_index.shape[0] == 2
    assert torch.isfinite(graph.x).all()
    assert torch.allclose(graph.x.mean(dim=0), torch.zeros(7), atol=1e-5)


def test_gnn_encoder_returns_graph_level_embedding():
    graphs = [mri_to_graph(torch.rand(3, 96, 96), n_segments=16, seed=seed) for seed in (1, 2)]
    batch = Batch.from_data_list(graphs)
    encoder = GNNEncoder(in_dim=7, hidden_dim=8, num_layers=2)

    output = encoder(batch)

    assert output.shape == (2, 16)
    assert torch.isfinite(output).all()