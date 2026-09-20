import torch
from torch_geometric.data import Data, Batch

from models.fusion import CNNGNNFusionVQC
from train import get_backbone_and_head_params


def _make_fake_graph(n_nodes: int, node_feature_dim: int = 7):
    x = torch.randn(n_nodes, node_feature_dim)
    # simple ring adjacency so every graph has at least some edges
    src = torch.arange(n_nodes)
    dst = torch.roll(src, shifts=-1)
    edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])], dim=0)
    return Data(x=x, edge_index=edge_index)


def _make_fusion_model(n_qubits=2, n_layers=1, readout="xyz", n_classes=4):
    return CNNGNNFusionVQC(
        n_classes=n_classes, backbone_arch="mobilenet_v2", pretrained=False,
        node_feature_dim=7, gnn_hidden_dim=8, gnn_num_layers=2,
        d_model=8, n_attention_heads=2, d_fusion=8,
        n_qubits=n_qubits, n_layers=n_layers, entanglement="circular",
        data_reuploading=True, diff_method="backprop", device_name="default.qubit",
        readout=readout,
    )


def test_forward_shape_uniform_node_counts():
    model = _make_fusion_model()
    images = torch.randn(3, 3, 64, 64)  # small spatial size keeps the test fast
    graphs = [_make_fake_graph(5) for _ in range(3)]
    graph_batch = Batch.from_data_list(graphs)

    logits = model(images, graph_batch)
    assert logits.shape == (3, 4)
    assert torch.isfinite(logits).all()


def test_forward_shape_variable_node_counts():
    """Different samples having different numbers of SLIC superpixels is the
    normal case (mri_to_graph's segment count varies per image) -- confirm
    the per-sample attention loop handles a ragged batch correctly."""
    model = _make_fusion_model()
    images = torch.randn(4, 3, 64, 64)
    graphs = [_make_fake_graph(n) for n in (3, 9, 5, 1)]  # deliberately uneven, incl. n=1
    graph_batch = Batch.from_data_list(graphs)

    logits = model(images, graph_batch)
    assert logits.shape == (4, 4)
    assert torch.isfinite(logits).all()


def test_readout_z_vs_xyz_output_dim_affects_classifier_input():
    model_z = _make_fusion_model(readout="z")
    model_xyz = _make_fusion_model(readout="xyz")
    assert model_z.qlayer.output_dim == model_z.qlayer.n_qubits
    assert model_xyz.qlayer.output_dim == model_xyz.qlayer.n_qubits * 3
    # classifier's first Linear in_features should reflect n_qubits + qlayer.output_dim
    z_in = model_z.classifier[0].in_features
    xyz_in = model_xyz.classifier[0].in_features
    assert xyz_in > z_in, "xyz readout should give the classifier a wider input than z-only"


def test_cross_attention_batch_isolation():
    """Two samples with very different node embeddings should NOT produce
    identical graph_vec outputs if attention were (incorrectly) mixing them
    with a shared feature map -- this is a smoke check, not a numerical
    guarantee, but catches a wired-wrong batch index immediately."""
    model = _make_fusion_model()
    images = torch.randn(2, 3, 64, 64)
    g1 = _make_fake_graph(4)
    g2 = _make_fake_graph(4)
    g1.x = torch.zeros_like(g1.x)
    g2.x = torch.ones_like(g2.x) * 5.0
    graph_batch = Batch.from_data_list([g1, g2])

    model.eval()
    with torch.no_grad():
        node_embeddings = model.gnn_encoder(graph_batch.x, graph_batch.edge_index)
        cnn_map = model.cnn_backbone(images)
        graph_vecs = model.cross_attention(cnn_map, node_embeddings, graph_batch.batch)

    assert graph_vecs.shape == (2, 2 * model.cross_attention.d_model)
    # With deliberately very different node features per sample, the two
    # pooled graph vectors should differ (not be numerically identical).
    assert not torch.allclose(graph_vecs[0], graph_vecs[1], atol=1e-4)


def test_fusion_backbone_head_split():
    model = _make_fusion_model()
    split = get_backbone_and_head_params(model, "hybrid", representation="fusion")
    assert split is not None
    backbone_params, head_params, _ = split
    assert len(backbone_params) > 0  # cnn_backbone (mobilenet_v2.features)
    assert len(head_params) > 0      # gnn_encoder + cross_attention + fusion + reducer + qlayer + classifier

    all_ids = {id(p) for p in model.parameters()}
    backbone_ids = {id(p) for p in backbone_params}
    head_ids = {id(p) for p in head_params}
    assert backbone_ids.isdisjoint(head_ids)
    assert backbone_ids | head_ids == all_ids
