import pytest
import torch

from models.classical import build_baseline, ClassicalProposedHead
from models.hybrid import HybridQCNN
from train import get_backbone_and_head_params, set_requires_grad


def _assert_full_disjoint_coverage(model, backbone_params, head_params):
    all_ids = {id(p) for p in model.parameters()}
    backbone_ids = {id(p) for p in backbone_params}
    head_ids = {id(p) for p in head_params}

    assert backbone_ids.isdisjoint(head_ids), "backbone/head param sets overlap"
    assert backbone_ids | head_ids == all_ids, (
        f"param split doesn't cover all params: "
        f"missing={len(all_ids - (backbone_ids | head_ids))}, "
        f"extra={len((backbone_ids | head_ids) - all_ids)}"
    )


@pytest.mark.parametrize("model_name", ["resnet18", "mobilenet_v2"])
def test_baseline_backbone_split(model_name):
    model = build_baseline(model_name, n_classes=4, pretrained=False)
    split = get_backbone_and_head_params(model, model_name, representation="cnn")
    assert split is not None
    backbone_params, head_params, _ = split
    assert len(backbone_params) > 0
    assert len(head_params) > 0
    _assert_full_disjoint_coverage(model, backbone_params, head_params)


def test_hybrid_backbone_split():
    model = HybridQCNN(
        n_classes=4, backbone_arch="mobilenet_v2", pretrained=False, freeze_backbone=False,
        n_qubits=4, n_layers=2, entanglement="circular", data_reuploading=True,
        diff_method="backprop", device_name="default.qubit",
    )
    split = get_backbone_and_head_params(model, "hybrid", representation="cnn")
    assert split is not None
    backbone_params, head_params, _ = split
    assert len(backbone_params) > 0
    assert len(head_params) > 0  # reducer + qlayer + classifier
    _assert_full_disjoint_coverage(model, backbone_params, head_params)


def test_classical_proposed_backbone_split():
    model = ClassicalProposedHead(
        n_classes=4, backbone_arch="mobilenet_v2", pretrained=False,
        freeze_backbone=False, reduced_dim=8,
    )
    split = get_backbone_and_head_params(model, "classical_proposed", representation="cnn")
    assert split is not None
    backbone_params, head_params, _ = split
    _assert_full_disjoint_coverage(model, backbone_params, head_params)


def test_simple_cnn_and_gnn_have_no_split():
    model = build_baseline("simple_cnn", n_classes=4, pretrained=False)
    assert get_backbone_and_head_params(model, "simple_cnn", representation="cnn") is None
    # representation="gnn" always returns None regardless of model_name/model
    assert get_backbone_and_head_params(model, "hybrid", representation="gnn") is None


def test_freeze_unfreeze_toggles_requires_grad():
    model = build_baseline("mobilenet_v2", n_classes=4, pretrained=False)
    backbone_params, head_params, _ = get_backbone_and_head_params(model, "mobilenet_v2", representation="cnn")

    set_requires_grad(backbone_params, False)
    assert all(not p.requires_grad for p in backbone_params)
    assert all(p.requires_grad for p in head_params)  # untouched

    set_requires_grad(backbone_params, True)
    assert all(p.requires_grad for p in backbone_params)
