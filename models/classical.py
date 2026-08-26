"""
Classical models: baselines (Simple CNN, ResNet18, MobileNetV2) plus the
lightweight feature extractor + dimensionality-reduction head shared by the
proposed hybrid architecture and its classical-only ablation variant.
"""
import torch
import torch.nn as nn
from torchvision import models


class SimpleCNN(nn.Module):
    """A small from-scratch CNN baseline — the floor everything else must beat."""

    def __init__(self, n_classes: int, in_channels: int = 3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(64, n_classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def build_resnet18(n_classes: int, pretrained: bool = True) -> nn.Module:
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, n_classes)
    return model


def build_mobilenet_v2(n_classes: int, pretrained: bool = True) -> nn.Module:
    weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v2(weights=weights)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, n_classes)
    return model


def build_baseline(name: str, n_classes: int, pretrained: bool = True) -> nn.Module:
    if name == "simple_cnn":
        return SimpleCNN(n_classes)
    if name == "resnet18":
        return build_resnet18(n_classes, pretrained)
    if name == "mobilenet_v2":
        return build_mobilenet_v2(n_classes, pretrained)
    raise ValueError(f"Unknown baseline architecture '{name}'")


class LightweightFeatureExtractor(nn.Module):
    """
    Wraps a pretrained MobileNetV2 (or ResNet18) as a frozen/fine-tunable
    feature extractor, exposing pooled features of dimension `feature_dim`.
    This is the "Classical lightweight feature extractor" stage of the
    proposed pipeline (MRI -> ... -> classical feature extractor -> ...).
    """

    def __init__(self, architecture: str = "mobilenet_v2", pretrained: bool = True, freeze: bool = False):
        super().__init__()
        if architecture == "mobilenet_v2":
            weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
            backbone = models.mobilenet_v2(weights=weights)
            self.features = backbone.features
            self.out_dim = 1280
        elif architecture == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            backbone = models.resnet18(weights=weights)
            self.features = nn.Sequential(*list(backbone.children())[:-2])
            self.out_dim = 512
        else:
            raise ValueError(f"Unsupported backbone '{architecture}'")

        self.pool = nn.AdaptiveAvgPool2d(1)
        if freeze:
            for p in self.features.parameters():
                p.requires_grad = False

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return torch.flatten(x, 1)  # (B, out_dim)


class DimensionalityReducer(nn.Module):
    """
    Reduces the classical feature vector down to `reduced_dim` == n_qubits,
    the input width the quantum encoding layer expects. A small MLP with a
    bounded output (tanh) so encoded values map cleanly onto rotation angles.
    """

    def __init__(self, in_dim: int, reduced_dim: int):
        super().__init__()
        hidden = max(reduced_dim * 4, 16)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, reduced_dim),
            nn.Tanh(),  # bound to [-1, 1] before scaling to rotation angles
        )

    def forward(self, x):
        return self.net(x)


class ClassicalProposedHead(nn.Module):
    """
    Classical-only counterpart of the proposed architecture: same backbone +
    dimensionality reduction, but the reduced features go straight to a
    classical classifier instead of the VQC. Used as ablation baseline E.g.
    'CNN + dimensionality reduction' (spec Section 11, item B) and as
    baseline #4 ('classical version of the proposed architecture').
    """

    def __init__(self, n_classes: int, backbone_arch: str, pretrained: bool,
                 freeze_backbone: bool, reduced_dim: int):
        super().__init__()
        self.extractor = LightweightFeatureExtractor(backbone_arch, pretrained, freeze_backbone)
        self.reducer = DimensionalityReducer(self.extractor.out_dim, reduced_dim)
        self.classifier = nn.Sequential(
            nn.Linear(reduced_dim, reduced_dim * 2),
            nn.ReLU(),
            nn.Linear(reduced_dim * 2, n_classes),
        )

    def forward(self, x):
        feats = self.extractor(x)
        reduced = self.reducer(feats)
        return self.classifier(reduced)
