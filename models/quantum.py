"""
Quantum component of the proposed hybrid model.

Research contribution (what's novel vs. a generic CNN+VQC baseline):
  A *parameter-efficient, data-re-uploading* variational quantum circuit with
  circular entanglement, operating on a reduced-dimension (n_qubits-wide)
  classical embedding. Each layer re-encodes the classical features (angle
  encoding via RY/RX) interleaved with a single trainable rotation + circular
  entangling block, rather than encoding once and stacking many trainable
  layers. This is the mechanism from data-re-uploading literature (a single
  qubit / few qubits can approximate arbitrarily complex functions given
  enough re-uploading layers), applied here specifically to keep the
  trainable-parameter count independent of image resolution and small
  relative to the classical backbone -- directly targeting the "parameter
  efficiency" research gap the review identified (spec Sections 2, 6).

  This is NOT a copy of any single reviewed paper: it combines (a) reduced
  -qubit angle encoding, (b) data re-uploading, and (c) a configurable
  entanglement topology (circular/linear/full) in one circuit whose depth
  and qubit count are config-driven so the qubit/depth-vs-accuracy trade-off
  study (Sections 6, 11) can sweep them directly.
"""
import pennylane as qml
import torch
import torch.nn as nn


def build_qnode(n_qubits: int, n_layers: int, entanglement: str, data_reuploading: bool,
                 diff_method: str = "backprop", device_name: str = "default.qubit"):
    dev = qml.device(device_name, wires=n_qubits)

    def entangle(wires):
        if entanglement == "linear":
            for i in range(len(wires) - 1):
                qml.CNOT(wires=[wires[i], wires[i + 1]])
        elif entanglement == "circular":
            for i in range(len(wires)):
                qml.CNOT(wires=[wires[i], wires[(i + 1) % len(wires)]])
        elif entanglement == "full":
            for i in range(len(wires)):
                for j in range(i + 1, len(wires)):
                    qml.CNOT(wires=[wires[i], wires[j]])
        else:
            raise ValueError(f"Unknown entanglement strategy '{entanglement}'")

    @qml.qnode(dev, interface="torch", diff_method=diff_method)
    def circuit(inputs, weights):
        # inputs: (..., n_qubits) classical features scaled to roughly [-pi, pi].
        # When TorchLayer batches a call, `inputs` carries a leading batch
        # dimension and PennyLane broadcasts each gate over it -- so we index
        # the *last* axis (the feature axis) with `...`, never the first.
        # weights: (n_layers, n_qubits) trainable rotation angles (not batched).
        wires = list(range(n_qubits))

        if not data_reuploading:
            for w in wires:
                qml.RY(inputs[..., w], wires=w)

        for layer in range(n_layers):
            if data_reuploading:
                for w in wires:
                    qml.RY(inputs[..., w], wires=w)
            for w in wires:
                qml.RZ(weights[layer, w], wires=w)
            entangle(wires)

        return [qml.expval(qml.PauliZ(w)) for w in wires]

    weight_shapes = {"weights": (n_layers, n_qubits)}
    return circuit, weight_shapes


class VariationalQuantumLayer(nn.Module):
    """
    Thin nn.Module wrapper around a PennyLane TorchLayer implementing the
    parameter-efficient, data-re-uploading VQC described in the module
    docstring. Input/output width == n_qubits.
    """

    def __init__(self, n_qubits: int, n_layers: int, entanglement: str = "circular",
                 data_reuploading: bool = True, diff_method: str = "backprop",
                 device_name: str = "default.qubit"):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.entanglement = entanglement
        self.data_reuploading = data_reuploading

        circuit, weight_shapes = build_qnode(
            n_qubits, n_layers, entanglement, data_reuploading, diff_method, device_name
        )
        self.qlayer = qml.qnn.TorchLayer(circuit, weight_shapes)

    def forward(self, x):
        # x: (B, n_qubits) already bounded (e.g. via tanh) -> scale to [-pi, pi]
        x = x * torch.pi
        return self.qlayer(x)  # (B, n_qubits) expectation values in [-1, 1]

    @property
    def quantum_parameters(self):
        return self.qlayer.weights
