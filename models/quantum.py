"""
Quantum component of the proposed hybrid model.

Step 2 addition: configurable multi-observable readout. The circuit itself
(encoding + trainable rotations + entanglement) is UNCHANGED from Step 1 —
only what we measure at the end changes. Previously every wire was measured
in a single basis (PauliZ), giving (B, n_qubits) expectation values. With
`readout="xyz"`, each wire is measured in all three Pauli bases, giving
(B, 3*n_qubits) values from the exact same circuit evaluation (no extra
qubits, no extra depth, no extra trainable parameters) — the motivation
being that a single Z-expectation per qubit is a lossy summary of that
qubit's full Bloch-sphere state; X/Y/Z together recover much more of it.

`readout="z"` (the default) reproduces Step 1's behavior exactly, so
models/hybrid.py and models/gnn_hybrid.py are unaffected unless you opt in.
"""
import pennylane as qml
import torch
import torch.nn as nn

# Noise channel imports
try:
    from pennylane.ops import BitFlip, PhaseFlip, DepolarizingChannel
    NOISE_AVAILABLE = True
except ImportError:
    NOISE_AVAILABLE = False

VALID_READOUTS = ("z", "xyz")


def build_qnode(n_qubits: int, n_layers: int, entanglement: str, data_reuploading: bool,
                 diff_method: str = "backprop", device_name: str = "default.qubit",
                 noise_type: str = "ideal", noise_prob: float = 0.0, readout: str = "z"):
    if readout not in VALID_READOUTS:
        raise ValueError(f"Unknown readout '{readout}', expected one of {VALID_READOUTS}")

    # Configure device with noise model if specified
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

    def apply_noise(wire):
        """Apply noise channel based on noise type."""
        if not NOISE_AVAILABLE or noise_type == "ideal" or noise_prob <= 0:
            return

        if noise_type == "bit_flip":
            BitFlip(noise_prob, wires=wire)
        elif noise_type == "phase_flip":
            PhaseFlip(noise_prob, wires=wire)
        elif noise_type == "depolarizing":
            DepolarizingChannel(noise_prob, wires=wire)
        else:
            raise ValueError(f"Unknown noise type '{noise_type}'")

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
                apply_noise(w)

        for layer in range(n_layers):
            if data_reuploading:
                for w in wires:
                    qml.RY(inputs[..., w], wires=w)
                    apply_noise(w)
            for w in wires:
                qml.RZ(weights[layer, w], wires=w)
                apply_noise(w)
            entangle(wires)
            # Apply noise after entanglement
            for w in wires:
                apply_noise(w)

        if readout == "z":
            return [qml.expval(qml.PauliZ(w)) for w in wires]
        # readout == "xyz": measure every wire in all three Pauli bases.
        # Order is [X_0..X_{n-1}, Y_0..Y_{n-1}, Z_0..Z_{n-1}] so output
        # reshapes cleanly to (B, 3, n_qubits) if ever needed downstream.
        return (
            [qml.expval(qml.PauliX(w)) for w in wires]
            + [qml.expval(qml.PauliY(w)) for w in wires]
            + [qml.expval(qml.PauliZ(w)) for w in wires]
        )

    weight_shapes = {"weights": (n_layers, n_qubits)}
    return circuit, weight_shapes


class VariationalQuantumLayer(nn.Module):
    """
    Thin nn.Module wrapper around a PennyLane TorchLayer implementing the
    parameter-efficient, data-re-uploading VQC described in the module
    docstring. Input width == n_qubits. Output width == n_qubits (readout="z",
    Step 1 default) or 3*n_qubits (readout="xyz", Step 2 opt-in).
    """

    def __init__(self, n_qubits: int, n_layers: int, entanglement: str = "circular",
                 data_reuploading: bool = True, diff_method: str = "backprop",
                 device_name: str = "default.qubit", noise_type: str = "ideal",
                 noise_prob: float = 0.0, readout: str = "z"):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.entanglement = entanglement
        self.data_reuploading = data_reuploading
        self.noise_type = noise_type
        self.noise_prob = noise_prob
        self.readout = readout

        circuit, weight_shapes = build_qnode(
            n_qubits, n_layers, entanglement, data_reuploading, diff_method, device_name,
            noise_type, noise_prob, readout,
        )
        self.qlayer = qml.qnn.TorchLayer(circuit, weight_shapes)

    @property
    def output_dim(self) -> int:
        return self.n_qubits * 3 if self.readout == "xyz" else self.n_qubits

    def forward(self, x):
        # x: (B, n_qubits) already bounded (e.g. via tanh) -> scale to [-pi, pi]
        x = x * torch.pi
        return self.qlayer(x)  # (B, output_dim) expectation values in [-1, 1]

    @property
    def quantum_parameters(self):
        return self.qlayer.weights
