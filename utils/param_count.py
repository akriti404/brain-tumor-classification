"""
Parameter-accounting utilities.

These are used to populate the parameter-efficiency table (classical vs.
quantum parameter counts, qubits, circuit depth, gate counts) that the
research spec requires for every model.
"""
from dataclasses import dataclass, asdict
import torch


@dataclass
class ParamReport:
    model_name: str
    total_params: int
    trainable_params: int
    classical_params: int
    quantum_params: int
    n_qubits: int
    circuit_depth: int
    n_quantum_gates: int

    def as_dict(self):
        return asdict(self)


def count_torch_params(module: torch.nn.Module):
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


def count_quantum_gates(n_qubits: int, n_layers: int, entanglement: str, data_reuploading: bool) -> int:
    """
    Approximate gate count for the variational circuit used in models/quantum.py.

    Per layer: n_qubits rotation gates (RY) for the trainable block, plus
    n_qubits entangling CNOTs (circular/linear -> n_qubits gates; full -> full
    pairwise). Data re-uploading repeats the encoding gates (n_qubits RX/RY)
    once per layer instead of only at the start.
    """
    encode_gates_per_layer = n_qubits if data_reuploading else 0
    rotation_gates_per_layer = n_qubits
    if entanglement == "full":
        entangle_gates_per_layer = n_qubits * (n_qubits - 1) // 2
    else:  # circular or linear -> ~n_qubits CNOTs
        entangle_gates_per_layer = n_qubits

    per_layer = encode_gates_per_layer + rotation_gates_per_layer + entangle_gates_per_layer
    total = per_layer * n_layers
    if not data_reuploading:
        total += n_qubits  # single initial encoding block
    return total


def build_param_report(model_name: str, model: torch.nn.Module, quantum_param_tensor,
                        n_qubits: int, n_layers: int, entanglement: str,
                        data_reuploading: bool) -> ParamReport:
    total, trainable = count_torch_params(model)
    quantum_params = 0 if quantum_param_tensor is None else quantum_param_tensor.numel()
    classical_params = total - quantum_params
    gates = 0 if n_qubits == 0 else count_quantum_gates(n_qubits, n_layers, entanglement, data_reuploading)
    depth = n_layers * (2 if data_reuploading else 1)
    return ParamReport(
        model_name=model_name,
        total_params=total,
        trainable_params=trainable,
        classical_params=classical_params,
        quantum_params=quantum_params,
        n_qubits=n_qubits,
        circuit_depth=depth,
        n_quantum_gates=gates,
    )
