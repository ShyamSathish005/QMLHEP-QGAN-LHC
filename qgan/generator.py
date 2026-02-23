"""Quantum generator built on a PennyLane variational circuit."""

import numpy as np
import torch
import torch.nn as nn
import pennylane as qml

# ---------------------------------------------------------------------------
# Circuit hyper-parameters (importable so other modules stay in sync)
# ---------------------------------------------------------------------------
N_QUBITS = 4
N_LAYERS = 2

dev = qml.device("default.qubit", wires=N_QUBITS)


@qml.qnode(dev, interface="torch", diff_method="backprop")
def generator_circuit(inputs: torch.Tensor, weights: torch.Tensor):
    """Variational circuit: angle-embed latent noise then entangle."""
    qml.AngleEmbedding(inputs, wires=range(N_QUBITS))
    qml.StronglyEntanglingLayers(weights, wires=range(N_QUBITS))
    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]


class QuantumGenerator(nn.Module):
    """Hybrid quantum–classical generator producing 4-vectors (E, px, py, pz).

    Barren-plateau mitigation strategy
    -----------------------------------
    * Small-variance weight initialisation (σ = 0.1).
    * Hardware-efficient ansatz (StronglyEntanglingLayers).
    * Runtime gradient-norm monitoring.
    """

    def __init__(self, n_qubits: int = N_QUBITS, n_layers: int = N_LAYERS):
        super().__init__()
        self.n_qubits = n_qubits
        shape = qml.StronglyEntanglingLayers.shape(n_layers=n_layers, n_wires=n_qubits)
        self.q_weights = nn.Parameter(torch.randn(shape) * 0.1)
        self.linear = nn.Linear(n_qubits, 4)

        # Gradient diagnostics
        self.gradient_history: list[float] = []

    # ------------------------------------------------------------------
    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        q_out = torch.stack(
            [torch.stack(generator_circuit(x, self.q_weights)) for x in noise]
        ).float()

        out = self.linear(q_out)
        # Enforce E > 0 (soft, no in-place ops)
        energy = torch.abs(out[:, 0]) + 1e-3
        out = torch.cat([energy.unsqueeze(1), out[:, 1:]], dim=1)
        return out

    # ------------------------------------------------------------------
    def get_gradient_norm(self) -> float:
        """Compute and record the total L2 gradient norm (√Σ‖gᵢ‖²)."""
        sq_sum = 0.0
        for p in self.parameters():
            if p.grad is not None:
                sq_sum += p.grad.data.norm(2).item() ** 2
        total = sq_sum ** 0.5
        self.gradient_history.append(total)
        return total

    def detect_barren_plateau(self, threshold: float = 1e-6) -> bool:
        """Return True when recent gradient variance falls below *threshold*."""
        if len(self.gradient_history) < 2:
            return False
        recent = self.gradient_history[-10:]
        return float(np.var(recent)) < threshold
