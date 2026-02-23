"""QGAN-HEP: Quantum Generative Adversarial Network for High Energy Physics.

A hybrid quantum-classical GAN for simulating LHC particle data using
PennyLane variational circuits and PyTorch classical networks.
"""

from qgan.generator import QuantumGenerator, generator_circuit
from qgan.discriminator import Discriminator
from qgan.losses import physics_aware_loss, mmd_loss
from qgan.train import train_qgan
from qgan.visualize import visualize_results

__version__ = "0.3.0"
__all__ = [
    "QuantumGenerator",
    "generator_circuit",
    "Discriminator",
    "physics_aware_loss",
    "mmd_loss",
    "train_qgan",
    "visualize_results",
]
