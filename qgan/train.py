"""QGAN training loop with real-time diagnostics."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from qgan.generator import QuantumGenerator, N_QUBITS
from qgan.discriminator import Discriminator
from qgan.losses import physics_aware_loss, mmd_loss


def _generate_real_data(n_samples: int = 1000, mass: float = 1.0) -> torch.Tensor:
    """Simulate Double-Higgs–like 4-vectors obeying E² = p² + m²."""
    p = torch.randn(n_samples, 3) * 2.0
    E = torch.sqrt(torch.sum(p**2, dim=1) + mass**2).unsqueeze(1)
    return torch.cat([E, p], dim=1)


def train_qgan(
    epochs: int = 300,
    batch_size: int = 16,
    lr: float = 0.01,
    target_mass: float = 1.0,
    lambda_phys: float = 0.5,
    lambda_mmd: float = 1.0,
    seed: int = 42,
    verbose: bool = True,
) -> tuple[QuantumGenerator, dict]:
    """Train the hybrid quantum GAN and return the generator + metrics.

    Parameters
    ----------
    epochs : int
        Number of training iterations.
    batch_size : int
        Mini-batch size.
    lr : float
        Learning rate for both Adam optimisers.
    target_mass : float
        On-shell mass target (normalised units).
    lambda_phys, lambda_mmd : float
        Weights for the physics and MMD loss components.
    seed : int
        Random seed for reproducibility.
    verbose : bool
        Print progress every 50 epochs.

    Returns
    -------
    generator : QuantumGenerator
    history : dict
        Keys: ``d_loss``, ``g_loss``, ``g_bce``, ``g_phys``, ``g_mmd``,
        ``grad_norms``, ``avg_fake_mass``, ``barren_plateau_detected``,
        ``mode_collapse_detected``.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    generator = QuantumGenerator()
    discriminator = Discriminator()

    opt_g = optim.Adam(generator.parameters(), lr=lr)
    opt_d = optim.Adam(discriminator.parameters(), lr=lr)
    bce = nn.BCELoss()

    real_data = _generate_real_data(1000, mass=target_mass)

    history: dict = {
        "d_loss": [],
        "g_loss": [],
        "g_bce": [],
        "g_phys": [],
        "g_mmd": [],
        "grad_norms": [],
        "avg_fake_mass": [],
        "barren_plateau_detected": False,
        "mode_collapse_detected": False,
    }

    ones = torch.ones(batch_size, 1)
    zeros = torch.zeros(batch_size, 1)

    for epoch in range(epochs):
        idx = torch.randint(0, real_data.size(0), (batch_size,))
        real_batch = real_data[idx]

        noise = torch.rand(batch_size, N_QUBITS) * np.pi
        fake_batch = generator(noise)

        # --- Discriminator step ---
        opt_d.zero_grad()
        loss_d = bce(discriminator(real_batch), ones) + bce(
            discriminator(fake_batch.detach()), zeros
        )
        loss_d.backward()
        opt_d.step()

        # --- Generator step ---
        opt_g.zero_grad()
        loss_g_bce = bce(discriminator(fake_batch), ones)
        loss_g_phys = physics_aware_loss(fake_batch, target_mass)
        loss_g_mmd = mmd_loss(fake_batch, real_batch)
        loss_g = loss_g_bce + lambda_phys * loss_g_phys + lambda_mmd * loss_g_mmd
        loss_g.backward()
        opt_g.step()

        grad_norm = generator.get_gradient_norm()

        # --- Bookkeeping ---
        history["d_loss"].append(loss_d.item())
        history["g_loss"].append(loss_g.item())
        history["g_bce"].append(loss_g_bce.item())
        history["g_phys"].append(loss_g_phys.item())
        history["g_mmd"].append(loss_g_mmd.item())
        history["grad_norms"].append(grad_norm)

        if epoch % 50 == 0:
            with torch.no_grad():
                fE = fake_batch[:, 0]
                fp2 = torch.sum(fake_batch[:, 1:] ** 2, dim=1)
                fm = torch.sqrt(torch.clamp(fE**2 - fp2, min=1e-8))
                avg_m = fm.mean().item()
                history["avg_fake_mass"].append(avg_m)

                if generator.detect_barren_plateau():
                    history["barren_plateau_detected"] = True
                    if verbose:
                        print(
                            f"  [!] Barren plateau detected at epoch {epoch}"
                        )

                if fm.var().item() < 0.01 and epoch > 100:
                    history["mode_collapse_detected"] = True
                    if verbose:
                        print(
                            f"  [!] Mode collapse detected at epoch {epoch}"
                        )

            if verbose:
                print(
                    f"Epoch {epoch:03d} | D {loss_d.item():.4f} | "
                    f"G {loss_g.item():.4f} | Phys {loss_g_phys.item():.4f} | "
                    f"MMD {loss_g_mmd.item():.4f} | Mass {avg_m:.3f} | "
                    f"||grad|| {grad_norm:.6f}"
                )

    return generator, history
