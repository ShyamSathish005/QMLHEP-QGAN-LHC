"""Visualisation utilities for QGAN-HEP results."""

from __future__ import annotations

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance

from qgan.generator import QuantumGenerator, N_QUBITS


def visualize_results(
    generator: QuantumGenerator,
    history: dict,
    n_samples: int = 2000,
    target_mass: float = 1.0,
    save_path: str = "qgan_hep_analysis.png",
) -> None:
    """Generate the 6-panel diagnostic figure and print statistics.

    Parameters
    ----------
    target_mass : float
        The on-shell mass used during training (normalised units).
    """
    generator.eval()
    torch.manual_seed(42)
    np.random.seed(42)

    # ---- Real reference data (same distribution as training) ----
    from qgan.train import _generate_real_data

    real_data = _generate_real_data(n_samples, mass=target_mass)
    real_E = real_data[:, 0]
    real_p = real_data[:, 1:]

    real_E_np = real_E.numpy()
    real_p_sq = torch.sum(real_p**2, dim=1).numpy()
    real_mass = np.sqrt(np.maximum(real_E_np**2 - real_p_sq, 0.0))

    # ---- Generated data ----
    with torch.no_grad():
        noise = torch.rand(n_samples, N_QUBITS) * np.pi
        fake_data = generator(noise)

    fake_E = fake_data[:, 0].numpy()
    fake_p_sq = torch.sum(fake_data[:, 1:] ** 2, dim=1).numpy()
    fake_mass = np.sqrt(np.maximum(fake_E**2 - fake_p_sq, 0))

    # ---- Plot ----
    fig, axes = plt.subplots(2, 3, figsize=(16, 12))

    # (a) Mass distribution
    ax = axes[0, 0]
    ax.hist(real_mass, bins=50, alpha=0.6, color="blue", edgecolor="black",
            label="Real", density=True)
    ax.hist(fake_mass, bins=50, alpha=0.6, color="teal", edgecolor="black",
            label="QGAN", density=True)
    ax.axvline(target_mass, color="red", ls="--", lw=2,
               label=f"Target m={target_mass}")
    ax.set(xlabel="Invariant mass", ylabel="Density",
           title="(a) Mass distribution")
    ax.legend(); ax.grid(alpha=0.3)

    # (b) Transverse momentum
    ax = axes[0, 1]
    real_pt = np.sqrt(real_p[:, 0].numpy() ** 2 + real_p[:, 1].numpy() ** 2)
    fake_pt = np.sqrt(fake_data[:, 1].numpy() ** 2 + fake_data[:, 2].numpy() ** 2)
    ax.hist(real_pt, bins=40, alpha=0.6, color="blue", edgecolor="black",
            label="Real pT", density=True)
    ax.hist(fake_pt, bins=40, alpha=0.6, color="teal", edgecolor="black",
            label="QGAN pT", density=True)
    ax.set(xlabel="pT (GeV)", ylabel="Density",
           title="(b) Transverse momentum")
    ax.legend(); ax.grid(alpha=0.3)

    # (c) Energy
    ax = axes[0, 2]
    ax.hist(real_E_np, bins=40, alpha=0.6, color="blue", edgecolor="black",
            label="Real E", density=True)
    ax.hist(fake_E, bins=40, alpha=0.6, color="teal", edgecolor="black",
            label="QGAN E", density=True)
    ax.set(xlabel="Energy (GeV)", ylabel="Density",
           title="(c) Energy distribution")
    ax.legend(); ax.grid(alpha=0.3)

    # (d) Training losses
    ax = axes[1, 0]
    ax.plot(history["d_loss"], lw=2, color="darkred", label="D loss")
    ax.plot(history["g_loss"], lw=2, color="darkgreen", label="G loss")
    ax.set(xlabel="Epoch", ylabel="Loss", title="(d) Training dynamics")
    ax.legend(); ax.grid(alpha=0.3)

    # (e) Gradient norm
    ax = axes[1, 1]
    ax.semilogy(history["grad_norms"], lw=2, color="purple",
                label="||grad||")
    ax.axhline(1e-6, color="red", ls="--", lw=2, label="Plateau threshold")
    ax.set(xlabel="Step", ylabel="||grad|| (log)", title="(e) Gradient health")
    ax.legend(); ax.grid(alpha=0.3, which="both")

    # (f) Physics / MMD loss
    ax = axes[1, 2]
    ax.plot(history["g_phys"], lw=2, color="orange", label="Physics loss")
    ax.plot(history["g_mmd"], lw=2, color="brown", label="MMD loss")
    ax.set(xlabel="Epoch", ylabel="Loss", title="(f) Physics vs MMD")
    ax.legend(); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[saved] {save_path}")

    # ---- Statistics ----
    w = wasserstein_distance(real_mass, fake_mass)
    print(f"\nReal  mass  mean={real_mass.mean():.4f}  std={real_mass.std():.4f}")
    print(f"Fake  mass  mean={fake_mass.mean():.4f}  std={fake_mass.std():.4f}")
    print(f"Wasserstein distance (mass): {w:.4f}")
    if history["barren_plateau_detected"]:
        print("[!] Barren plateau was detected during training")
    if history["mode_collapse_detected"]:
        print("[!] Mode collapse was detected during training")
