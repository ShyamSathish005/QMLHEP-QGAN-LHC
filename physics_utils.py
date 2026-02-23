"""Relativistic physics utilities for particle 4-vector validation.

This module provides standalone functions that enforce (or measure)
compliance with the relativistic energy-momentum relation:

    E² = |p|² + m²

They are usable both as **training losses** (differentiable, torch) and
as **numpy diagnostics** for post-hoc analysis.
"""

from __future__ import annotations

import numpy as np
import torch


# -----------------------------------------------------------------------
# Core loss (PyTorch – differentiable)
# -----------------------------------------------------------------------
def relativistic_energy_momentum_loss(
    four_vectors: torch.Tensor,
    mass: float = 1.0,
) -> torch.Tensor:
    r"""Relativistic energy-momentum loss (mass-shell constraint).

    Computes

    .. math::
        \mathcal{L}_{\text{phys}}
            = \frac{1}{N}\sum_{i=1}^{N}
              \bigl(E_i^2 - |\mathbf{p}_i|^2 - m^2\bigr)^2

    This is zero if and only if every event sits exactly on the mass
    shell.  It can be back-propagated through in a quantum–classical
    hybrid training loop.

    Parameters
    ----------
    four_vectors : torch.Tensor, shape (N, 4)
        Each row is ``[E, px, py, pz]``.
    mass : float
        Target invariant mass (in whatever energy units the data uses).

    Returns
    -------
    torch.Tensor
        Scalar loss value (non-negative).

    Examples
    --------
    >>> import torch
    >>> from physics_utils import relativistic_energy_momentum_loss
    >>> p = torch.tensor([[5.0, 3.0, 0.0, 4.0]])   # E²-p² = 0 ≠ m²=1
    >>> relativistic_energy_momentum_loss(p, mass=1.0)
    tensor(1.)
    """
    E = four_vectors[:, 0]
    p_sq = torch.sum(four_vectors[:, 1:] ** 2, dim=1)
    deviation = E**2 - p_sq - mass**2
    return torch.mean(deviation**2)


# -----------------------------------------------------------------------
# NumPy version (non-differentiable, for evaluation / plotting)
# -----------------------------------------------------------------------
def invariant_mass(four_vectors: np.ndarray) -> np.ndarray:
    """Return the invariant mass for each event.

    Parameters
    ----------
    four_vectors : (N, 4) ndarray
        ``[E, px, py, pz]`` per row.

    Returns
    -------
    (N,) ndarray
        ``sqrt(max(E² − |p|², 0))`` per event.
    """
    E = four_vectors[:, 0]
    p_sq = np.sum(four_vectors[:, 1:] ** 2, axis=1)
    return np.sqrt(np.maximum(E**2 - p_sq, 0.0))


def mass_shell_residual(
    four_vectors: np.ndarray,
    mass: float = 1.0,
) -> np.ndarray:
    r"""Per-event residual :math:`E^2 - |p|^2 - m^2`.

    A perfectly on-shell event has residual 0.
    """
    E = four_vectors[:, 0]
    p_sq = np.sum(four_vectors[:, 1:] ** 2, axis=1)
    return E**2 - p_sq - mass**2


def physics_compliance_fraction(
    four_vectors: np.ndarray,
    mass: float = 1.0,
    tolerance: float = 0.5,
) -> float:
    """Fraction of events satisfying ``|m_inv − m| < tolerance``."""
    m_inv = invariant_mass(four_vectors)
    return float(np.mean(np.abs(m_inv - mass) < tolerance))


# -----------------------------------------------------------------------
# Quick self-test
# -----------------------------------------------------------------------
if __name__ == "__main__":
    # On-shell 4-vectors (m=1): E² = p² + 1
    p = torch.randn(500, 3) * 2.0
    E = torch.sqrt(torch.sum(p**2, dim=1) + 1.0).unsqueeze(1)
    on_shell = torch.cat([E, p], dim=1)

    loss = relativistic_energy_momentum_loss(on_shell, mass=1.0)
    print(f"On-shell loss (should be ~0): {loss.item():.2e}")

    off_shell = torch.randn(500, 4) * 3.0
    loss2 = relativistic_energy_momentum_loss(off_shell, mass=1.0)
    print(f"Off-shell loss (should be >>0): {loss2.item():.2e}")

    frac = physics_compliance_fraction(on_shell.numpy(), mass=1.0, tolerance=0.1)
    print(f"On-shell compliance (tol=0.1): {frac:.1%}")
