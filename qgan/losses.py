"""Loss functions used during QGAN training.

* ``physics_aware_loss`` – enforces the relativistic mass-shell condition.
* ``mmd_loss``           – Maximum Mean Discrepancy for distribution matching.
"""

import torch


def physics_aware_loss(
    fake_data: torch.Tensor,
    target_mass: float = 1.0,
) -> torch.Tensor:
    r"""Relativistic energy-momentum constraint.

    .. math::
        \mathcal{L}_{\text{phys}}
            = \frac{1}{N}\sum_i\!\bigl(E_i^2 - |\mathbf{p}_i|^2 - m^2\bigr)^2

    Parameters
    ----------
    fake_data : (N, 4) tensor
        Generated 4-vectors ``[E, px, py, pz]``.
    target_mass : float
        On-shell mass (normalised units, e.g. 1.0 for the Higgs).
    """
    E = fake_data[:, 0]
    px, py, pz = fake_data[:, 1], fake_data[:, 2], fake_data[:, 3]
    m_inv_sq = E**2 - (px**2 + py**2 + pz**2)
    return torch.mean((m_inv_sq - target_mass**2) ** 2)


def mmd_loss(
    x: torch.Tensor,
    y: torch.Tensor,
    sigma: float = 1.0,
) -> torch.Tensor:
    """Maximum Mean Discrepancy with a Gaussian kernel.

    Measures how far two sample sets are from living on the same
    distribution.  Lower is better.
    """

    def _kernel(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return torch.exp(-torch.cdist(u, v) ** 2 / (2 * sigma**2))

    return _kernel(x, x).mean() + _kernel(y, y).mean() - 2 * _kernel(x, y).mean()
