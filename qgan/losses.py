"""Loss functions used during QGAN training.

* ``physics_aware_loss`` – re-exported from ``physics_utils`` for convenience.
* ``mmd_loss``           – Maximum Mean Discrepancy for distribution matching.
"""

import torch

# Single source of truth for the mass-shell loss lives in physics_utils.
from physics_utils import relativistic_energy_momentum_loss as physics_aware_loss  # noqa: F401 — re-export


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
