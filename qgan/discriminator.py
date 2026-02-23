"""Classical discriminator network for the QGAN."""

import torch
import torch.nn as nn


class Discriminator(nn.Module):
    """Lightweight MLP that classifies 4-vectors as real or fake.

    Architecture: 4 → 32 → 16 → 1 with LeakyReLU activations.
    """

    def __init__(self, input_dim: int = 4):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 16),
            nn.LeakyReLU(0.2),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
