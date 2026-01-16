"""Convolutional autoencoder definition."""

from __future__ import annotations

from typing import Tuple

import torch
from torch import nn

from .config import LATENT_DIM


class ConvAutoencoder(nn.Module):
    def __init__(self, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 16 * 16, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32 * 16 * 16),
            nn.ReLU(),
            nn.Unflatten(1, (32, 16, 16)),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        latent = self.encoder(x)
        recon = self.decoder(latent)
        return recon, latent
