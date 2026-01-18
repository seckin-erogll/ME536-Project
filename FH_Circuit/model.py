"""Neural network models for supervised circuit symbol classification."""

from __future__ import annotations

import torch
from torch import nn

from FH_Circuit.config import GRAPH_FEATURE_DIM


class HybridClassifier(nn.Module):
    """CNN + MLP classifier that exposes fused_embedding for unknown detection."""

    def __init__(
        self,
        num_classes: int,
        image_embedding_dim: int = 128,
        graph_embedding_dim: int = 64,
    ) -> None:
        super().__init__()
        self.image_branch = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, image_embedding_dim),
            nn.ReLU(),
        )
        self.graph_branch = nn.Sequential(
            nn.Linear(GRAPH_FEATURE_DIM, graph_embedding_dim),
            nn.ReLU(),
            nn.Linear(graph_embedding_dim, graph_embedding_dim),
            nn.ReLU(),
        )
        fused_dim = image_embedding_dim + graph_embedding_dim
        self.classifier = nn.Linear(fused_dim, num_classes)

    def forward(self, images: torch.Tensor, graph_feats: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        image_embedding = self.image_branch(images)
        graph_embedding = self.graph_branch(graph_feats)
        # Fused feature space used for prototype distance unknown detection.
        fused_embedding = torch.cat([image_embedding, graph_embedding], dim=1)
        logits = self.classifier(fused_embedding)
        return logits, fused_embedding
