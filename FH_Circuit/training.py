"""Training utilities for Auto-Schematic."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .config import SYMBOLS
from .data import Sample
from .model import ConvAutoencoder
from .preprocessing import preprocess


class SymbolDataset(Dataset):
    def __init__(self, samples: List[Sample]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        sample = self.samples[idx]
        image = preprocess(sample.image)
        tensor = torch.from_numpy(image).unsqueeze(0)
        label = SYMBOLS.index(sample.label)
        return tensor, label


@dataclass
class TrainingArtifacts:
    model: ConvAutoencoder
    pca: PCA
    kmeans: KMeans


def train_autoencoder(dataset: SymbolDataset, epochs: int = 5, batch_size: int = 32) -> ConvAutoencoder:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConvAutoencoder().to(device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    model.train()
    for _ in range(epochs):
        for inputs, _ in loader:
            inputs = inputs.to(device)
            recon, _ = model(inputs)
            loss = criterion(recon, inputs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model


def fit_pca_kmeans(latents: np.ndarray) -> Tuple[PCA, KMeans]:
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(latents)
    kmeans = KMeans(n_clusters=len(SYMBOLS), random_state=42, n_init=10)
    kmeans.fit(reduced)
    return pca, kmeans


def train_pipeline(samples: List[Sample], epochs: int = 5, batch_size: int = 32) -> TrainingArtifacts:
    dataset = SymbolDataset(samples)
    model = train_autoencoder(dataset, epochs=epochs, batch_size=batch_size)
    latents: List[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for image, _ in dataset:
            _, latent = model(image.unsqueeze(0))
            latents.append(latent.squeeze(0).numpy())
    pca, kmeans = fit_pca_kmeans(np.array(latents))
    return TrainingArtifacts(model=model, pca=pca, kmeans=kmeans)


def save_artifacts(artifacts: TrainingArtifacts, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(artifacts.model.state_dict(), output_dir / "autoencoder.pt")
    joblib.dump(artifacts.pca, output_dir / "pca.joblib")
    joblib.dump(artifacts.kmeans, output_dir / "kmeans.joblib")


def load_artifacts(output_dir: Path) -> TrainingArtifacts:
    model = ConvAutoencoder()
    model.load_state_dict(torch.load(output_dir / "autoencoder.pt", map_location="cpu"))
    pca = joblib.load(output_dir / "pca.joblib")
    kmeans = joblib.load(output_dir / "kmeans.joblib")
    return TrainingArtifacts(model=model, pca=pca, kmeans=kmeans)
