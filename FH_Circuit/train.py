"""Training utilities for Auto-Schematic."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from torch import nn
from torch.utils.data import DataLoader

from FH_Circuit.config import SYMBOLS
from FH_Circuit.data import Sample
from FH_Circuit.dataset import SymbolDataset
from FH_Circuit.model import ConvAutoencoder


def train_autoencoder(
    dataset: SymbolDataset,
    epochs: int = 5,
    batch_size: int = 32,
    latent_dim: int = 32,
) -> ConvAutoencoder:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConvAutoencoder(latent_dim=latent_dim).to(device)
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


def extract_latents(model: ConvAutoencoder, dataset: SymbolDataset) -> Tuple[np.ndarray, List[int]]:
    latents = []
    labels = []
    model.eval()
    with torch.no_grad():
        for image, label in dataset:
            _, latent = model(image.unsqueeze(0))
            latents.append(latent.squeeze(0).numpy())
            labels.append(label)
    return np.array(latents), labels


def save_artifacts(
    output_dir: Path,
    model: ConvAutoencoder,
    pca: PCA,
    kmeans: KMeans,
    latent_dim: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "latent_dim": latent_dim}, output_dir / "autoencoder.pt")
    with (output_dir / "pca.pkl").open("wb") as file:
        pickle.dump(pca, file)
    with (output_dir / "kmeans.pkl").open("wb") as file:
        pickle.dump(kmeans, file)


def train_pipeline(
    samples: List[Sample],
    output_dir: Path,
    epochs: int = 5,
    batch_size: int = 32,
    latent_dim: int = 32,
) -> None:
    dataset = SymbolDataset(samples)
    model = train_autoencoder(dataset, epochs=epochs, batch_size=batch_size, latent_dim=latent_dim)
    latents, _ = extract_latents(model, dataset)
    pca, kmeans = fit_pca_kmeans(latents)
    save_artifacts(output_dir, model, pca, kmeans, latent_dim)
