"""Training utilities for Auto-Schematic."""

from __future__ import annotations

import pickle
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from torch import nn
from torch.utils.data import DataLoader

if __package__ is None:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

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


def fit_pca_kmeans(latents: np.ndarray, labels: List[str]) -> Tuple[PCA, KMeans]:
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(latents)
    kmeans = KMeans(n_clusters=len(labels), random_state=42, n_init=10)
    kmeans.fit(reduced)
    return pca, kmeans


def extract_latents(model, dataset):
    model.eval()
    latents = []
    labels = []

    device = next(model.parameters()).device  # <-- bunu loop dışında almak daha iyi

    with torch.no_grad():
        for image, label in dataset:
            image = image.to(device)          # <-- ASIL EKLEYECEĞİN SATIR
            _, latent = model(image.unsqueeze(0))
            latents.append(latent.detach().cpu())  # <-- numpy vs için güvenli
            labels.append(label)

    return torch.cat(latents, dim=0), labels



def save_artifacts(
    output_dir: Path,
    model: ConvAutoencoder,
    pca: PCA,
    kmeans: KMeans,
    latent_dim: int,
    labels: List[str],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": model.state_dict(), "latent_dim": latent_dim, "labels": labels},
        output_dir / "autoencoder.pt",
    )
    with (output_dir / "pca.pkl").open("wb") as file:
        pickle.dump(pca, file)
    with (output_dir / "kmeans.pkl").open("wb") as file:
        pickle.dump(kmeans, file)


def train_pipeline(
    samples: List[Sample],
    labels: List[str],
    output_dir: Path,
    epochs: int = 5,
    batch_size: int = 32,
    latent_dim: int = 32,
) -> None:
    dataset = SymbolDataset(samples, labels)
    model = train_autoencoder(dataset, epochs=epochs, batch_size=batch_size, latent_dim=latent_dim)
    latents, _ = extract_latents(model, dataset)
    pca, kmeans = fit_pca_kmeans(latents, labels)
    import time

    # train_pipeline içinde, epoch loop'tan önce:
    log_interval = 50  # her 50 batch'te bir yazdır (istersen 10 yap)
    device = next(model.parameters()).device

    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        model.train()
        running = 0.0

        for step, batch in enumerate(loader, start=1):
            # batch bazen (image, label) gelir
            image = batch[0] if isinstance(batch, (tuple, list)) else batch
            image = image.to(device)

            recon, latent = model(image)              # senin forward'una göre
            loss = loss_fn(recon, image)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running += float(loss.item())

            if step % log_interval == 0:
                avg = running / step
                print(f"epoch {epoch:03d}/{epochs} | step {step:04d}/{len(loader)} | loss {avg:.6f}")

        avg_epoch = running / max(1, len(loader))
        dt = time.perf_counter() - t0
        print(f"epoch {epoch:03d} done | avg_loss {avg_epoch:.6f} | {dt:.1f}s")

    save_artifacts(output_dir, model, pca, kmeans, latent_dim, labels)


