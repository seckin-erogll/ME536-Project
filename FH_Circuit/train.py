"""Training utilities for Auto-Schematic."""

from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from torch import nn
from torch.utils.data import DataLoader

if __package__ is None:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from FH_Circuit.data import Sample
from FH_Circuit.dataset import SymbolDataset
from FH_Circuit.model import ConvAutoencoder


def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def describe_device(device: torch.device) -> str:
    if device.type == "cuda":
        return f"cuda ({torch.cuda.get_device_name(device)})"
    return device.type


def train_autoencoder(
    dataset: SymbolDataset,
    epochs: int = 5,
    batch_size: int = 32,
    latent_dim: int = 32,
    log_interval: int = 50,
) -> ConvAutoencoder:
    device = resolve_device()
    print(f"Training on device: {describe_device(device)}")
    model = ConvAutoencoder(latent_dim=latent_dim).to(device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        start = time.perf_counter()
        for step, (inputs, _) in enumerate(loader, start=1):
            inputs = inputs.to(device)
            recon, _ = model(inputs)
            loss = criterion(recon, inputs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if log_interval and step % log_interval == 0:
                avg_loss = running_loss / step
                print(f"Epoch {epoch}/{epochs} | Step {step}/{len(loader)} | Loss {avg_loss:.6f}")
        avg_epoch_loss = running_loss / max(1, len(loader))
        duration = time.perf_counter() - start
        print(f"Epoch {epoch}/{epochs} complete | Avg Loss {avg_epoch_loss:.6f} | {duration:.1f}s")
    return model


def fit_pca_classifier(latents: np.ndarray, labels: List[str], components: int) -> Tuple[PCA, KNeighborsClassifier]:
    pca = PCA(n_components=components)
    reduced = pca.fit_transform(latents)
    classifier = KNeighborsClassifier(n_neighbors=min(3, len(labels)))
    classifier.fit(reduced, labels)
    return pca, classifier


def extract_latents(model, dataset):
    model.eval()
    latents = []
    labels = []

    device = next(model.parameters()).device

    with torch.no_grad():
        for image, label in dataset:
            image = image.to(device)
            _, latent = model(image.unsqueeze(0))
            latents.append(latent.detach().cpu())
            labels.append(label)

    return torch.cat(latents, dim=0), labels



def save_artifacts(
    output_dir: Path,
    model: ConvAutoencoder,
    pca: PCA,
    classifier: KNeighborsClassifier,
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
    with (output_dir / "classifier.pkl").open("wb") as file:
        pickle.dump(classifier, file)


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
    components = min(latent_dim, 16)
    pca, classifier = fit_pca_classifier(latents, labels, components)
    save_artifacts(output_dir, model, pca, classifier, latent_dim, labels)
