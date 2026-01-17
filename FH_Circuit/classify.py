"""Classification utilities for Auto-Schematic."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from FH_Circuit.config import AMBIGUITY_THRESHOLD, ERROR_THRESHOLD
from FH_Circuit.model import ConvAutoencoder
from FH_Circuit.preprocess import preprocess


def load_artifacts(model_dir: Path) -> Tuple[ConvAutoencoder, PCA, KMeans, List[str]]:
    checkpoint = torch.load(model_dir / "autoencoder.pt", map_location="cpu")
    model = ConvAutoencoder(latent_dim=checkpoint["latent_dim"])
    model.load_state_dict(checkpoint["state_dict"])
    labels = checkpoint["labels"]
    with (model_dir / "pca.pkl").open("rb") as file:
        pca = pickle.load(file)
    with (model_dir / "kmeans.pkl").open("rb") as file:
        kmeans = pickle.load(file)
    return model, pca, kmeans, labels


def classify_sketch(
    model: ConvAutoencoder,
    pca: PCA,
    kmeans: KMeans,
    sketch: np.ndarray,
    labels: List[str],
    error_threshold: float = ERROR_THRESHOLD,
    ambiguity_threshold: float = AMBIGUITY_THRESHOLD,
) -> str:
    processed = preprocess(sketch)
    tensor = torch.from_numpy(processed).unsqueeze(0).unsqueeze(0).float()
    model.eval()
    with torch.no_grad():
        recon, latent = model(tensor)
    recon_error = torch.mean((recon - tensor) ** 2).item()
    if recon_error > error_threshold:
        return "Novelty detected: unknown component."
    reduced = pca.transform(latent.cpu().numpy())
    cluster = kmeans.predict(reduced)[0]
    distances = np.linalg.norm(kmeans.cluster_centers_ - reduced, axis=1)
    sorted_dist = np.sort(distances)
    if len(sorted_dist) > 1 and sorted_dist[1] - sorted_dist[0] < ambiguity_threshold:
        return "Ambiguity detected: ask user to clarify between closest symbols."
    if cluster < 0 or cluster >= len(labels):
        return "Model output out of range. Check training labels."
    return f"Detected: {labels[cluster]}"
