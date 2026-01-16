"""Inference and decision logic for sketches."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from .config import SYMBOLS
from .model import ConvAutoencoder
from .preprocessing import preprocess


def classify_sketch(
    model: ConvAutoencoder,
    pca: PCA,
    kmeans: KMeans,
    sketch: np.ndarray,
    error_threshold: float = 0.08,
    ambiguity_threshold: float = 0.15,
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
    return f"Detected: {SYMBOLS[cluster]}"
