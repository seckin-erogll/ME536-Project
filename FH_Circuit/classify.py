"""Classification utilities for Auto-Schematic."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

from FH_Circuit.config import AMBIGUITY_THRESHOLD, ERROR_THRESHOLD
from FH_Circuit.model import ConvAutoencoder
from FH_Circuit.preprocess import preprocess


def load_artifacts(model_dir: Path) -> Tuple[ConvAutoencoder, PCA, KNeighborsClassifier, List[str]]:
    checkpoint = torch.load(model_dir / "autoencoder.pt", map_location="cpu")
    model = ConvAutoencoder(latent_dim=checkpoint["latent_dim"])
    model.load_state_dict(checkpoint["state_dict"])
    labels = checkpoint["labels"]
    with (model_dir / "pca.pkl").open("rb") as file:
        pca = pickle.load(file)
    with (model_dir / "classifier.pkl").open("rb") as file:
        classifier = pickle.load(file)
    return model, pca, classifier, labels


def classify_sketch(
    model: ConvAutoencoder,
    pca: PCA,
    classifier: KNeighborsClassifier,
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
    distances, indices = classifier.kneighbors(reduced, n_neighbors=min(3, len(labels)))
    predicted = classifier.predict(reduced)[0]
    if distances.shape[1] > 1 and (distances[0, 1] - distances[0, 0]) < ambiguity_threshold:
        return "Ambiguity detected: ask user to clarify between closest symbols."
    if predicted not in labels:
        return "Model output out of range. Check training labels."
    return f"Detected: {predicted}"
