"""Classification utilities for Auto-Schematic."""

from __future__ import annotations

import dataclasses
import pickle
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

if __package__ is None:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from FH_Circuit.config import AMBIGUITY_THRESHOLD, ERROR_THRESHOLD
from FH_Circuit.model import ConvAutoencoder, SupervisedAutoencoder
from FH_Circuit.preprocess import preprocess


@dataclasses.dataclass(frozen=True)
class StageArtifacts:
    model: ConvAutoencoder
    pca: PCA
    classifier: SVC
    labels: List[str]
    latent_scaler: StandardScaler


def _load_stage_artifacts(model_dir: Path) -> StageArtifacts:
    checkpoint_path = model_dir / "autoencoder.pt"
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_type = checkpoint.get("model_type", "autoencoder")
    num_classes = checkpoint.get("num_classes", len(checkpoint["labels"]))
    if model_type == "supervised":
        model = SupervisedAutoencoder(
            latent_dim=checkpoint["latent_dim"],
            num_classes=num_classes,
        )
    else:
        model = ConvAutoencoder(latent_dim=checkpoint["latent_dim"])
    model.load_state_dict(checkpoint["state_dict"])
    labels = checkpoint["labels"]
    with (model_dir / "pca.pkl").open("rb") as file:
        pca = pickle.load(file)
    with (model_dir / "classifier.pkl").open("rb") as file:
        classifier = pickle.load(file)
    latent_scaler_path = model_dir / "latent_scaler.pkl"
    if latent_scaler_path.exists():
        with latent_scaler_path.open("rb") as file:
            latent_scaler = pickle.load(file)
    else:
        latent_scaler = StandardScaler()
        latent_scaler.mean_ = np.zeros(checkpoint["latent_dim"], dtype=np.float64)
        latent_scaler.scale_ = np.ones(checkpoint["latent_dim"], dtype=np.float64)
        latent_scaler.var_ = np.ones(checkpoint["latent_dim"], dtype=np.float64)
        latent_scaler.n_features_in_ = checkpoint["latent_dim"]
        latent_scaler.n_samples_seen_ = 1
    return StageArtifacts(
        model=model,
        pca=pca,
        classifier=classifier,
        labels=labels,
        latent_scaler=latent_scaler,
    )


def load_artifacts(model_dir: Path) -> StageArtifacts:
    return _load_stage_artifacts(model_dir)


def _predict_label(
    stage: StageArtifacts,
    reduced: np.ndarray,
    ambiguity_threshold: float,
) -> Tuple[str, bool]:
    probabilities = stage.classifier.predict_proba(reduced)[0]
    top_indices = np.argsort(probabilities)[-2:]
    predicted_index = int(top_indices[-1])
    if predicted_index < 0 or predicted_index >= len(stage.labels):
        raise ValueError("Model output out of range. Check training labels.")
    top_score = probabilities[predicted_index]
    second_score = probabilities[top_indices[-2]] if len(top_indices) > 1 else 0.0
    if (top_score - second_score) < ambiguity_threshold:
        return "", True
    return stage.labels[predicted_index], False


def classify_sketch(
    artifacts: StageArtifacts,
    sketch: np.ndarray,
    error_threshold: float = ERROR_THRESHOLD,
    ambiguity_threshold: float = AMBIGUITY_THRESHOLD,
) -> str:
    processed = preprocess(sketch)
    tensor = torch.from_numpy(processed).unsqueeze(0).unsqueeze(0).float()
    artifacts.model.eval()
    with torch.no_grad():
        outputs = artifacts.model(tensor)
        if len(outputs) == 2:
            recon, latent = outputs
        else:
            recon, latent, _ = outputs
    recon_error = torch.mean((recon - tensor) ** 2).item()
    if recon_error > error_threshold:
        return "Novelty detected: unknown component."
    normalized_latent = artifacts.latent_scaler.transform(latent.cpu().numpy())
    reduced = artifacts.pca.transform(normalized_latent)
    label, ambiguous = _predict_label(artifacts, reduced, ambiguity_threshold)
    if ambiguous or not label:
        return "Ambiguity detected: ask user to clarify between closest symbols."
    return f"Detected: {label}"
