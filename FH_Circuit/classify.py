"""Classification utilities for Auto-Schematic."""

from __future__ import annotations

import dataclasses
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

if __package__ is None:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from FH_Circuit.config import AMBIGUITY_THRESHOLD, ERROR_THRESHOLD
from FH_Circuit.data import AMBIGUOUS_COARSE_GROUPS, labels_for_coarse_group
from FH_Circuit.model import ConvAutoencoder
from FH_Circuit.preprocess import extract_graph_features_from_binary, preprocess


@dataclasses.dataclass(frozen=True)
class StageArtifacts:
    model: ConvAutoencoder
    pca: PCA
    classifier: SVC
    labels: List[str]
    feature_mean: np.ndarray
    feature_std: np.ndarray
    latent_scaler: StandardScaler


@dataclasses.dataclass(frozen=True)
class TwoStageArtifacts:
    coarse: StageArtifacts
    fine: Dict[str, StageArtifacts]


def _load_stage_artifacts(model_dir: Path) -> StageArtifacts:
    checkpoint_path = model_dir / "autoencoder.pt"
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = ConvAutoencoder(latent_dim=checkpoint["latent_dim"])
    model.load_state_dict(checkpoint["state_dict"])
    labels = checkpoint["labels"]
    feature_mean = checkpoint.get("feature_mean")
    feature_std = checkpoint.get("feature_std")
    if feature_mean is None or feature_std is None:
        feature_mean = np.zeros(6, dtype=np.float32)
        feature_std = np.ones(6, dtype=np.float32)
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
        feature_mean=np.array(feature_mean, dtype=np.float32),
        feature_std=np.array(feature_std, dtype=np.float32),
        latent_scaler=latent_scaler,
    )


def load_artifacts(model_dir: Path) -> TwoStageArtifacts:
    coarse_dir = model_dir / "coarse"
    if coarse_dir.exists():
        coarse = _load_stage_artifacts(coarse_dir)
        fine_artifacts: Dict[str, StageArtifacts] = {}
        for group in AMBIGUOUS_COARSE_GROUPS:
            group_dir = model_dir / "fine" / group
            if group_dir.exists():
                fine_artifacts[group] = _load_stage_artifacts(group_dir)
        return TwoStageArtifacts(coarse=coarse, fine=fine_artifacts)
    legacy = _load_stage_artifacts(model_dir)
    return TwoStageArtifacts(coarse=legacy, fine={})


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
    artifacts: TwoStageArtifacts,
    sketch: np.ndarray,
    error_threshold: float = ERROR_THRESHOLD,
    ambiguity_threshold: float = AMBIGUITY_THRESHOLD,
) -> str:
    processed = preprocess(sketch)
    graph_features = extract_graph_features_from_binary(processed)
    tensor = torch.from_numpy(processed).unsqueeze(0).unsqueeze(0).float()
    artifacts.coarse.model.eval()
    with torch.no_grad():
        recon, latent = artifacts.coarse.model(tensor)
    recon_error = torch.mean((recon - tensor) ** 2).item()
    if recon_error > error_threshold:
        return "Novelty detected: unknown component."
    normalized_latent = artifacts.coarse.latent_scaler.transform(latent.cpu().numpy())
    normalized_features = (graph_features - artifacts.coarse.feature_mean) / artifacts.coarse.feature_std
    combined = np.concatenate([normalized_latent, normalized_features[None, :]], axis=1)
    reduced = artifacts.coarse.pca.transform(combined)
    coarse_label, coarse_ambiguous = _predict_label(artifacts.coarse, reduced, ambiguity_threshold)
    if coarse_ambiguous or not coarse_label:
        return "Ambiguity detected: ask user to clarify between closest symbols."
    if coarse_label in AMBIGUOUS_COARSE_GROUPS and coarse_label in artifacts.fine:
        fine_stage = artifacts.fine[coarse_label]
        fine_stage.model.eval()
        with torch.no_grad():
            _, fine_latent = fine_stage.model(tensor)
        normalized_fine_latent = fine_stage.latent_scaler.transform(fine_latent.cpu().numpy())
        fine_features = (graph_features - fine_stage.feature_mean) / fine_stage.feature_std
        fine_combined = np.concatenate([normalized_fine_latent, fine_features[None, :]], axis=1)
        fine_reduced = fine_stage.pca.transform(fine_combined)
        fine_label, fine_ambiguous = _predict_label(fine_stage, fine_reduced, ambiguity_threshold)
        if fine_ambiguous or not fine_label:
            return "Ambiguity detected: ask user to clarify between closest symbols."
        return f"Detected: {fine_label}"
    fine_labels = labels_for_coarse_group(coarse_label)
    if len(fine_labels) == 1:
        return f"Detected: {fine_labels[0]}"
    return f"Detected: {coarse_label}"
