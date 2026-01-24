"""Classification utilities for Auto-Schematic."""

from __future__ import annotations

import dataclasses
import pickle
import sys
from pathlib import Path
from typing import List, Literal

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

if __package__ is None:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from FH_Circuit.config import (
    AMBIGUITY_THRESHOLD,
    ENABLE_TTA_ROTATIONS,
    ERROR_THRESHOLD,
    MAHALANOBIS_RECOMMENDED,
    MSE_NOISE_THRESHOLD,
    MSE_NOISE_THRESHOLD_LOOSE,
)
from FH_Circuit.latent_density import LatentDensityArtifacts, distance_to_label, nearest_class
from FH_Circuit.model import ConvAutoencoder, SupervisedAutoencoder
from FH_Circuit.preprocess import preprocess


@dataclasses.dataclass(frozen=True)
class StageArtifacts:
    model: ConvAutoencoder
    pca: PCA
    classifier: SVC
    labels: List[str]
    latent_scaler: StandardScaler
    latent_density: LatentDensityArtifacts | None


@dataclasses.dataclass(frozen=True)
class ClassificationResult:
    status: Literal["ok", "ambiguous", "novel", "noisy"]
    label: str | None
    candidates: list[tuple[str, float]]
    recon_error: float
    best_angle_deg: int | None
    nearest_label: str | None
    nearest_distance: float | None
    nearest_threshold: float | None
    message: str


_SCORE_SAMPLE_LIMIT = 5
_score_sample_count = 0
_mse_range: list[float] = []
_maha_range: list[float] = []


def _log_score_sample(recon_error: float, maha_distance: float | None) -> None:
    global _score_sample_count
    if _score_sample_count >= _SCORE_SAMPLE_LIMIT:
        return
    _score_sample_count += 1
    _mse_range.append(recon_error)
    if maha_distance is not None:
        _maha_range.append(maha_distance)
    mse_min = min(_mse_range)
    mse_max = max(_mse_range)
    if _maha_range:
        maha_min = min(_maha_range)
        maha_max = max(_maha_range)
        maha_part = f"maha=[{maha_min:.4f}, {maha_max:.4f}]"
    else:
        maha_part = "maha=disabled"
    print(
        "Inference score sample"
        f" {_score_sample_count}/{_SCORE_SAMPLE_LIMIT}:"
        f" mse={recon_error:.4f} (range [{mse_min:.4f}, {mse_max:.4f}]), {maha_part}"
    )


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
    latent_density_path = model_dir / "latent_density.pkl"
    has_scaler = latent_scaler_path.exists()
    has_density = latent_density_path.exists()
    if latent_scaler_path.exists():
        with latent_scaler_path.open("rb") as file:
            latent_scaler = pickle.load(file)
    else:
        print(
            f"Warning: latent_scaler.pkl not found in {model_dir}. "
            "Using identity scaler; classification may be degraded.",
            file=sys.stderr,
        )
        latent_scaler = StandardScaler()
        latent_scaler.mean_ = np.zeros(checkpoint["latent_dim"], dtype=np.float64)
        latent_scaler.scale_ = np.ones(checkpoint["latent_dim"], dtype=np.float64)
        latent_scaler.var_ = np.ones(checkpoint["latent_dim"], dtype=np.float64)
        latent_scaler.n_features_in_ = checkpoint["latent_dim"]
        latent_scaler.n_samples_seen_ = 1
    latent_density = None
    if latent_density_path.exists():
        with latent_density_path.open("rb") as file:
            latent_density = pickle.load(file)
        if not has_scaler:
            print(
                f"Warning: latent_density.pkl found in {model_dir} but latent_scaler.pkl is missing. "
                "Disabling Mahalanobis novelty detection.",
                file=sys.stderr,
            )
            latent_density = None
    density_status = "loaded" if latent_density is not None else ("disabled" if has_density else "missing")
    scaler_status = "loaded" if has_scaler else "missing"
    density_summary = ""
    if latent_density is not None:
        density_summary = (
            f" quantile={latent_density.quantile:.4f}"
            f" scale={latent_density.threshold_scale:.2f}"
            f" global_threshold={latent_density.global_threshold:.4f}"
        )
    else:
        density_summary = f" recommended={MAHALANOBIS_RECOMMENDED}"
    print(
        "Artifacts loaded"
        f" from={model_dir}"
        f" autoencoder={'yes' if checkpoint_path.exists() else 'no'}"
        f" pca={'yes' if (model_dir / 'pca.pkl').exists() else 'no'}"
        f" classifier={'yes' if (model_dir / 'classifier.pkl').exists() else 'no'}"
        f" latent_scaler={scaler_status}"
        f" latent_density={density_status}"
        f"{density_summary}"
    )
    return StageArtifacts(
        model=model,
        pca=pca,
        classifier=classifier,
        labels=labels,
        latent_scaler=latent_scaler,
        latent_density=latent_density,
    )


def load_artifacts(model_dir: Path) -> StageArtifacts:
    model_path = model_dir / "autoencoder.pt"
    if model_path.exists():
        return _load_stage_artifacts(model_dir)
    coarse_dir = model_dir / "coarse"
    if coarse_dir.exists():
        return _load_stage_artifacts(coarse_dir)
    return _load_stage_artifacts(model_dir)


def _compute_candidates(stage: StageArtifacts, reduced: np.ndarray, k: int = 2) -> tuple[list[tuple[str, float]], float]:
    probabilities = stage.classifier.predict_proba(reduced)[0]
    sorted_indices = np.argsort(probabilities)[::-1]
    top_indices = sorted_indices[:k]
    candidates: list[tuple[str, float]] = []
    for index in top_indices:
        if index < 0 or index >= len(stage.labels):
            raise ValueError("Model output out of range. Check training labels.")
        candidates.append((stage.labels[int(index)], float(probabilities[int(index)])))
    top_score = float(probabilities[int(sorted_indices[0])])
    second_score = float(probabilities[int(sorted_indices[1])]) if len(sorted_indices) > 1 else 0.0
    margin = top_score - second_score
    return candidates, margin


def _rotated_variants(processed: np.ndarray) -> list[tuple[int, np.ndarray]]:
    angles = (0, 90, 180, 270)
    variants: list[tuple[int, np.ndarray]] = []
    for k, angle in enumerate(angles):
        rotated = np.rot90(processed, k).copy() if k else processed.copy()
        variants.append((angle, rotated))
    return variants


def _forward_latent(artifacts: StageArtifacts, processed: np.ndarray) -> tuple[float, np.ndarray]:
    tensor = torch.from_numpy(processed).unsqueeze(0).unsqueeze(0).float()
    with torch.no_grad():
        outputs = artifacts.model(tensor)
        if len(outputs) == 2:
            recon, latent = outputs
        else:
            recon, latent, _ = outputs
    recon_error = torch.mean((recon - tensor) ** 2).item()
    normalized_latent = artifacts.latent_scaler.transform(latent.cpu().numpy())
    return recon_error, normalized_latent


def _best_rotation_by_density(artifacts: StageArtifacts, processed: np.ndarray) -> dict | None:
    density = artifacts.latent_density
    if density is None:
        return None
    best: dict | None = None
    for angle, rotated in _rotated_variants(processed):
        recon_error, normalized_latent = _forward_latent(artifacts, rotated)
        if recon_error > MSE_NOISE_THRESHOLD_LOOSE:
            continue
        nearest_label, dist, threshold = nearest_class(normalized_latent[0], density)
        # Pick the rotation with the tightest Mahalanobis fit to known classes.
        if best is None or dist < best["dist"]:
            best = {
                "angle": angle,
                "recon_error": recon_error,
                "normalized_latent": normalized_latent,
                "nearest_label": nearest_label,
                "dist": dist,
                "threshold": threshold,
            }
    return best


def classify_sketch_detailed(
    artifacts: StageArtifacts,
    sketch: np.ndarray,
    error_threshold: float = ERROR_THRESHOLD,
    ambiguity_threshold: float = AMBIGUITY_THRESHOLD,
    noise_threshold: float = MSE_NOISE_THRESHOLD,
) -> ClassificationResult:
    try:
        processed = preprocess(sketch)
    except ValueError as exc:
        message = f"Novelty detected: sketch too noisy. ({exc})"
        return ClassificationResult(
            status="noisy",
            label=None,
            candidates=[],
            recon_error=float("inf"),
            best_angle_deg=None,
            nearest_label=None,
            nearest_distance=None,
            nearest_threshold=None,
            message=message,
        )
    artifacts.model.eval()
    density = artifacts.latent_density
    nearest_label: str | None = None
    nearest_distance: float | None = None
    nearest_threshold: float | None = None
    best_angle_deg: int | None = 0
    if density is not None and ENABLE_TTA_ROTATIONS:
        best_rotation = _best_rotation_by_density(artifacts, processed)
        if best_rotation is None:
            message = "Novelty detected: sketch too noisy."
            return ClassificationResult(
                status="noisy",
                label=None,
                candidates=[],
                recon_error=float("inf"),
                best_angle_deg=None,
                nearest_label=None,
                nearest_distance=None,
                nearest_threshold=None,
                message=message,
            )
        recon_error = float(best_rotation["recon_error"])
        normalized_latent = best_rotation["normalized_latent"]
        best_angle_deg = int(best_rotation["angle"])
        nearest_label = best_rotation["nearest_label"]
        nearest_distance = float(best_rotation["dist"])
        nearest_threshold = float(best_rotation["threshold"])
        _log_score_sample(recon_error, nearest_distance)
        if nearest_distance > nearest_threshold:
            message = "Novelty detected: unknown component."
            return ClassificationResult(
                status="novel",
                label=None,
                candidates=[],
                recon_error=recon_error,
                best_angle_deg=best_angle_deg,
                nearest_label=nearest_label,
                nearest_distance=nearest_distance,
                nearest_threshold=nearest_threshold,
                message=message,
            )
    else:
        recon_error, normalized_latent = _forward_latent(artifacts, processed)
        if recon_error > noise_threshold:
            message = "Novelty detected: sketch too noisy."
            return ClassificationResult(
                status="noisy",
                label=None,
                candidates=[],
                recon_error=recon_error,
                best_angle_deg=best_angle_deg,
                nearest_label=None,
                nearest_distance=None,
                nearest_threshold=None,
                message=message,
            )
        if density is not None:
            nearest_label, nearest_distance, nearest_threshold = nearest_class(normalized_latent[0], density)
            _log_score_sample(recon_error, nearest_distance)
            if nearest_distance > nearest_threshold:
                message = "Novelty detected: unknown component."
                return ClassificationResult(
                    status="novel",
                    label=None,
                    candidates=[],
                    recon_error=recon_error,
                    best_angle_deg=best_angle_deg,
                    nearest_label=nearest_label,
                    nearest_distance=nearest_distance,
                    nearest_threshold=nearest_threshold,
                    message=message,
                )
        elif recon_error > error_threshold:
            message = "Novelty detected: unknown component."
            return ClassificationResult(
                status="novel",
                label=None,
                candidates=[],
                recon_error=recon_error,
                best_angle_deg=best_angle_deg,
                nearest_label=None,
                nearest_distance=None,
                nearest_threshold=None,
                message=message,
            )
        else:
            _log_score_sample(recon_error, None)
    reduced = artifacts.pca.transform(normalized_latent)
    candidates, margin = _compute_candidates(artifacts, reduced, k=2)
    label = candidates[0][0] if candidates else None
    ambiguous = margin < ambiguity_threshold or not label
    if density is not None and label:
        predicted_distance = distance_to_label(normalized_latent[0], label, density)
        predicted_threshold = density.class_stats[label].threshold
        if predicted_distance > predicted_threshold:
            message = "Novelty detected: unknown component."
            return ClassificationResult(
                status="novel",
                label=None,
                candidates=candidates,
                recon_error=recon_error,
                best_angle_deg=best_angle_deg,
                nearest_label=label,
                nearest_distance=predicted_distance,
                nearest_threshold=predicted_threshold,
                message=message,
            )
    if ambiguous:
        message = "Ambiguity detected: ask user to clarify between closest symbols."
        return ClassificationResult(
            status="ambiguous",
            label=None,
            candidates=candidates,
            recon_error=recon_error,
            best_angle_deg=best_angle_deg,
            nearest_label=nearest_label,
            nearest_distance=nearest_distance,
            nearest_threshold=nearest_threshold,
            message=message,
        )
    message = f"Detected: {label}"
    return ClassificationResult(
        status="ok",
        label=label,
        candidates=candidates,
        recon_error=recon_error,
        best_angle_deg=best_angle_deg,
        nearest_label=nearest_label,
        nearest_distance=nearest_distance,
        nearest_threshold=nearest_threshold,
        message=message,
    )


def classify_sketch(
    artifacts: StageArtifacts,
    sketch: np.ndarray,
    error_threshold: float = ERROR_THRESHOLD,
    ambiguity_threshold: float = AMBIGUITY_THRESHOLD,
    noise_threshold: float = MSE_NOISE_THRESHOLD,
) -> str:
    result = classify_sketch_detailed(
        artifacts,
        sketch,
        error_threshold=error_threshold,
        ambiguity_threshold=ambiguity_threshold,
        noise_threshold=noise_threshold,
    )
    if result.status == "ok" and result.label:
        return f"Detected: {result.label}"
    return result.message
