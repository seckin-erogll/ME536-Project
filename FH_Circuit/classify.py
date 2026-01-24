"""Classification utilities for Auto-Schematic."""

from __future__ import annotations

import dataclasses
import json
import pickle
import sys
from pathlib import Path
from typing import List, Literal

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM, SVC

if __package__ is None:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from FH_Circuit.config import (
    AMBIGUITY_THRESHOLD,
    ENABLE_TTA_ROTATIONS,
    ERROR_THRESHOLD,
    MIN_CONFIDENCE,
    MSE_NOISE_THRESHOLD,
    MSE_NOISE_THRESHOLD_LOOSE,
)
from FH_Circuit.latent_density import LatentDensityArtifacts, distance_to_label, nearest_class
from FH_Circuit.model import ConvAutoencoder, SupervisedAutoencoder
from FH_Circuit.preprocess import preprocess


@dataclasses.dataclass(frozen=True)
class StageArtifacts:
    model: ConvAutoencoder
    classifier: SVC
    labels: List[str]
    latent_scaler: StandardScaler
    latent_density: LatentDensityArtifacts | None
    latent_protos: dict | None
    ocsvm: OneClassSVM | None
    ocsvm_meta: dict | None
    recon_error_threshold: float | None


@dataclasses.dataclass(frozen=True)
class ClassificationResult:
    status: Literal["ok", "ambiguous", "novel", "noisy"]
    novelty_label: Literal["KNOWN", "UNKNOWN", "BAD_CROP"]
    label: str | None
    candidates: list[tuple[str, float]]
    recon_error: float
    recon_threshold: float | None
    ocsvm_score: float | None
    ocsvm_pred: int | None
    best_angle_deg: int | None
    nearest_label: str | None
    nearest_distance: float | None
    nearest_threshold: float | None
    message: str


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
    with (model_dir / "classifier.pkl").open("rb") as file:
        classifier = pickle.load(file)
    latent_scaler_path = model_dir / "latent_scaler.pkl"
    has_latent_scaler = latent_scaler_path.exists()
    if has_latent_scaler:
        with latent_scaler_path.open("rb") as file:
            latent_scaler = pickle.load(file)
    else:
        latent_scaler = StandardScaler()
        latent_scaler.mean_ = np.zeros(checkpoint["latent_dim"], dtype=np.float64)
        latent_scaler.scale_ = np.ones(checkpoint["latent_dim"], dtype=np.float64)
        latent_scaler.var_ = np.ones(checkpoint["latent_dim"], dtype=np.float64)
        latent_scaler.n_features_in_ = checkpoint["latent_dim"]
        latent_scaler.n_samples_seen_ = 1
    latent_density_path = model_dir / "latent_density.pkl"
    latent_density = None
    if latent_density_path.exists():
        with latent_density_path.open("rb") as file:
            latent_density = pickle.load(file)
    latent_protos_path = model_dir / "latent_protos.pkl"
    latent_protos = None
    if latent_protos_path.exists():
        with latent_protos_path.open("rb") as file:
            latent_protos = pickle.load(file)
    ocsvm_path = model_dir / "ocsvm.pkl"
    ocsvm_meta_path = model_dir / "ocsvm_meta.json"
    ocsvm = None
    ocsvm_meta = None
    recon_error_threshold = None
    if ocsvm_path.exists():
        if not has_latent_scaler:
            print(
                "WARNING: ocsvm.pkl found but latent_scaler.pkl is missing. Disabling One-Class SVM."
            )
        else:
            with ocsvm_path.open("rb") as file:
                ocsvm = pickle.load(file)
            if ocsvm_meta_path.exists():
                with ocsvm_meta_path.open("r", encoding="utf-8") as file:
                    ocsvm_meta = json.load(file)
                recon_error_threshold = ocsvm_meta.get("recon_error_threshold")
            else:
                print("WARNING: ocsvm_meta.json missing; using default recon threshold.")
    return StageArtifacts(
        model=model,
        classifier=classifier,
        labels=labels,
        latent_scaler=latent_scaler,
        latent_density=latent_density,
        latent_protos=latent_protos,
        ocsvm=ocsvm,
        ocsvm_meta=ocsvm_meta,
        recon_error_threshold=recon_error_threshold,
    )


def load_artifacts(model_dir: Path) -> StageArtifacts:
    model_path = model_dir / "autoencoder.pt"
    if model_path.exists():
        artifacts = _load_stage_artifacts(model_dir)
        _log_artifacts(model_dir, artifacts)
        return artifacts
    coarse_dir = model_dir / "coarse"
    if coarse_dir.exists():
        artifacts = _load_stage_artifacts(coarse_dir)
        _log_artifacts(coarse_dir, artifacts)
        return artifacts
    artifacts = _load_stage_artifacts(model_dir)
    _log_artifacts(model_dir, artifacts)
    return artifacts


def _log_artifacts(model_dir: Path, artifacts: StageArtifacts) -> None:
    paths = {
        "autoencoder.pt": model_dir / "autoencoder.pt",
        "classifier.pkl": model_dir / "classifier.pkl",
        "latent_scaler.pkl": model_dir / "latent_scaler.pkl",
        "latent_density.pkl": model_dir / "latent_density.pkl",
        "latent_protos.pkl": model_dir / "latent_protos.pkl",
        "ocsvm.pkl": model_dir / "ocsvm.pkl",
        "ocsvm_meta.json": model_dir / "ocsvm_meta.json",
    }
    status = ", ".join(
        f"{name}={'loaded' if path.exists() else 'missing'}" for name, path in paths.items()
    )
    ocsvm_state = "enabled" if artifacts.ocsvm is not None else "disabled"
    print(f"Artifacts ({model_dir}): {status} | ocsvm={ocsvm_state}")


def _compute_candidates(
    stage: StageArtifacts, reduced: np.ndarray, k: int = 2
) -> tuple[list[tuple[str, float]], float, float]:
    if not hasattr(stage.classifier, "predict_proba"):
        raise ValueError(
            "Classifier missing predict_proba; ensure SVC was trained with probability=True."
        )
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
    return candidates, margin, top_score


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


def _ocsvm_decision(
    artifacts: StageArtifacts,
    normalized_latent: np.ndarray,
) -> tuple[float | None, int | None]:
    if artifacts.ocsvm is None:
        return None, None
    score = float(artifacts.ocsvm.decision_function(normalized_latent)[0])
    pred = int(artifacts.ocsvm.predict(normalized_latent)[0])
    return score, pred


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


def _nearest_proto(
    normalized_latent: np.ndarray, latent_protos: dict
) -> tuple[str, float, float]:
    labels = latent_protos.get("labels", [])
    mu_map = latent_protos.get("mu", {})
    sigma_map = latent_protos.get("sigma", {})
    thr_map = latent_protos.get("thr", {})
    if not labels:
        raise ValueError("latent_protos is missing label metadata.")
    nearest_label = ""
    nearest_distance = float("inf")
    nearest_threshold = float("inf")
    for label in labels:
        mu = np.asarray(mu_map[label])
        sigma = np.asarray(sigma_map[label])
        diff = (normalized_latent - mu) / sigma
        dist = float(np.linalg.norm(diff))
        if dist < nearest_distance:
            nearest_label = label
            nearest_distance = dist
            nearest_threshold = float(thr_map[label])
    return nearest_label, nearest_distance, nearest_threshold


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
            novelty_label="BAD_CROP",
            label=None,
            candidates=[],
            recon_error=float("inf"),
            recon_threshold=artifacts.recon_error_threshold or error_threshold,
            ocsvm_score=None,
            ocsvm_pred=None,
            best_angle_deg=None,
            nearest_label=None,
            nearest_distance=None,
            nearest_threshold=None,
            message=message,
        )
    artifacts.model.eval()
    density = artifacts.latent_density
    latent_protos = artifacts.latent_protos
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
                novelty_label="BAD_CROP",
                label=None,
                candidates=[],
                recon_error=float("inf"),
                recon_threshold=artifacts.recon_error_threshold or error_threshold,
                ocsvm_score=None,
                ocsvm_pred=None,
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
    else:
        recon_error, normalized_latent = _forward_latent(artifacts, processed)
        if density is not None:
            nearest_label, nearest_distance, nearest_threshold = nearest_class(normalized_latent[0], density)
    ocsvm_score, ocsvm_pred = _ocsvm_decision(artifacts, normalized_latent)
    recon_threshold = artifacts.recon_error_threshold or error_threshold
    if recon_error > noise_threshold:
        message = "Novelty detected: sketch too noisy."
        return ClassificationResult(
            status="noisy",
            novelty_label="BAD_CROP",
            label=None,
            candidates=[],
            recon_error=recon_error,
            recon_threshold=recon_threshold,
            ocsvm_score=ocsvm_score,
            ocsvm_pred=ocsvm_pred,
            best_angle_deg=best_angle_deg,
            nearest_label=nearest_label,
            nearest_distance=nearest_distance,
            nearest_threshold=nearest_threshold,
            message=message,
        )
    if recon_error > error_threshold:
        message = "Novelty detected: unknown component."
        return ClassificationResult(
            status="novel",
            novelty_label="UNKNOWN",
            label=None,
            candidates=[],
            recon_error=recon_error,
            recon_threshold=recon_threshold,
            ocsvm_score=ocsvm_score,
            ocsvm_pred=ocsvm_pred,
            best_angle_deg=best_angle_deg,
            nearest_label=nearest_label,
            nearest_distance=nearest_distance,
            nearest_threshold=nearest_threshold,
            message=message,
        )
    candidates, margin, max_proba = _compute_candidates(artifacts, normalized_latent, k=2)
    if max_proba < MIN_CONFIDENCE:
        if recon_error > error_threshold:
            novelty_label = "UNKNOWN"
            message = "Novelty detected: unknown component."
            status = "novel"
        else:
            novelty_label = "BAD_CROP"
            message = "Novelty detected: bad crop."
            status = "noisy"
        return ClassificationResult(
            status=status,
            novelty_label=novelty_label,
            label=None,
            candidates=candidates,
            recon_error=recon_error,
            recon_threshold=recon_threshold,
            ocsvm_score=ocsvm_score,
            ocsvm_pred=ocsvm_pred,
            best_angle_deg=best_angle_deg,
            nearest_label=nearest_label,
            nearest_distance=nearest_distance,
            nearest_threshold=nearest_threshold,
            message=message,
        )
    if latent_protos is not None:
        proto_label, proto_distance, proto_threshold = _nearest_proto(
            normalized_latent[0], latent_protos
        )
        nearest_label = proto_label
        nearest_distance = proto_distance
        nearest_threshold = proto_threshold
        if proto_distance > proto_threshold:
            if recon_error > error_threshold:
                novelty_label = "UNKNOWN"
                message = "Novelty detected: unknown component."
                status = "novel"
            else:
                novelty_label = "BAD_CROP"
                message = "Novelty detected: bad crop."
                status = "noisy"
            return ClassificationResult(
                status=status,
                novelty_label=novelty_label,
                label=None,
                candidates=candidates,
                recon_error=recon_error,
                recon_threshold=recon_threshold,
                ocsvm_score=ocsvm_score,
                ocsvm_pred=ocsvm_pred,
                best_angle_deg=best_angle_deg,
                nearest_label=nearest_label,
                nearest_distance=nearest_distance,
                nearest_threshold=nearest_threshold,
                message=message,
            )
    label = candidates[0][0] if candidates else None
    ambiguous = margin < ambiguity_threshold or not label
    if density is not None and label and latent_protos is None:
        predicted_distance = distance_to_label(normalized_latent[0], label, density)
        predicted_threshold = density.class_stats[label].threshold
        nearest_label = label
        nearest_distance = predicted_distance
        nearest_threshold = predicted_threshold
    if ambiguous:
        message = "Ambiguity detected: ask user to clarify between closest symbols."
        return ClassificationResult(
            status="ambiguous",
            novelty_label="KNOWN",
            label=None,
            candidates=candidates,
            recon_error=recon_error,
            recon_threshold=recon_threshold,
            ocsvm_score=ocsvm_score,
            ocsvm_pred=ocsvm_pred,
            best_angle_deg=best_angle_deg,
            nearest_label=nearest_label,
            nearest_distance=nearest_distance,
            nearest_threshold=nearest_threshold,
            message=message,
        )
    message = f"Detected: {label}"
    return ClassificationResult(
        status="ok",
        novelty_label="KNOWN",
        label=label,
        candidates=candidates,
        recon_error=recon_error,
        recon_threshold=recon_threshold,
        ocsvm_score=ocsvm_score,
        ocsvm_pred=ocsvm_pred,
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
