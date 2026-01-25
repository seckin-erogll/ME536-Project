"""Classification utilities for Auto-Schematic."""

from __future__ import annotations

import dataclasses
import json
import pickle
import sys
from pathlib import Path
from typing import List

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
    MSE_NOISE_THRESHOLD,
    MSE_NOISE_THRESHOLD_LOOSE,
)
from FH_Circuit.latent_density import LatentDensityArtifacts, nearest_class
from FH_Circuit.model import ConvAutoencoder, SupervisedAutoencoder
from FH_Circuit.preprocess import preprocess


@dataclasses.dataclass(frozen=True)
class StageArtifacts:
    model: ConvAutoencoder
    classifier: SVC
    labels: List[str]
    latent_scaler: StandardScaler
    latent_density: LatentDensityArtifacts | None
    ocsvm: OneClassSVM | None
    ocsvm_meta: dict | None
    recon_error_threshold: float | None


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
        "ocsvm.pkl": model_dir / "ocsvm.pkl",
        "ocsvm_meta.json": model_dir / "ocsvm_meta.json",
    }
    status = ", ".join(
        f"{name}={'loaded' if path.exists() else 'missing'}" for name, path in paths.items()
    )
    ocsvm_state = "enabled" if artifacts.ocsvm is not None else "disabled"
    print(f"Artifacts ({model_dir}): {status} | ocsvm={ocsvm_state}")


def _compute_topk(
    stage: StageArtifacts, reduced: np.ndarray, k: int = 3
) -> tuple[list[tuple[str, float]], float, float, float]:
    probabilities = stage.classifier.predict_proba(reduced)[0]
    sorted_indices = np.argsort(probabilities)[::-1]
    top_indices = sorted_indices[:k]
    topk: list[tuple[str, float]] = []
    for index in top_indices:
        if index < 0 or index >= len(stage.labels):
            raise ValueError("Model output out of range. Check training labels.")
        topk.append((stage.labels[int(index)], float(probabilities[int(index)])))
    max_proba = float(probabilities[int(sorted_indices[0])]) if len(sorted_indices) > 0 else 0.0
    p2 = float(probabilities[int(sorted_indices[1])]) if len(sorted_indices) > 1 else 0.0
    margin = max_proba - p2
    return topk, max_proba, p2, margin


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


def classify_sketch_detailed(
    artifacts: StageArtifacts,
    sketch: np.ndarray,
    recon_thresh: float = ERROR_THRESHOLD,
    min_confidence: float = 0.30,
    ambiguity_margin: float = 0.15,
    ambiguity_conf_floor: float = 0.70,
    ocsvm_cutoff: float = 0.0,
    noise_threshold: float = MSE_NOISE_THRESHOLD,
) -> dict:
    try:
        processed = preprocess(sketch)
    except ValueError as exc:
        return {
            "status": "REVIEW",
            "reasons": ["recon"],
            "topk": [],
            "max_proba": 0.0,
            "margin": 0.0,
            "recon_error": float("inf"),
            "ocsvm_score": None,
            "ocsvm_pred": None,
            "error": str(exc),
        }
    artifacts.model.eval()
    density = artifacts.latent_density
    if density is not None and ENABLE_TTA_ROTATIONS:
        best_rotation = _best_rotation_by_density(artifacts, processed)
        if best_rotation is None:
            return {
                "status": "REVIEW",
                "reasons": ["recon"],
                "topk": [],
                "max_proba": 0.0,
                "margin": 0.0,
                "recon_error": float("inf"),
                "ocsvm_score": None,
                "ocsvm_pred": None,
            }
        recon_error = float(best_rotation["recon_error"])
        normalized_latent = best_rotation["normalized_latent"]
    else:
        recon_error, normalized_latent = _forward_latent(artifacts, processed)
    ocsvm_score, ocsvm_pred = _ocsvm_decision(artifacts, normalized_latent)
    recon_threshold = artifacts.recon_error_threshold or recon_thresh
    if recon_error > noise_threshold:
        return {
            "status": "REVIEW",
            "reasons": ["recon"],
            "topk": [],
            "max_proba": 0.0,
            "margin": 0.0,
            "recon_error": recon_error,
            "ocsvm_score": ocsvm_score,
            "ocsvm_pred": ocsvm_pred,
        }
    topk, max_proba, _p2, margin = _compute_topk(artifacts, normalized_latent, k=3)
    reasons: list[str] = []
    if recon_error > recon_threshold:
        reasons.append("recon")
    ocsvm_available = ocsvm_score is not None or ocsvm_pred is not None
    if ocsvm_available:
        if ocsvm_pred == -1 or (ocsvm_score is not None and ocsvm_score < ocsvm_cutoff):
            reasons.append("ocsvm")
    if max_proba < min_confidence:
        reasons.append("low_conf")
    if reasons:
        return {
            "status": "REVIEW",
            "reasons": reasons,
            "topk": topk,
            "max_proba": max_proba,
            "margin": margin,
            "recon_error": recon_error,
            "ocsvm_score": ocsvm_score,
            "ocsvm_pred": ocsvm_pred,
        }
    if max_proba >= ambiguity_conf_floor:
        top_label = topk[0][0] if topk else None
        return {
            "status": "OK",
            "label": top_label,
            "confidence": max_proba,
            "topk": topk,
            "recon_error": recon_error,
            "ocsvm_score": ocsvm_score,
            "ocsvm_pred": ocsvm_pred,
        }
    if max_proba >= min_confidence and margin <= ambiguity_margin:
        return {
            "status": "AMBIGUITY",
            "topk": topk,
            "max_proba": max_proba,
            "margin": margin,
            "recon_error": recon_error,
            "ocsvm_score": ocsvm_score,
            "ocsvm_pred": ocsvm_pred,
        }
    top_label = topk[0][0] if topk else None
    return {
        "status": "OK",
        "label": top_label,
        "confidence": max_proba,
        "topk": topk,
        "recon_error": recon_error,
        "ocsvm_score": ocsvm_score,
        "ocsvm_pred": ocsvm_pred,
    }


def classify_sketch(
    artifacts: StageArtifacts,
    sketch: np.ndarray,
    recon_thresh: float = ERROR_THRESHOLD,
    ambiguity_margin: float = AMBIGUITY_THRESHOLD,
    noise_threshold: float = MSE_NOISE_THRESHOLD,
) -> str:
    result = classify_sketch_detailed(
        artifacts,
        sketch,
        recon_thresh=recon_thresh,
        ambiguity_margin=ambiguity_margin,
        noise_threshold=noise_threshold,
    )
    status = result.get("status")
    if status == "OK" and result.get("label"):
        return f"Detected: {result['label']}"
    if status == "AMBIGUITY":
        return "Ambiguity detected: ask user to clarify between closest symbols."
    reasons = ", ".join(result.get("reasons", []))
    prefix = "Needs review (novel/noise)."
    if reasons:
        return f"{prefix} Reasons: {reasons}."
    return prefix
