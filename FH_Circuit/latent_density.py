"""Latent-space density estimation utilities."""

from __future__ import annotations

import dataclasses
from typing import Dict, Iterable, List, Tuple

import numpy as np


@dataclasses.dataclass(frozen=True)
class LatentClassStats:
    """Statistics for a single class in latent space."""

    mean: np.ndarray
    inv_cov: np.ndarray
    threshold: float


@dataclasses.dataclass(frozen=True)
class LatentDensityArtifacts:
    """Collection of latent density statistics across all classes."""

    labels: List[str]
    class_stats: Dict[str, LatentClassStats]
    quantile: float
    reg_eps: float
    threshold_scale: float
    global_threshold: float


def _as_matrix(values: np.ndarray) -> np.ndarray:
    matrix = np.atleast_2d(values)
    if matrix.ndim != 2:
        raise ValueError("Expected a 2D matrix of latent embeddings.")
    return matrix


def mahalanobis_distance(latents: np.ndarray, mean: np.ndarray, inv_cov: np.ndarray) -> np.ndarray:
    """Compute Mahalanobis distances for a batch of latent embeddings."""

    latents = _as_matrix(latents)
    mean = np.asarray(mean)
    if mean.ndim != 1:
        raise ValueError("Mean vector must be one-dimensional.")
    centered = latents - mean
    distances_sq = np.einsum("bi,ij,bj->b", centered, inv_cov, centered)
    distances_sq = np.clip(distances_sq, a_min=0.0, a_max=None)
    return np.sqrt(distances_sq)


def _regularized_inverse_covariance(class_latents: np.ndarray, reg_eps: float) -> np.ndarray:
    cov = np.cov(class_latents, rowvar=False)
    cov = np.atleast_2d(cov)
    cov += reg_eps * np.eye(cov.shape[0], dtype=cov.dtype)
    return np.linalg.pinv(cov)


def compute_latent_density(
    latents: np.ndarray,
    sample_labels: Iterable[int],
    labels: List[str],
    *,
    reg_eps: float,
    quantile: float,
    threshold_scale: float,
) -> LatentDensityArtifacts:
    """Estimate per-class latent density using Mahalanobis distance."""

    latents = _as_matrix(np.asarray(latents))
    label_array = np.asarray(list(sample_labels), dtype=np.int64)
    if latents.shape[0] != label_array.shape[0]:
        raise ValueError("Latents and labels must have the same number of samples.")
    if not labels:
        raise ValueError("At least one class label is required.")

    latent_dim = latents.shape[1]
    class_stats: Dict[str, LatentClassStats] = {}
    thresholds: List[float] = []

    for index, label in enumerate(labels):
        class_latents = latents[label_array == index]
        if class_latents.size == 0:
            raise ValueError(f"No latent samples available for class '{label}'.")
        mean = class_latents.mean(axis=0)
        if class_latents.shape[0] < 2:
            inv_cov = np.eye(latent_dim, dtype=latents.dtype) / max(reg_eps, 1e-12)
            threshold = float("inf")
        else:
            inv_cov = _regularized_inverse_covariance(class_latents, reg_eps)
            distances = mahalanobis_distance(class_latents, mean, inv_cov)
            base_threshold = float(np.quantile(distances, quantile))
            threshold = base_threshold * threshold_scale
        thresholds.append(threshold)
        class_stats[label] = LatentClassStats(mean=mean, inv_cov=inv_cov, threshold=threshold)

    finite_thresholds = [value for value in thresholds if np.isfinite(value)]
    global_threshold = float(max(finite_thresholds)) if finite_thresholds else float("inf")
    return LatentDensityArtifacts(
        labels=list(labels),
        class_stats=class_stats,
        quantile=quantile,
        reg_eps=reg_eps,
        threshold_scale=threshold_scale,
        global_threshold=global_threshold,
    )


def distance_to_label(latent: np.ndarray, label: str, density: LatentDensityArtifacts) -> float:
    """Compute Mahalanobis distance from a latent vector to a specific class."""

    stats = density.class_stats[label]
    return float(mahalanobis_distance(np.asarray(latent), stats.mean, stats.inv_cov)[0])


def nearest_class(latent: np.ndarray, density: LatentDensityArtifacts) -> Tuple[str, float, float]:
    """Find the nearest class centroid and its Mahalanobis distance."""

    latent = np.asarray(latent)
    distances = {
        label: distance_to_label(latent, label, density) for label in density.labels
    }
    nearest_label = min(distances, key=distances.get)
    nearest_distance = distances[nearest_label]
    threshold = density.class_stats[nearest_label].threshold
    return nearest_label, nearest_distance, threshold
