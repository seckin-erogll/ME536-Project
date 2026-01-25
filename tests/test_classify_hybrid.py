"""Tests for hybrid reconstruction + Mahalanobis novelty detection."""

from __future__ import annotations

import unittest

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from FH_Circuit.classify import StageArtifacts, classify_sketch
from FH_Circuit.config import IMAGE_SIZE
from FH_Circuit.latent_density import compute_latent_density


def _square_sketch() -> np.ndarray:
    sketch = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
    sketch[16:48, 16:48] = 255.0
    return sketch


class _DummyModel(torch.nn.Module):
    def __init__(self, recon_scale: float, latent: np.ndarray) -> None:
        super().__init__()
        self._recon_scale = recon_scale
        self._latent = torch.tensor(latent, dtype=torch.float32).unsqueeze(0)

    def forward(self, tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        recon = tensor * self._recon_scale
        return recon, self._latent.to(tensor.device)


def _artifacts_for_latent(latent: np.ndarray, recon_scale: float) -> StageArtifacts:
    rng = np.random.default_rng(42)
    class0 = rng.normal(loc=0.0, scale=0.35, size=(200, 2))
    class1 = rng.normal(loc=3.0, scale=0.4, size=(200, 2))
    train_latents = np.vstack([class0, class1]).astype(np.float64)
    train_labels = [0] * len(class0) + [1] * len(class1)
    label_names = ["resistor", "capacitor"]

    scaler = StandardScaler().fit(train_latents)
    normalized = scaler.transform(train_latents)
    pca = PCA(n_components=2).fit(normalized)
    reduced = pca.transform(normalized)
    classifier = SVC(kernel="rbf", probability=True, gamma="scale").fit(reduced, train_labels)
    density = compute_latent_density(
        normalized,
        train_labels,
        label_names,
        reg_eps=1e-3,
        quantile=0.99,
        threshold_scale=1.05,
    )
    model = _DummyModel(recon_scale=recon_scale, latent=np.asarray(latent, dtype=np.float32))
    return StageArtifacts(
        model=model,
        pca=pca,
        classifier=classifier,
        labels=label_names,
        latent_scaler=scaler,
        latent_density=density,
    )


class HybridClassificationTests(unittest.TestCase):
    def test_high_mse_rejects_as_noise_before_density_check(self) -> None:
        artifacts = _artifacts_for_latent(latent=[0.1, 0.0], recon_scale=0.0)
        result = classify_sketch(artifacts, _square_sketch(), noise_threshold=0.05)
        self.assertEqual(result, "Novelty detected: sketch too noisy.")

    def test_low_mse_but_far_latent_rejects_as_unknown_component(self) -> None:
        artifacts = _artifacts_for_latent(latent=[10.0, 10.0], recon_scale=1.0)
        result = classify_sketch(artifacts, _square_sketch())
        self.assertEqual(result, "Novelty detected: unknown component.")

    def test_low_mse_and_near_latent_classifies_successfully(self) -> None:
        artifacts = _artifacts_for_latent(latent=[0.05, 0.0], recon_scale=1.0)
        result = classify_sketch(artifacts, _square_sketch())
        self.assertEqual(result, "Detected: resistor")


if __name__ == "__main__":
    unittest.main()
