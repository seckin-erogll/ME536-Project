"""Unit tests for latent density estimation."""

from __future__ import annotations

import unittest

import numpy as np

from FH_Circuit.latent_density import compute_latent_density, distance_to_label, nearest_class


class LatentDensityTests(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(1234)
        class0 = rng.normal(loc=0.0, scale=0.4, size=(200, 4))
        class1 = rng.normal(loc=3.0, scale=0.5, size=(200, 4))
        self.latents = np.vstack([class0, class1]).astype(np.float64)
        self.labels = [0] * len(class0) + [1] * len(class1)
        self.label_names = ["resistor", "capacitor"]
        self.density = compute_latent_density(
            self.latents,
            self.labels,
            self.label_names,
            reg_eps=1e-3,
            quantile=0.99,
            threshold_scale=1.05,
        )

    def test_nearest_class_for_in_distribution_sample(self) -> None:
        sample = np.zeros(4, dtype=np.float64)
        label, distance, threshold = nearest_class(sample, self.density)
        self.assertEqual(label, "resistor")
        self.assertLessEqual(distance, threshold)

    def test_far_sample_exceeds_threshold(self) -> None:
        sample = np.full(4, 10.0, dtype=np.float64)
        _, distance, threshold = nearest_class(sample, self.density)
        self.assertGreater(distance, threshold * 2.0)

    def test_distance_to_label_respects_separation(self) -> None:
        sample = np.full(4, 3.0, dtype=np.float64)
        dist_cap = distance_to_label(sample, "capacitor", self.density)
        dist_res = distance_to_label(sample, "resistor", self.density)
        self.assertLess(dist_cap, dist_res)


if __name__ == "__main__":
    unittest.main()
