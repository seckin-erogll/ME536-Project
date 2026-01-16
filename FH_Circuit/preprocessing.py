"""Sketch preprocessing utilities."""

from __future__ import annotations

import numpy as np
from skimage import filters, morphology


def preprocess(image: np.ndarray, min_area: int = 15) -> np.ndarray:
    thresh = filters.threshold_otsu(image)
    binary = (image > thresh).astype(np.uint8)
    if binary.sum() < min_area:
        raise ValueError("Noise detected: sketch too small.")
    skeleton = morphology.skeletonize(binary)
    dilated = morphology.dilation(skeleton, morphology.square(3))
    return dilated.astype(np.float32)
