"""Image preprocessing utilities."""

from __future__ import annotations

import numpy as np
from PIL import Image
from skimage import filters, morphology

from FH_Circuit.config import IMAGE_SIZE, MIN_AREA


def preprocess(image: np.ndarray, min_area: int = MIN_AREA) -> np.ndarray:
    thresh = filters.threshold_otsu(image)
    binary = (image > thresh).astype(np.uint8)
    if binary.sum() < min_area:
        raise ValueError("Noise detected: sketch too small.")
    binary = _normalize_to_canvas(binary)
    footprint = morphology.footprint_rectangle((3, 3))
    dilated = morphology.dilation(binary, footprint=footprint)
    return dilated.astype(np.float32)


def _normalize_to_canvas(binary: np.ndarray) -> np.ndarray:
    coords = np.column_stack(np.where(binary > 0))
    if coords.size == 0:
        return binary
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    cropped = binary[y_min : y_max + 1, x_min : x_max + 1]
    height, width = cropped.shape
    if height == 0 or width == 0:
        return binary
    scale = (IMAGE_SIZE * 0.8) / max(height, width)
    new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    resized = Image.fromarray((cropped * 255).astype(np.uint8)).resize(new_size, resample=Image.BILINEAR)
    canvas = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    y_offset = (IMAGE_SIZE - new_size[1]) // 2
    x_offset = (IMAGE_SIZE - new_size[0]) // 2
    canvas[y_offset : y_offset + new_size[1], x_offset : x_offset + new_size[0]] = (
        np.array(resized) > 0
    ).astype(np.uint8)
    return canvas
