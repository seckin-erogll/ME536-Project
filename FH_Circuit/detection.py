"""Component detection utilities for hand-drawn circuits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from PIL import Image
from skimage import feature, filters, measure, morphology, transform
from skimage.draw import line as draw_line

from FH_Circuit.classify import classify_sketch, TwoStageArtifacts


@dataclass(frozen=True)
class ComponentRegion:
    bbox: tuple[int, int, int, int]
    crop: np.ndarray


def detect_component_regions(
    image: np.ndarray,
    *,
    min_area: int = 200,
    line_length: int = 40,
    line_gap: int = 5,
    line_thickness: int = 3,
    padding: int = 8,
) -> list[ComponentRegion]:
    if image.ndim != 2:
        raise ValueError("Expected a grayscale image.")
    if np.max(image) == 0:
        return []
    threshold = filters.threshold_otsu(image)
    binary = image > threshold
    if binary.sum() < min_area:
        return []

    edges = feature.canny(binary.astype(float), sigma=1.2)
    lines = transform.probabilistic_hough_line(
        edges,
        threshold=10,
        line_length=line_length,
        line_gap=line_gap,
    )
    wire_mask = np.zeros_like(binary, dtype=bool)
    for (x0, y0), (x1, y1) in lines:
        rr, cc = draw_line(int(y0), int(x0), int(y1), int(x1))
        wire_mask[rr, cc] = True
    if wire_mask.any():
        wire_mask = morphology.dilation(wire_mask, footprint=morphology.disk(line_thickness))

    component_mask = binary & ~wire_mask
    labeled = measure.label(component_mask)
    filtered = np.zeros_like(labeled, dtype=np.int32)
    next_label = 1
    for region in measure.regionprops(labeled):
        if region.area < min_area:
            continue
        coords = region.coords
        filtered[coords[:, 0], coords[:, 1]] = next_label
        next_label += 1
    labeled = filtered
    regions = measure.regionprops(labeled)
    detections: list[ComponentRegion] = []

    for region in regions:
        min_row, min_col, max_row, max_col = region.bbox
        min_row = max(min_row - padding, 0)
        min_col = max(min_col - padding, 0)
        max_row = min(max_row + padding, image.shape[0])
        max_col = min(max_col + padding, image.shape[1])
        crop = image[min_row:max_row, min_col:max_col]
        if crop.size == 0:
            continue
        detections.append(ComponentRegion(bbox=(min_row, min_col, max_row, max_col), crop=crop))

    return detections


def classify_component_with_rotations(
    artifacts: TwoStageArtifacts,
    crop: np.ndarray,
    rotations: Iterable[int] = (0, 90, 180, 270),
) -> tuple[str, int]:
    if crop.size == 0:
        return "Empty crop", 0
    best_result = "Novelty detected: unknown component."
    best_rotation = 0
    for rotation in rotations:
        rotated = Image.fromarray(crop).rotate(rotation, expand=True, fillcolor=0)
        resized = rotated.resize((64, 64), resample=Image.BILINEAR)
        sketch = np.array(resized)
        result = classify_sketch(artifacts, sketch)
        if result.startswith("Detected: "):
            return result, rotation
        if result.startswith("Ambiguity") and not best_result.startswith("Detected"):
            best_result = result
            best_rotation = rotation
        elif best_result.startswith("Novelty"):
            best_result = result
            best_rotation = rotation
    return best_result, best_rotation
