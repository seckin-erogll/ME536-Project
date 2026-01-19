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
    binary = _center_of_mass_align(binary)
    footprint = morphology.disk(2)
    dilated = morphology.dilation(binary, footprint=footprint)
    closed = morphology.closing(dilated, footprint=footprint)
    return closed.astype(np.float32)


def extract_graph_features(image: np.ndarray, min_area: int = MIN_AREA) -> np.ndarray:
    processed = preprocess(image, min_area=min_area)
    return extract_graph_features_from_binary(processed)


def extract_graph_features_from_binary(binary: np.ndarray) -> np.ndarray:
    skeleton = morphology.skeletonize(binary > 0)
    coords = np.column_stack(np.where(skeleton))
    if coords.size == 0:
        return np.zeros(6, dtype=np.float32)
    skeleton_set = {tuple(coord) for coord in coords}
    neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    def neighbor_coords(point: tuple[int, int]) -> list[tuple[int, int]]:
        row, col = point
        return [
            (row + dr, col + dc)
            for dr, dc in neighbors
            if (row + dr, col + dc) in skeleton_set
        ]

    degrees = {point: len(neighbor_coords(point)) for point in skeleton_set}
    endpoints = [point for point, degree in degrees.items() if degree == 1]
    junctions = [point for point, degree in degrees.items() if degree >= 3]
    nodes = set(endpoints + junctions)
    edge_lengths: list[int] = []
    visited_steps: set[tuple[tuple[int, int], tuple[int, int]]] = set()

    for node in nodes:
        for neighbor in neighbor_coords(node):
            step = (node, neighbor)
            if step in visited_steps:
                continue
            length = 1
            prev = node
            current = neighbor
            visited_steps.add(step)
            while current not in nodes:
                next_candidates = [cand for cand in neighbor_coords(current) if cand != prev]
                if not next_candidates:
                    break
                next_point = next_candidates[0]
                visited_steps.add((current, next_point))
                prev, current = current, next_point
                length += 1
            edge_lengths.append(length)

    total_pixels = len(skeleton_set)
    num_endpoints = len(endpoints)
    num_junctions = len(junctions)
    avg_edge_length = float(np.mean(edge_lengths)) if edge_lengths else 0.0
    max_edge_length = float(np.max(edge_lengths)) if edge_lengths else 0.0
    mean_degree = float(np.mean(list(degrees.values()))) if degrees else 0.0
    return np.array(
        [total_pixels, num_endpoints, num_junctions, avg_edge_length, max_edge_length, mean_degree],
        dtype=np.float32,
    )


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


def _center_of_mass_align(binary: np.ndarray) -> np.ndarray:
    coords = np.column_stack(np.where(binary > 0))
    if coords.size == 0:
        return binary
    center_yx = coords.mean(axis=0)
    target = np.array(binary.shape, dtype=np.float32) / 2.0
    shift = np.round(target - center_yx).astype(int)
    shifted = np.roll(binary, shift=tuple(shift), axis=(0, 1))
    if shift[0] > 0:
        shifted[: shift[0], :] = 0
    elif shift[0] < 0:
        shifted[shift[0] :, :] = 0
    if shift[1] > 0:
        shifted[:, : shift[1]] = 0
    elif shift[1] < 0:
        shifted[:, shift[1] :] = 0
    return shifted
