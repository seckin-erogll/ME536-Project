"""Skeleton graph extraction utilities."""

from __future__ import annotations

import importlib.util
from typing import Iterable

import numpy as np

_SKIMAGE_AVAILABLE = importlib.util.find_spec("skimage") is not None
if _SKIMAGE_AVAILABLE:
    from skimage import morphology

NEIGHBOR_OFFSETS = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]


def skeletonize_image(binary: np.ndarray) -> np.ndarray:
    if _SKIMAGE_AVAILABLE:
        return morphology.skeletonize(binary > 0)
    return binary > 0


def extract_graph(skeleton: np.ndarray) -> dict[str, list[dict]]:
    """Return nodes and edges for an 8-connected skeleton image."""
    coords = np.column_stack(np.where(skeleton))
    if coords.size == 0:
        return {"nodes": [], "edges": []}
    skeleton_set = {tuple(coord) for coord in coords}
    degrees = {point: _neighbor_count(point, skeleton_set) for point in skeleton_set}
    nodes = [point for point, degree in degrees.items() if degree == 1 or degree >= 3]
    node_map = {point: idx for idx, point in enumerate(nodes)}
    node_entries = [
        {
            "id": node_map[point],
            "coord": (int(point[1]), int(point[0])),
            "degree": degrees[point],
            "kind": "endpoint" if degrees[point] == 1 else "junction",
        }
        for point in nodes
    ]

    edges: list[dict] = []
    visited_steps: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    for node in nodes:
        for neighbor in _neighbor_coords(node, skeleton_set):
            if (node, neighbor) in visited_steps:
                continue
            path = [node, neighbor]
            prev = node
            current = neighbor
            visited_steps.add((node, neighbor))
            while current not in node_map:
                candidates = [cand for cand in _neighbor_coords(current, skeleton_set) if cand != prev]
                if not candidates:
                    break
                next_point = candidates[0]
                visited_steps.add((current, next_point))
                prev, current = current, next_point
                path.append(current)
            if current not in node_map or current == node:
                continue
            edges.append(
                {
                    "start": node_map[node],
                    "end": node_map[current],
                    "points": [(int(col), int(row)) for row, col in path],
                    "length": len(path) - 1,
                }
            )
    return {"nodes": node_entries, "edges": edges}


def _neighbor_coords(point: tuple[int, int], skeleton_set: set[tuple[int, int]]) -> list[tuple[int, int]]:
    row, col = point
    return [
        (row + dr, col + dc)
        for dr, dc in NEIGHBOR_OFFSETS
        if (row + dr, col + dc) in skeleton_set
    ]


def _neighbor_count(point: tuple[int, int], skeleton_set: set[tuple[int, int]]) -> int:
    return len(_neighbor_coords(point, skeleton_set))


def polyline_curvature_angles(points: Iterable[tuple[int, int]]) -> list[float]:
    coords = np.array(list(points), dtype=np.float32)
    if coords.shape[0] < 3:
        return []
    angles: list[float] = []
    for idx in range(1, len(coords) - 1):
        v1 = coords[idx] - coords[idx - 1]
        v2 = coords[idx + 1] - coords[idx]
        if np.linalg.norm(v1) < 1e-6 or np.linalg.norm(v2) < 1e-6:
            continue
        cos_angle = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        cos_angle = float(np.clip(cos_angle, -1.0, 1.0))
        angle = float(np.degrees(np.arccos(cos_angle)))
        angles.append(angle)
    return angles
