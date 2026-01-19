"""Circuit analysis utilities (symbol candidates + wire graph visualization).

Usage (CLI via main.py):
    python main.py analyze_circuit --image path/to/img.png --output ./artifacts/debug_circuit
"""

from __future__ import annotations

import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from PIL import Image, ImageDraw, ImageFilter

from FH_Circuit.graph_extract import (
    extract_graph,
    polyline_curvature_angles,
    skeletonize_image,
)
from FH_Circuit.preprocess import binarize_image

_SKIMAGE_AVAILABLE = importlib.util.find_spec("skimage") is not None
if _SKIMAGE_AVAILABLE:
    from skimage import filters, measure, morphology
    from skimage.transform import probabilistic_hough_line


@dataclass(frozen=True)
class CircuitParams:
    wire_edge_min_length: int = 80
    wire_edge_max_curvature_deg: float = 15.0
    corner_angle_threshold_deg: float = 35.0
    rotated_threshold_deg: float = 10.0
    symbol_bbox_min_area: int = 120
    wire_mask_dilate: int = 2
    wire_border_margin: int = 5
    symbol_bbox_padding: int = 6


def analyze_circuit(
    image: np.ndarray | Image.Image,
    params: CircuitParams | None = None,
    debug_dir: Path | None = Path("artifacts/debug_circuit"),
) -> dict[str, Any]:
    params = params or CircuitParams()
    gray = _ensure_grayscale(image)
    smoothed = _denoise(gray)
    binary = binarize_image(smoothed)
    binary = _binary_close(binary, radius=1)

    skeleton = skeletonize_image(binary)
    graph = extract_graph(skeleton)
    wire_mask = _wire_mask_from_graph(graph, binary.shape, params)
    symbols_mask = np.logical_and(binary > 0, np.logical_not(wire_mask))
    symbols_mask = _binary_close(symbols_mask.astype(np.uint8), radius=1)

    components = _connected_components(symbols_mask, min_area=params.symbol_bbox_min_area)
    if not components and _SKIMAGE_AVAILABLE:
        hough_mask = _wire_mask_from_hough(binary)
        symbols_mask = np.logical_and(binary > 0, np.logical_not(hough_mask))
        components = _connected_components(symbols_mask, min_area=params.symbol_bbox_min_area)

    candidates = []
    for component in components:
        x0, y0, x1, y1 = component["bbox"]
        x0, y0, x1, y1 = _pad_bbox(x0, y0, x1, y1, binary.shape, params.symbol_bbox_padding)
        crop = binary[y0 : y1 + 1, x0 : x1 + 1]
        angle = _pca_angle(crop)
        terminals = _extract_terminals(crop)
        terminals_global = [(x0 + x, y0 + y) for x, y in terminals]
        candidates.append(
            {
                "bbox": (x0, y0, x1, y1),
                "crop": crop,
                "rotation_angle_deg": angle,
                "terminals": terminals_global,
            }
        )

    wire_corners = _detect_wire_corners(graph["edges"], params.corner_angle_threshold_deg)
    result = {
        "symbol_candidates": candidates,
        "wire_graph": graph,
        "wire_corners": wire_corners,
    }

    if debug_dir is not None:
        _write_debug_images(debug_dir, gray, binary, skeleton, wire_mask, symbols_mask, result)
        _write_debug_json(debug_dir, result)

    return result


def serialize_analysis(result: dict[str, Any], include_crops: bool = False) -> dict[str, Any]:
    def serialize_point(point: tuple[int, int]) -> list[int]:
        return [int(point[0]), int(point[1])]

    symbols = []
    for candidate in result.get("symbol_candidates", []):
        entry = {
            "bbox": [int(v) for v in candidate["bbox"]],
            "rotation_angle_deg": float(candidate["rotation_angle_deg"]),
            "terminals": [serialize_point(p) for p in candidate["terminals"]],
        }
        if include_crops and "crop" in candidate:
            entry["crop"] = candidate["crop"].astype(int).tolist()
        symbols.append(entry)

    nodes = []
    for node in result.get("wire_graph", {}).get("nodes", []):
        nodes.append(
            {
                "id": int(node["id"]),
                "coord": serialize_point(node["coord"]),
                "degree": int(node["degree"]),
                "kind": node["kind"],
            }
        )

    edges = []
    for edge in result.get("wire_graph", {}).get("edges", []):
        edges.append(
            {
                "start": int(edge["start"]),
                "end": int(edge["end"]),
                "length": int(edge["length"]),
                "points": [serialize_point(p) for p in edge["points"]],
            }
        )

    return {
        "symbol_candidates": symbols,
        "wire_graph": {"nodes": nodes, "edges": edges},
        "wire_corners": [serialize_point(p) for p in result.get("wire_corners", [])],
    }


def _ensure_grayscale(image: np.ndarray | Image.Image) -> np.ndarray:
    if isinstance(image, Image.Image):
        image = image.convert("L")
        return np.array(image)
    if image.ndim == 3:
        return (0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]).astype(
            np.uint8
        )
    return image.astype(np.uint8)


def _denoise(image: np.ndarray) -> np.ndarray:
    pil_image = Image.fromarray(image)
    blurred = pil_image.filter(ImageFilter.GaussianBlur(radius=0.8))
    return np.array(blurred)


def _binary_close(binary: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return binary
    if _SKIMAGE_AVAILABLE:
        return morphology.binary_closing(binary > 0, morphology.disk(radius))
    dilated = _binary_dilation(binary > 0, radius)
    return _binary_erosion(dilated, radius)


def _binary_dilation(binary: np.ndarray, radius: int) -> np.ndarray:
    padded = np.pad(binary, radius, mode="constant")
    window = _sliding_window_view(padded, (radius * 2 + 1, radius * 2 + 1))
    return window.max(axis=(-1, -2))


def _binary_erosion(binary: np.ndarray, radius: int) -> np.ndarray:
    padded = np.pad(binary, radius, mode="constant")
    window = _sliding_window_view(padded, (radius * 2 + 1, radius * 2 + 1))
    return window.min(axis=(-1, -2))


def _sliding_window_view(array: np.ndarray, window_shape: tuple[int, int]) -> np.ndarray:
    return sliding_window_view(array, window_shape)


def _wire_mask_from_graph(
    graph: dict[str, list[dict]],
    shape: tuple[int, int],
    params: CircuitParams,
) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for edge in graph.get("edges", []):
        length = edge["length"]
        angles = polyline_curvature_angles(edge["points"])
        max_curv = max(angles) if angles else 0.0
        start_point = edge["points"][0]
        end_point = edge["points"][-1]
        near_border = _is_near_border(start_point, shape, params.wire_border_margin) or _is_near_border(
            end_point, shape, params.wire_border_margin
        )
        if length >= params.wire_edge_min_length and max_curv <= params.wire_edge_max_curvature_deg:
            if near_border or length >= int(params.wire_edge_min_length * 1.5):
                for x, y in edge["points"]:
                    mask[y, x] = True
    if params.wire_mask_dilate > 0:
        mask = _binary_dilation(mask, params.wire_mask_dilate)
    return mask


def _wire_mask_from_hough(binary: np.ndarray) -> np.ndarray:
    if not _SKIMAGE_AVAILABLE:
        return np.zeros_like(binary, dtype=bool)
    edges = binary > 0
    lines = probabilistic_hough_line(edges, threshold=10, line_length=50, line_gap=5)
    mask = np.zeros_like(binary, dtype=bool)
    for (x0, y0), (x1, y1) in lines:
        points = _interpolate_line((x0, y0), (x1, y1))
        for x, y in points:
            mask[y, x] = True
    return _binary_dilation(mask, 2)


def _connected_components(binary: np.ndarray, min_area: int) -> list[dict[str, Any]]:
    if _SKIMAGE_AVAILABLE:
        labeled = measure.label(binary > 0, connectivity=2)
        components = []
        for label in range(1, labeled.max() + 1):
            coords = np.column_stack(np.where(labeled == label))
            if coords.size == 0:
                continue
            area = coords.shape[0]
            if area < min_area:
                continue
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)
            components.append({"bbox": (x_min, y_min, x_max, y_max), "area": area})
        return components
    return _connected_components_fallback(binary > 0, min_area)


def _connected_components_fallback(binary: np.ndarray, min_area: int) -> list[dict[str, Any]]:
    height, width = binary.shape
    labels = np.zeros_like(binary, dtype=int)
    label = 0
    components = []
    for y in range(height):
        for x in range(width):
            if not binary[y, x] or labels[y, x] != 0:
                continue
            label += 1
            stack = [(y, x)]
            labels[y, x] = label
            coords = []
            while stack:
                cy, cx = stack.pop()
                coords.append((cy, cx))
                for ny in range(cy - 1, cy + 2):
                    for nx in range(cx - 1, cx + 2):
                        if (
                            0 <= ny < height
                            and 0 <= nx < width
                            and binary[ny, nx]
                            and labels[ny, nx] == 0
                        ):
                            labels[ny, nx] = label
                            stack.append((ny, nx))
            if len(coords) < min_area:
                continue
            ys, xs = zip(*coords)
            components.append(
                {
                    "bbox": (min(xs), min(ys), max(xs), max(ys)),
                    "area": len(coords),
                }
            )
    return components


def _pad_bbox(
    x0: int, y0: int, x1: int, y1: int, shape: tuple[int, int], padding: int
) -> tuple[int, int, int, int]:
    height, width = shape
    x0 = max(0, x0 - padding)
    y0 = max(0, y0 - padding)
    x1 = min(width - 1, x1 + padding)
    y1 = min(height - 1, y1 + padding)
    return x0, y0, x1, y1


def _pca_angle(binary: np.ndarray) -> float:
    coords = np.column_stack(np.where(binary > 0))
    if coords.size == 0:
        return 0.0
    points = coords[:, ::-1].astype(np.float32)
    mean = points.mean(axis=0)
    centered = points - mean
    cov = np.cov(centered, rowvar=False)
    if cov.shape == ():
        return 0.0
    eigvals, eigvecs = np.linalg.eigh(cov)
    principal = eigvecs[:, np.argmax(eigvals)]
    angle = float(np.degrees(np.arctan2(principal[1], principal[0])))
    if angle > 90:
        angle -= 180
    if angle < -90:
        angle += 180
    return angle


def _extract_terminals(binary: np.ndarray) -> list[tuple[int, int]]:
    skeleton = skeletonize_image(binary)
    coords = np.column_stack(np.where(skeleton))
    if coords.size == 0:
        return []
    skeleton_set = {tuple(coord) for coord in coords}
    endpoints = [
        point
        for point in skeleton_set
        if _neighbor_count(point, skeleton_set) == 1
    ]
    if len(endpoints) < 2:
        return []
    endpoint_points = [(pt[1], pt[0]) for pt in endpoints]
    return _pick_terminal_pair(endpoint_points, binary.shape)


def _neighbor_count(point: tuple[int, int], skeleton_set: set[tuple[int, int]]) -> int:
    count = 0
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            if (point[0] + dy, point[1] + dx) in skeleton_set:
                count += 1
    return count


def _pick_terminal_pair(endpoints: list[tuple[int, int]], shape: tuple[int, int]) -> list[tuple[int, int]]:
    best_pair: tuple[tuple[int, int], tuple[int, int]] | None = None
    best_score = -1.0
    for i in range(len(endpoints)):
        for j in range(i + 1, len(endpoints)):
            p1 = endpoints[i]
            p2 = endpoints[j]
            dist = np.hypot(p1[0] - p2[0], p1[1] - p2[1])
            border_penalty = _border_distance(p1, shape) + _border_distance(p2, shape)
            score = dist - 0.5 * border_penalty
            if score > best_score:
                best_score = score
                best_pair = (p1, p2)
    if best_pair is None:
        return []
    return [best_pair[0], best_pair[1]]


def _border_distance(point: tuple[int, int], shape: tuple[int, int]) -> float:
    width, height = shape[1], shape[0]
    return float(min(point[0], point[1], width - 1 - point[0], height - 1 - point[1]))


def _detect_wire_corners(edges: list[dict], threshold_deg: float) -> list[tuple[int, int]]:
    corners: list[tuple[int, int]] = []
    for edge in edges:
        points = edge["points"]
        if len(points) < 3:
            continue
        for idx in range(1, len(points) - 1):
            p0 = np.array(points[idx - 1], dtype=np.float32)
            p1 = np.array(points[idx], dtype=np.float32)
            p2 = np.array(points[idx + 1], dtype=np.float32)
            v1 = p1 - p0
            v2 = p2 - p1
            if np.linalg.norm(v1) < 1e-6 or np.linalg.norm(v2) < 1e-6:
                continue
            cos_angle = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
            cos_angle = float(np.clip(cos_angle, -1.0, 1.0))
            angle = float(np.degrees(np.arccos(cos_angle)))
            if angle >= threshold_deg:
                corners.append((int(p1[0]), int(p1[1])))
    return _dedupe_points(corners, min_distance=3)


def _dedupe_points(points: list[tuple[int, int]], min_distance: int) -> list[tuple[int, int]]:
    if not points:
        return []
    kept: list[tuple[int, int]] = []
    for point in points:
        if all(np.hypot(point[0] - other[0], point[1] - other[1]) >= min_distance for other in kept):
            kept.append(point)
    return kept


def _is_near_border(point: tuple[int, int], shape: tuple[int, int], margin: int) -> bool:
    x, y = point
    height, width = shape
    return x < margin or y < margin or x >= width - margin or y >= height - margin


def _interpolate_line(p0: tuple[int, int], p1: tuple[int, int]) -> list[tuple[int, int]]:
    x0, y0 = p0
    x1, y1 = p1
    length = int(max(abs(x1 - x0), abs(y1 - y0)))
    points = []
    for step in range(length + 1):
        t = step / max(1, length)
        x = int(round(x0 + (x1 - x0) * t))
        y = int(round(y0 + (y1 - y0) * t))
        points.append((x, y))
    return points


def _write_debug_images(
    debug_dir: Path,
    gray: np.ndarray,
    binary: np.ndarray,
    skeleton: np.ndarray,
    wire_mask: np.ndarray,
    symbols_mask: np.ndarray,
    result: dict[str, Any],
) -> None:
    debug_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(gray).save(debug_dir / "grayscale.png")
    Image.fromarray((binary > 0).astype(np.uint8) * 255).save(debug_dir / "binarized.png")
    Image.fromarray(skeleton.astype(np.uint8) * 255).save(debug_dir / "skeleton.png")
    Image.fromarray(wire_mask.astype(np.uint8) * 255).save(debug_dir / "wire_mask.png")
    Image.fromarray(symbols_mask.astype(np.uint8) * 255).save(debug_dir / "symbols_mask.png")
    overlay = _draw_overlay(gray, result)
    overlay.save(debug_dir / "overlay.png")


def _write_debug_json(debug_dir: Path, result: dict[str, Any]) -> None:
    output = serialize_analysis(result, include_crops=False)
    (debug_dir / "analysis.json").write_text(json.dumps(output, indent=2))


def _draw_overlay(gray: np.ndarray, result: dict[str, Any]) -> Image.Image:
    rgb = Image.fromarray(gray).convert("RGB")
    draw = ImageDraw.Draw(rgb)
    for candidate in result.get("symbol_candidates", []):
        x0, y0, x1, y1 = candidate["bbox"]
        draw.rectangle([x0, y0, x1, y1], outline="yellow", width=2)
        for x, y in candidate["terminals"]:
            draw.ellipse([x - 3, y - 3, x + 3, y + 3], outline="cyan", width=2)
    for node in result.get("wire_graph", {}).get("nodes", []):
        x, y = node["coord"]
        color = "lime" if node["kind"] == "junction" else "orange"
        draw.ellipse([x - 3, y - 3, x + 3, y + 3], outline=color, width=2)
    for corner in result.get("wire_corners", []):
        x, y = corner
        draw.rectangle([x - 2, y - 2, x + 2, y + 2], outline="red", width=2)
    return rgb
