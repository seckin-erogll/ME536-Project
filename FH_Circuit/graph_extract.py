"""Skeleton-based graph extraction for circuit symbols."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw

_SKIMAGE_AVAILABLE = importlib.util.find_spec("skimage") is not None
if _SKIMAGE_AVAILABLE:
    from skimage import morphology


@dataclass
class GraphData:
    nodes: np.ndarray
    edges: np.ndarray
    node_features: np.ndarray
    edge_features: np.ndarray
    graph_features: np.ndarray
    skeleton: np.ndarray
    node_types: List[str]
    edge_polylines: List[List[Tuple[int, int]]]


def extract_graph(
    cleaned: np.ndarray,
    debug_dir: Optional[Path] = None,
    debug_prefix: str = "sample",
    path_spacing: int = 8,
) -> GraphData:
    """Extract topology-based graph features from a cleaned binary image.

    Nodes are defined by skeleton endpoints and junctions (no corner detector + DBSCAN).
    """
    skeleton = _skeletonize(cleaned)
    degrees = _compute_degrees(skeleton)
    endpoint_mask = (skeleton == 1) & (degrees == 1)
    junction_mask = (skeleton == 1) & (degrees >= 3)

    node_coords, node_types, node_pixels = _build_nodes(endpoint_mask, junction_mask)
    node_pixel_to_index = _map_node_pixels(node_pixels)

    edges, edge_polylines = _trace_edges(skeleton, node_pixel_to_index)
    (
        dense_nodes,
        dense_types,
        dense_edges,
        dense_polylines,
        path_node_count,
        segment_lengths,
        curvature_proxy,
    ) = _insert_path_nodes(node_coords, node_types, edges, edge_polylines, path_spacing)
    node_features = _build_node_features(dense_nodes, dense_types, dense_edges, skeleton)
    edge_features = _build_edge_features(dense_polylines, skeleton.shape)
    graph_features = _build_graph_features(
        dense_types,
        dense_edges,
        edge_features,
        skeleton,
        path_node_count=path_node_count,
        segment_lengths=segment_lengths,
        curvature_proxy=curvature_proxy,
    )

    graph = GraphData(
        nodes=dense_nodes,
        edges=dense_edges,
        node_features=node_features,
        edge_features=edge_features,
        graph_features=graph_features,
        skeleton=skeleton,
        node_types=dense_types,
        edge_polylines=dense_polylines,
    )
    if debug_dir is not None:
        render_graph_overlay(graph, debug_dir, debug_prefix)
    return graph


def _skeletonize(binary: np.ndarray) -> np.ndarray:
    if _SKIMAGE_AVAILABLE:
        return morphology.skeletonize(binary > 0).astype(np.uint8)
    return _zhang_suen_thinning(binary > 0)


def _zhang_suen_thinning(binary: np.ndarray) -> np.ndarray:
    image = binary.astype(np.uint8)
    changed = True
    while changed:
        changed = False
        to_remove = []
        for step in range(2):
            for i in range(1, image.shape[0] - 1):
                for j in range(1, image.shape[1] - 1):
                    if image[i, j] == 0:
                        continue
                    neighbors = _neighbors(image, i, j)
                    n_count = sum(neighbors)
                    transitions = _transitions(neighbors)
                    if n_count < 2 or n_count > 6 or transitions != 1:
                        continue
                    if step == 0:
                        if neighbors[0] * neighbors[2] * neighbors[4] != 0:
                            continue
                        if neighbors[2] * neighbors[4] * neighbors[6] != 0:
                            continue
                    else:
                        if neighbors[0] * neighbors[2] * neighbors[6] != 0:
                            continue
                        if neighbors[0] * neighbors[4] * neighbors[6] != 0:
                            continue
                    to_remove.append((i, j))
            if to_remove:
                changed = True
                for x, y in to_remove:
                    image[x, y] = 0
            to_remove = []
    return image


def _neighbors(image: np.ndarray, i: int, j: int) -> List[int]:
    return [
        image[i - 1, j],
        image[i - 1, j + 1],
        image[i, j + 1],
        image[i + 1, j + 1],
        image[i + 1, j],
        image[i + 1, j - 1],
        image[i, j - 1],
        image[i - 1, j - 1],
    ]


def _transitions(neighbors: List[int]) -> int:
    transitions = 0
    for idx in range(len(neighbors)):
        if neighbors[idx] == 0 and neighbors[(idx + 1) % len(neighbors)] == 1:
            transitions += 1
    return transitions


def _compute_degrees(skeleton: np.ndarray) -> np.ndarray:
    padded = np.pad(skeleton, 1, mode="constant")
    degrees = np.zeros_like(skeleton, dtype=np.uint8)
    for i in range(skeleton.shape[0]):
        for j in range(skeleton.shape[1]):
            if skeleton[i, j] == 0:
                continue
            region = padded[i : i + 3, j : j + 3]
            degrees[i, j] = int(region.sum() - 1)
    return degrees


def _build_nodes(
    endpoint_mask: np.ndarray,
    junction_mask: np.ndarray,
) -> Tuple[np.ndarray, List[str], List[List[Tuple[int, int]]]]:
    node_coords: List[Tuple[int, int]] = []
    node_types: List[str] = []
    node_pixels: List[List[Tuple[int, int]]] = []

    endpoint_coords = list(zip(*np.where(endpoint_mask)))
    for coord in endpoint_coords:
        node_coords.append(coord)
        node_types.append("endpoint")
        node_pixels.append([coord])

    junction_labels, junction_clusters = _cluster_pixels(junction_mask)
    for label, pixels in junction_clusters.items():
        if label == 0 or not pixels:
            continue
        centroid = tuple(np.round(np.mean(pixels, axis=0)).astype(int))
        node_coords.append(centroid)
        node_types.append("junction")
        node_pixels.append(pixels)

    return np.array(node_coords, dtype=np.int32), node_types, node_pixels


def _cluster_pixels(mask: np.ndarray) -> Tuple[np.ndarray, Dict[int, List[Tuple[int, int]]]]:
    labels = np.zeros_like(mask, dtype=np.int32)
    clusters: Dict[int, List[Tuple[int, int]]] = {}
    label = 0
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i, j] == 0 or labels[i, j] != 0:
                continue
            label += 1
            stack = [(i, j)]
            labels[i, j] = label
            clusters[label] = []
            while stack:
                x, y = stack.pop()
                clusters[label].append((x, y))
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        if dx == 0 and dy == 0:
                            continue
                        nx, ny = x + dx, y + dy
                        if (
                            0 <= nx < mask.shape[0]
                            and 0 <= ny < mask.shape[1]
                            and mask[nx, ny] == 1
                            and labels[nx, ny] == 0
                        ):
                            labels[nx, ny] = label
                            stack.append((nx, ny))
    return labels, clusters


def _map_node_pixels(node_pixels: List[List[Tuple[int, int]]]) -> Dict[Tuple[int, int], int]:
    mapping: Dict[Tuple[int, int], int] = {}
    for idx, pixels in enumerate(node_pixels):
        for pixel in pixels:
            mapping[pixel] = idx
    return mapping


def _trace_edges(
    skeleton: np.ndarray,
    node_pixel_to_index: Dict[Tuple[int, int], int],
) -> Tuple[np.ndarray, List[List[Tuple[int, int]]]]:
    edges: List[Tuple[int, int]] = []
    polylines: List[List[Tuple[int, int]]] = []
    visited: set[Tuple[int, int]] = set()

    for node_pixel, node_idx in node_pixel_to_index.items():
        neighbors = _neighbor_coords(node_pixel, skeleton.shape)
        for neighbor in neighbors:
            if skeleton[neighbor] == 0:
                continue
            if neighbor in node_pixel_to_index:
                continue
            if neighbor in visited:
                continue
            polyline, end_node = _walk_edge(
                skeleton,
                node_pixel_to_index,
                node_pixel,
                neighbor,
                visited,
            )
            if end_node is None or end_node == node_idx:
                continue
            edge = (node_idx, end_node)
            if edge in edges or (edge[1], edge[0]) in edges:
                continue
            edges.append(edge)
            polylines.append(polyline)
    if not edges:
        return np.zeros((0, 2), dtype=np.int64), []
    return np.array(edges, dtype=np.int64), polylines


def _insert_path_nodes(
    node_coords: np.ndarray,
    node_types: List[str],
    edges: np.ndarray,
    polylines: List[List[Tuple[int, int]]],
    spacing: int,
) -> Tuple[
    np.ndarray,
    List[str],
    np.ndarray,
    List[List[Tuple[int, int]]],
    int,
    List[float],
    float,
]:
    # Sample path nodes every K pixels to densify the graph along skeleton paths.
    if edges.size == 0:
        return node_coords, node_types, edges, polylines, 0, [], 0.0
    new_nodes: List[Tuple[int, int]] = [tuple(coord) for coord in node_coords]
    new_types: List[str] = list(node_types)
    new_edges: List[Tuple[int, int]] = []
    new_polylines: List[List[Tuple[int, int]]] = []
    path_node_count = 0
    segment_lengths: List[float] = []
    curvature_samples: List[float] = []

    for edge_idx, (u, v) in enumerate(edges):
        polyline = polylines[edge_idx]
        if len(polyline) < 2:
            continue
        points = np.array(polyline, dtype=np.float32)
        diffs = np.diff(points, axis=0)
        lengths = np.linalg.norm(diffs, axis=1)
        cumulative = np.concatenate([[0.0], np.cumsum(lengths)])
        total_len = cumulative[-1]
        if total_len <= spacing:
            new_edges.append((int(u), int(v)))
            new_polylines.append(polyline)
            segment_lengths.append(float(total_len))
            continue
        sample_positions = np.arange(spacing, total_len, spacing)
        prev_node = int(u)
        for pos in sample_positions:
            idx = np.searchsorted(cumulative, pos) - 1
            idx = max(0, min(idx, len(diffs) - 1))
            t = (pos - cumulative[idx]) / max(1e-6, lengths[idx])
            interp = points[idx] + t * diffs[idx]
            new_nodes.append((int(round(interp[0])), int(round(interp[1]))))
            new_types.append("path")
            curr_node = len(new_nodes) - 1
            new_edges.append((prev_node, curr_node))
            new_polylines.append(
                [
                    (int(points[idx][0]), int(points[idx][1])),
                    (int(points[idx + 1][0]), int(points[idx + 1][1])),
                ]
            )
            segment_lengths.append(float(spacing))
            prev_node = curr_node
            path_node_count += 1
        new_edges.append((prev_node, int(v)))
        new_polylines.append(polyline)
        segment_lengths.append(float(max(1.0, total_len - sample_positions[-1])))
        if diffs.shape[0] > 1:
            angles = np.arctan2(diffs[:, 0], diffs[:, 1])
            curvature_samples.append(float(np.var(angles)))

    curvature_proxy = float(np.mean(curvature_samples)) if curvature_samples else 0.0
    return (
        np.array(new_nodes, dtype=np.int32),
        new_types,
        np.array(new_edges, dtype=np.int64),
        new_polylines,
        path_node_count,
        segment_lengths,
        curvature_proxy,
    )


def _walk_edge(
    skeleton: np.ndarray,
    node_pixel_to_index: Dict[Tuple[int, int], int],
    start_pixel: Tuple[int, int],
    next_pixel: Tuple[int, int],
    visited: set[Tuple[int, int]],
) -> Tuple[List[Tuple[int, int]], Optional[int]]:
    polyline = [start_pixel]
    prev = start_pixel
    current = next_pixel
    while True:
        polyline.append(current)
        visited.add(current)
        if current in node_pixel_to_index and current != start_pixel:
            return polyline, node_pixel_to_index[current]
        neighbors = [coord for coord in _neighbor_coords(current, skeleton.shape) if skeleton[coord] == 1]
        neighbors = [coord for coord in neighbors if coord != prev]
        if not neighbors:
            return polyline, None
        if len(neighbors) > 1:
            for coord in neighbors:
                if coord in node_pixel_to_index and coord != start_pixel:
                    return polyline, node_pixel_to_index[coord]
            return polyline, None
        prev, current = current, neighbors[0]


def _neighbor_coords(pixel: Tuple[int, int], shape: Tuple[int, int]) -> List[Tuple[int, int]]:
    neighbors = []
    x, y = pixel
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            nx, ny = x + dx, y + dy
            if 0 <= nx < shape[0] and 0 <= ny < shape[1]:
                neighbors.append((nx, ny))
    return neighbors


def _build_node_features(
    node_coords: np.ndarray,
    node_types: List[str],
    edges: np.ndarray,
    skeleton: np.ndarray,
) -> np.ndarray:
    if node_coords.size == 0:
        return np.zeros((0, 6), dtype=np.float32)
    degrees = _compute_graph_degrees(node_coords.shape[0], edges)
    features = []
    for idx, (coord, node_type) in enumerate(zip(node_coords, node_types)):
        y, x = coord
        norm_x = x / max(1, skeleton.shape[1] - 1)
        norm_y = y / max(1, skeleton.shape[0] - 1)
        degree = degrees[idx]
        endpoint = 1.0 if node_type == "endpoint" else 0.0
        junction = 1.0 if node_type == "junction" else 0.0
        direction = _local_direction((int(y), int(x)), skeleton)
        features.append([norm_x, norm_y, degree, endpoint, junction, direction])
    return np.array(features, dtype=np.float32)


def _local_direction(coord: Tuple[int, int], skeleton: np.ndarray) -> float:
    neighbors = [n for n in _neighbor_coords(coord, skeleton.shape) if skeleton[n] == 1]
    if not neighbors:
        return 0.0
    vectors = np.array([[n[1] - coord[1], n[0] - coord[0]] for n in neighbors], dtype=np.float32)
    mean_vec = vectors.mean(axis=0)
    return float(np.arctan2(mean_vec[1], mean_vec[0]))


def _compute_graph_degrees(num_nodes: int, edges: np.ndarray) -> np.ndarray:
    degrees = np.zeros(num_nodes, dtype=np.int32)
    for u, v in edges:
        degrees[u] += 1
        degrees[v] += 1
    return degrees


def _build_edge_features(
    polylines: List[List[Tuple[int, int]]],
    shape: Tuple[int, int],
) -> np.ndarray:
    features = []
    max_dim = max(shape)
    for polyline in polylines:
        points = np.array(polyline, dtype=np.float32)
        if points.shape[0] < 2:
            length = 0.0
            angle = 0.0
            curvature = 0.0
        else:
            diffs = np.diff(points, axis=0)
            lengths = np.linalg.norm(diffs, axis=1)
            length = float(lengths.sum()) / max_dim
            angles = np.arctan2(diffs[:, 0], diffs[:, 1])
            angle = float(_circular_mean(angles))
            curvature = float(np.var(angles)) if angles.size > 1 else 0.0
        features.append([length, angle, curvature])
    if not features:
        return np.zeros((0, 3), dtype=np.float32)
    return np.array(features, dtype=np.float32)


def _circular_mean(angles: np.ndarray) -> float:
    return float(np.arctan2(np.mean(np.sin(angles)), np.mean(np.cos(angles))))


def _build_graph_features(
    node_types: List[str],
    edges: np.ndarray,
    edge_features: np.ndarray,
    skeleton: np.ndarray,
    path_node_count: int,
    segment_lengths: List[float],
    curvature_proxy: float,
) -> np.ndarray:
    num_nodes = len(node_types)
    num_edges = int(edges.shape[0])
    degrees = _compute_graph_degrees(num_nodes, edges) if num_nodes else np.array([])
    degree_hist = np.zeros(4, dtype=np.float32)
    for degree in degrees:
        idx = min(degree, 4) - 1
        if idx >= 0:
            degree_hist[idx] += 1
    if edge_features.size == 0:
        length_stats = np.zeros(4, dtype=np.float32)
    else:
        lengths = edge_features[:, 0]
        length_stats = np.array(
            [lengths.min(), lengths.mean(), lengths.max(), lengths.std()],
            dtype=np.float32,
        )
    angles = edge_features[:, 1] if edge_features.size else np.array([])
    angle_hist, _ = np.histogram(angles, bins=8, range=(-np.pi, np.pi))
    angle_hist = angle_hist.astype(np.float32)
    components = _count_components(skeleton)
    cycle_indicator = num_edges - num_nodes + components
    endpoint_count = float(node_types.count("endpoint"))
    junction_count = float(node_types.count("junction"))
    avg_segment_length = float(np.mean(segment_lengths)) if segment_lengths else 0.0
    graph_feat = np.concatenate(
        [
            np.array(
                [num_nodes, num_edges, float(path_node_count), avg_segment_length, curvature_proxy],
                dtype=np.float32,
            ),
            degree_hist,
            length_stats,
            angle_hist,
            np.array([components, cycle_indicator, endpoint_count, junction_count], dtype=np.float32),
        ]
    )
    return graph_feat.astype(np.float32)


def _count_components(skeleton: np.ndarray) -> float:
    visited = np.zeros_like(skeleton, dtype=bool)
    count = 0
    for i in range(skeleton.shape[0]):
        for j in range(skeleton.shape[1]):
            if skeleton[i, j] == 0 or visited[i, j]:
                continue
            count += 1
            stack = [(i, j)]
            visited[i, j] = True
            while stack:
                x, y = stack.pop()
                for nx, ny in _neighbor_coords((x, y), skeleton.shape):
                    if skeleton[nx, ny] == 1 and not visited[nx, ny]:
                        visited[nx, ny] = True
                        stack.append((nx, ny))
    return float(count)


def render_graph_overlay(graph: GraphData, output_dir: Path, prefix: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    base = (graph.skeleton * 255).astype(np.uint8)
    image = Image.fromarray(base).convert("RGB")
    draw = ImageDraw.Draw(image)
    for polyline in graph.edge_polylines:
        if len(polyline) < 2:
            continue
        points = [(p[1], p[0]) for p in polyline]
        draw.line(points, fill=(80, 180, 255), width=1)
    for coord, node_type in zip(graph.nodes, graph.node_types):
        y, x = coord
        if node_type == "endpoint":
            color = (255, 80, 80)
            radius = 2
        elif node_type == "junction":
            color = (80, 255, 120)
            radius = 2
        else:
            color = (255, 60, 60)
            radius = 1
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=color, width=2)
    image.save(output_dir / f"{prefix}_graph.png")
