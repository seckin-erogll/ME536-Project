"""Image preprocessing utilities."""

from __future__ import annotations

import math

import cv2
import numpy as np
from PIL import Image
from numpy.lib.stride_tricks import sliding_window_view
from skimage.morphology import skeletonize

from FH_Circuit.config import IMAGE_SIZE, MIN_AREA


def preprocess(image: np.ndarray, min_area: int = MIN_AREA) -> np.ndarray:
    thresh = _otsu_threshold(image)
    binary = (image > thresh).astype(np.uint8)
    if binary.sum() < min_area:
        raise ValueError("Noise detected: sketch too small.")
    binary = _normalize_to_canvas(binary)
    binary = _center_of_mass_align(binary)
    footprint = _disk_footprint(2)
    dilated = _binary_dilation(binary, footprint)
    closed = _binary_closing(dilated, footprint)
    return closed.astype(np.float32)


def estimate_stroke_width(binary_img: np.ndarray) -> float:
    """Estimate stroke width using a distance transform sampled on the skeleton."""
    binary = (binary_img > 0).astype(np.uint8)
    if binary.sum() == 0:
        return 1.0
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 3)
    skel = skeletonize_binary(binary_img)
    skel_mask = skel > 0
    samples = dist[skel_mask]
    if samples.size == 0:
        samples = dist[binary > 0]
    if samples.size == 0:
        return 1.0
    stroke_w = max(1.0, float(2.0 * np.median(samples)))
    return stroke_w


def skeletonize_binary(binary_img: np.ndarray) -> np.ndarray:
    """Skeletonize a binary drawing, returning a uint8 mask in {0, 255}."""
    binary = (binary_img > 0).astype(np.uint8) * 255
    if cv2.countNonZero(binary) == 0:
        return np.zeros_like(binary, dtype=np.uint8)
    # A light closing helps bridge tiny sketch gaps before skeletonization.
    kernel = np.ones((3, 3), np.uint8)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    skel_bool = skeletonize(closed > 0)
    return skel_bool.astype(np.uint8) * 255


def segment_symbols(binary_img: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Segment circuit symbols by removing long, straight wire edges."""
    binary = (binary_img > 0).astype(np.uint8) * 255
    if cv2.countNonZero(binary) == 0:
        return []

    stroke_w = estimate_stroke_width(binary)
    skel = skeletonize_binary(binary)
    skel01 = (skel > 0).astype(np.uint8)

    nodes_mask, degree = _find_nodes(skel01)
    node_points = np.column_stack(np.where(nodes_mask))

    # If there are no skeleton nodes, fall back to the original binary mask.
    if node_points.size == 0:
        return _components_from_mask(binary, stroke_w)

    edges = _trace_edges(skel01, nodes_mask, degree)
    if not edges:
        return _components_from_mask(binary, stroke_w)

    density_map = _density_map(binary, stroke_w)
    wire_mask = _wire_mask_from_edges(edges, nodes_mask, density_map, stroke_w)

    symbol_mask = np.zeros_like(binary, dtype=np.uint8)
    for edge in edges:
        if edge["is_wire"]:
            continue
        for y, x in edge["pixels"]:
            symbol_mask[y, x] = 255

    # Reinflate skeleton strokes using the estimated stroke width.
    recon_k = max(3, int(round(stroke_w)))
    if recon_k % 2 == 0:
        recon_k += 1
    recon_kernel = np.ones((recon_k, recon_k), np.uint8)
    symbol_mask = cv2.dilate(symbol_mask, recon_kernel, iterations=1)
    wire_mask = cv2.dilate(wire_mask, recon_kernel, iterations=1)

    # Clamp to the original drawing and remove the thickened wire regions.
    symbol_mask = cv2.bitwise_and(symbol_mask, binary)
    symbol_mask[wire_mask > 0] = 0

    return _components_from_mask(symbol_mask, stroke_w)


def _find_nodes(skel01: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    degree = cv2.filter2D(skel01, -1, kernel, borderType=cv2.BORDER_CONSTANT)
    nodes_mask = (skel01 == 1) & (degree != 2)

    # Promote sharp bends (degree-2 corners) to nodes so loops split into edges.
    corner_mask = np.zeros_like(nodes_mask, dtype=bool)
    corner_cos = math.cos(math.radians(30.0))
    for y, x in np.column_stack(np.where((skel01 == 1) & (degree == 2))):
        neigh = _iter_neighbors(skel01, int(y), int(x))
        if len(neigh) != 2:
            continue
        (y1, x1), (y2, x2) = neigh
        v1 = np.array([y1 - y, x1 - x], dtype=np.float32)
        v2 = np.array([y2 - y, x2 - x], dtype=np.float32)
        denom = float(np.linalg.norm(v1) * np.linalg.norm(v2))
        if denom <= 1e-6:
            continue
        dot = float(np.dot(v1, v2) / denom)
        if dot > -corner_cos:
            corner_mask[int(y), int(x)] = True

    nodes_mask = nodes_mask | corner_mask
    return nodes_mask, degree


def _trace_edges(skel01: np.ndarray, nodes_mask: np.ndarray, degree: np.ndarray) -> list[dict]:
    node_points = np.column_stack(np.where(nodes_mask))

    visited: set[tuple[int, int, int, int]] = set()
    edges: list[dict] = []

    for y0, x0 in node_points:
        for ny, nx in _iter_neighbors(skel01, int(y0), int(x0)):
            key = (int(y0), int(x0), int(ny), int(nx))
            if key in visited:
                continue
            visited.add(key)
            visited.add((int(ny), int(nx), int(y0), int(x0)))

            pixels: list[tuple[int, int]] = [(int(y0), int(x0)), (int(ny), int(nx))]
            prev = (int(y0), int(x0))
            curr = (int(ny), int(nx))

            while not nodes_mask[curr] and degree[curr] == 2:
                next_candidates = [p for p in _iter_neighbors(skel01, *curr) if p != prev]
                if not next_candidates:
                    break
                nxt = next_candidates[0]
                visited.add((curr[0], curr[1], nxt[0], nxt[1]))
                visited.add((nxt[0], nxt[1], curr[0], curr[1]))
                pixels.append(nxt)
                prev, curr = curr, nxt

            edge = {
                "start": (int(y0), int(x0)),
                "end": curr,
                "pixels": pixels,
                "length": float(len(pixels)),
                "is_wire": False,
            }
            edges.append(edge)

    return edges


def _iter_neighbors(skel01: np.ndarray, y: int, x: int) -> list[tuple[int, int]]:
    neighbors = [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    ]
    height, width = skel01.shape
    out: list[tuple[int, int]] = []
    for dy, dx in neighbors:
        ny, nx = y + dy, x + dx
        if 0 <= ny < height and 0 <= nx < width and skel01[ny, nx]:
            out.append((ny, nx))
    return out


def _density_map(binary: np.ndarray, stroke_w: float) -> np.ndarray:
    binary_float = (binary > 0).astype(np.float32)
    # Use a wider window so a lone wire has low density while dense symbols stay high.
    k = max(25, int(round(8 * stroke_w)))
    if k % 2 == 0:
        k += 1
    return cv2.blur(binary_float, (k, k))


def _wire_mask_from_edges(
    edges: list[dict], nodes_mask: np.ndarray, density_map: np.ndarray, stroke_w: float
) -> np.ndarray:
    height, width = nodes_mask.shape
    node_mask_u8 = nodes_mask.astype(np.uint8)
    protect_radius = max(2.0, 2.0 * stroke_w)
    if node_mask_u8.any():
        node_dt = cv2.distanceTransform((node_mask_u8 == 0).astype(np.uint8), cv2.DIST_L2, 3)
        protect_mask = node_dt <= protect_radius
    else:
        protect_mask = np.zeros_like(node_mask_u8, dtype=bool)

    # Scale wire thresholds by the estimated stroke width to reduce hand-tuning.
    l_min = max(40.0, 20.0 * stroke_w)
    tau_max = 1.20
    # With the larger density window, this mainly guards dense symbol interiors.
    density_thresh = 0.18

    wire_mask = np.zeros((height, width), dtype=np.uint8)

    for edge in edges:
        start = edge["start"]
        end = edge["end"]
        dy = float(end[0] - start[0])
        dx = float(end[1] - start[1])
        d = math.hypot(dx, dy)
        l = edge["length"]
        tau = l / (d + 1e-5)

        ys = np.fromiter((p[0] for p in edge["pixels"]), dtype=np.int32)
        xs = np.fromiter((p[1] for p in edge["pixels"]), dtype=np.int32)
        mean_density = float(density_map[ys, xs].mean()) if ys.size else 0.0
        near_dense_region = mean_density > density_thresh

        is_wire = (l > l_min) and (tau < tau_max) and (not near_dense_region)
        edge["is_wire"] = bool(is_wire)
        if not is_wire:
            continue

        for y, x in edge["pixels"]:
            if protect_mask[y, x]:
                continue
            wire_mask[y, x] = 255

    return wire_mask


def _components_from_mask(mask: np.ndarray, stroke_w: float) -> list[tuple[int, int, int, int]]:
    mask_u8 = (mask > 0).astype(np.uint8) * 255
    if cv2.countNonZero(mask_u8) == 0:
        return []

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8)
    image_area = mask_u8.shape[0] * mask_u8.shape[1]
    a_min = max(10.0, 15.0 * (stroke_w ** 2))

    components: list[dict] = []
    for label in range(1, num_labels):
        area = float(stats[label, cv2.CC_STAT_AREA])
        if area < a_min:
            continue
        comp_mask = labels == label
        ys, xs = np.where(comp_mask)
        if xs.size == 0:
            continue
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        w = x1 - x0 + 1
        h = y1 - y0 + 1
        orientation = _component_orientation(xs, ys)
        linear = _is_linear_blob(w, h)
        components.append(
            {
                "pixels": np.column_stack((ys, xs)),
                "bbox": (x0, y0, w, h),
                "area": float(len(xs)),
                "theta": orientation,
                "linear": linear,
            }
        )

    merged = _merge_components(components, stroke_w, image_area)
    return [comp["bbox"] for comp in merged]


def _component_orientation(xs: np.ndarray, ys: np.ndarray) -> float:
    if xs.size < 2:
        return 0.0
    pts = np.column_stack((xs.astype(np.float32), ys.astype(np.float32)))
    vx, vy, _, _ = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)
    vx_f = float(vx[0]) if np.ndim(vx) else float(vx)
    vy_f = float(vy[0]) if np.ndim(vy) else float(vy)
    theta = math.degrees(math.atan2(vy_f, vx_f))
    theta = theta % 180.0
    return theta


def _is_linear_blob(w: int, h: int) -> bool:
    short = max(1, min(w, h))
    long = max(w, h)
    aspect = long / short
    return aspect >= 3.0


def _merge_components(components: list[dict], stroke_w: float, image_area: int) -> list[dict]:
    if not components:
        return []

    gap_thresh = 6.0 * stroke_w
    special_gap = 8.0 * stroke_w
    max_area = 0.25 * float(image_area)

    comps = components[:]
    changed = True
    while changed and len(comps) > 1:
        changed = False
        i = 0
        while i < len(comps):
            j = i + 1
            while j < len(comps):
                b1 = comps[i]["bbox"]
                b2 = comps[j]["bbox"]
                dist = bbox_distance(b1, b2)
                union_bbox = _union_bbox(b1, b2)
                union_area = float(union_bbox[2] * union_bbox[3])

                parallel_merge = (
                    comps[i]["linear"]
                    and comps[j]["linear"]
                    and _angle_diff(comps[i]["theta"], comps[j]["theta"]) < 15.0
                    and dist < special_gap
                )

                close_merge = dist < gap_thresh and union_area < max_area

                if parallel_merge or close_merge:
                    comps[i] = _combine_components(comps[i], comps[j])
                    comps.pop(j)
                    changed = True
                    continue
                j += 1
            i += 1

    return comps


def bbox_distance(b1: tuple[int, int, int, int], b2: tuple[int, int, int, int]) -> float:
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    x1_max = x1 + w1
    y1_max = y1 + h1
    x2_max = x2 + w2
    y2_max = y2 + h2

    dx = max(x2 - x1_max, x1 - x2_max, 0)
    dy = max(y2 - y1_max, y1 - y2_max, 0)
    if dx > 0 and dy > 0:
        return math.hypot(dx, dy)
    return float(max(dx, dy))


def _union_bbox(b1: tuple[int, int, int, int], b2: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    x0 = min(x1, x2)
    y0 = min(y1, y2)
    x1_max = max(x1 + w1, x2 + w2)
    y1_max = max(y1 + h1, y2 + h2)
    return x0, y0, x1_max - x0, y1_max - y0


def _angle_diff(theta1: float, theta2: float) -> float:
    diff = abs(theta1 - theta2) % 180.0
    return min(diff, 180.0 - diff)


def _combine_components(c1: dict, c2: dict) -> dict:
    pixels = np.vstack((c1["pixels"], c2["pixels"]))
    ys = pixels[:, 0]
    xs = pixels[:, 1]
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    w = x1 - x0 + 1
    h = y1 - y0 + 1
    theta = _component_orientation(xs, ys)
    linear = _is_linear_blob(w, h)
    return {
        "pixels": pixels,
        "bbox": (x0, y0, w, h),
        "area": float(len(xs)),
        "theta": theta,
        "linear": linear,
    }


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


def _otsu_threshold(image: np.ndarray) -> float:
    flat = image.astype(np.float32).ravel()
    if flat.size == 0:
        return 0.0
    min_val = float(flat.min())
    max_val = float(flat.max())
    if min_val == max_val:
        return min_val
    hist, bin_edges = np.histogram(flat, bins=256, range=(min_val, max_val))
    hist = hist.astype(np.float64)
    prob = hist / hist.sum()
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    weight1 = np.cumsum(prob)
    mean1 = np.cumsum(prob * bin_centers)
    mean_total = mean1[-1]
    weight2 = 1.0 - weight1
    numerator = (mean_total * weight1 - mean1) ** 2
    denominator = weight1 * weight2
    with np.errstate(divide="ignore", invalid="ignore"):
        variance = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)
    best_index = int(np.argmax(variance[:-1]))
    return float(bin_centers[best_index])


def _disk_footprint(radius: int) -> np.ndarray:
    side = radius * 2 + 1
    y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    mask = (x * x + y * y) <= radius * radius
    return mask.astype(np.uint8)


def _binary_dilation(binary: np.ndarray, footprint: np.ndarray) -> np.ndarray:
    pad_y = footprint.shape[0] // 2
    pad_x = footprint.shape[1] // 2
    padded = np.pad(binary, ((pad_y, pad_y), (pad_x, pad_x)), mode="constant")
    windows = sliding_window_view(padded, footprint.shape)
    mask = footprint.astype(bool)
    dilated = windows[..., mask].max(axis=-1)
    return dilated.astype(np.uint8)


def _binary_erosion(binary: np.ndarray, footprint: np.ndarray) -> np.ndarray:
    pad_y = footprint.shape[0] // 2
    pad_x = footprint.shape[1] // 2
    padded = np.pad(binary, ((pad_y, pad_y), (pad_x, pad_x)), mode="constant", constant_values=1)
    windows = sliding_window_view(padded, footprint.shape)
    mask = footprint.astype(bool)
    eroded = windows[..., mask].min(axis=-1)
    return eroded.astype(np.uint8)


def _binary_closing(binary: np.ndarray, footprint: np.ndarray) -> np.ndarray:
    return _binary_erosion(_binary_dilation(binary, footprint), footprint)
