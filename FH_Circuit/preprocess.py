"""Image preprocessing utilities."""

from __future__ import annotations

import numpy as np
from PIL import Image
from numpy.lib.stride_tricks import sliding_window_view

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
