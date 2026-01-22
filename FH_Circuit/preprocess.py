"""Image preprocessing utilities."""

from __future__ import annotations

import numpy as np
from PIL import Image

from FH_Circuit.config import IMAGE_SIZE, MIN_AREA


def preprocess(image: np.ndarray, min_area: int = MIN_AREA) -> np.ndarray:
    thresh = _otsu_threshold(image)
    binary = (image > thresh).astype(np.uint8)
    if binary.sum() < min_area:
        raise ValueError("Noise detected: sketch too small.")
    binary = _normalize_to_canvas(binary)
    binary = _center_of_mass_align(binary)
    offsets = _disk_offsets(2)
    dilated = _binary_dilation(binary, offsets)
    closed = _binary_closing(dilated, offsets)
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
    flattened = image.astype(np.uint8).ravel()
    hist = np.bincount(flattened, minlength=256).astype(np.float64)
    total = flattened.size
    if total == 0:
        return 0.0
    cumulative = np.cumsum(hist)
    cumulative_mean = np.cumsum(hist * np.arange(256))
    total_mean = cumulative_mean[-1]
    numerator = (total_mean * cumulative - cumulative_mean) ** 2
    denominator = cumulative * (total - cumulative)
    denominator[denominator == 0] = 1
    sigma_b_squared = numerator / denominator
    return float(np.argmax(sigma_b_squared))


def _disk_offsets(radius: int) -> list[tuple[int, int]]:
    offsets = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dy * dy + dx * dx <= radius * radius:
                offsets.append((dy, dx))
    return offsets


def _binary_dilation(binary: np.ndarray, offsets: list[tuple[int, int]]) -> np.ndarray:
    height, width = binary.shape
    radius = max(max(abs(dy), abs(dx)) for dy, dx in offsets + [(0, 0)])
    padded = np.pad(binary, radius, mode="constant", constant_values=0)
    result = np.zeros_like(binary)
    for dy, dx in offsets:
        result = np.maximum(
            result,
            padded[radius + dy : radius + dy + height, radius + dx : radius + dx + width],
        )
    return result


def _binary_erosion(binary: np.ndarray, offsets: list[tuple[int, int]]) -> np.ndarray:
    height, width = binary.shape
    radius = max(max(abs(dy), abs(dx)) for dy, dx in offsets + [(0, 0)])
    padded = np.pad(binary, radius, mode="constant", constant_values=0)
    result = np.ones_like(binary)
    for dy, dx in offsets:
        result = np.minimum(
            result,
            padded[radius + dy : radius + dy + height, radius + dx : radius + dx + width],
        )
    return result


def _binary_closing(binary: np.ndarray, offsets: list[tuple[int, int]]) -> np.ndarray:
    return _binary_erosion(_binary_dilation(binary, offsets), offsets)
