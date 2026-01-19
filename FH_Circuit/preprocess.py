"""Image preprocessing utilities for circuit symbols."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image, ImageFilter

from FH_Circuit.config import IMAGE_SIZE, MIN_AREA

_SKIMAGE_AVAILABLE = importlib.util.find_spec("skimage") is not None
if _SKIMAGE_AVAILABLE:
    from skimage import filters, measure, morphology


@dataclass
class PreprocessResult:
    original: np.ndarray
    binarized: np.ndarray
    cleaned: np.ndarray
    final: np.ndarray
    bbox: Tuple[int, int, int, int]


def preprocess_image(
    image: np.ndarray,
    image_size: int = IMAGE_SIZE,
    min_area: int = MIN_AREA,
    deskew: bool = False,
    debug_dir: Optional[Path] = None,
    debug_prefix: str = "sample",
    augment: bool = False,
    augment_seed: Optional[int] = None,
) -> PreprocessResult:
    """Run preprocessing pipeline and return intermediate outputs."""
    original = _to_grayscale_uint8(image)
    denoised = _median_denoise(original)
    binarized = _binarize(denoised)
    cleaned = _morphology_cleanup(binarized)
    cleaned = _keep_largest_component(cleaned, min_area=min_area)
    if augment:
        cleaned = _augment_binary(cleaned, seed=augment_seed)
    if deskew:
        cleaned = _deskew_binary(cleaned)
    final, bbox = _crop_pad_resize(cleaned, image_size=image_size)
    result = PreprocessResult(
        original=original,
        binarized=binarized,
        cleaned=cleaned,
        final=final,
        bbox=bbox,
    )
    if debug_dir is not None:
        _save_debug(result, debug_dir, debug_prefix)
    return result


def _to_grayscale_uint8(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        grayscale = image
    else:
        grayscale = np.array(Image.fromarray(image).convert("L"))
    if grayscale.dtype != np.uint8:
        grayscale = np.clip(grayscale, 0, 255).astype(np.uint8)
    return grayscale


def _median_denoise(image: np.ndarray) -> np.ndarray:
    pil = Image.fromarray(image)
    return np.array(pil.filter(ImageFilter.MedianFilter(size=3)))


def _binarize(image: np.ndarray) -> np.ndarray:
    threshold = _otsu_threshold(image)
    binary = (image > threshold).astype(np.uint8)
    return _auto_invert(binary)


def _auto_invert(binary: np.ndarray) -> np.ndarray:
    # Ensure strokes are foreground (1) by comparing foreground ratio.
    if binary.mean() > 0.5:
        return 1 - binary
    return binary


def _otsu_threshold(image: np.ndarray) -> float:
    if _SKIMAGE_AVAILABLE:
        return float(filters.threshold_otsu(image))
    histogram, bin_edges = np.histogram(image.ravel(), bins=256, range=(0, 255))
    total = image.size
    sum_total = np.dot(np.arange(256), histogram)
    sum_background = 0.0
    weight_background = 0.0
    max_between = 0.0
    threshold = 0.0
    for i in range(256):
        weight_background += histogram[i]
        if weight_background == 0:
            continue
        weight_foreground = total - weight_background
        if weight_foreground == 0:
            break
        sum_background += i * histogram[i]
        mean_background = sum_background / weight_background
        mean_foreground = (sum_total - sum_background) / weight_foreground
        between = weight_background * weight_foreground * (mean_background - mean_foreground) ** 2
        if between > max_between:
            max_between = between
            threshold = i
    return threshold


def _morphology_cleanup(binary: np.ndarray) -> np.ndarray:
    if _SKIMAGE_AVAILABLE:
        selem = morphology.footprint_rectangle((3, 3))
        opened = morphology.opening(binary.astype(bool), selem)
        closed = morphology.closing(opened, selem)
        cleaned = morphology.remove_small_objects(closed.astype(bool), min_size=5)
        return cleaned.astype(np.uint8)
    # Safe fallback: avoid aggressive erosion that would destroy thin strokes.
    cleaned = _remove_tiny_components(binary, min_size=5)
    return cleaned.astype(np.uint8)


def _remove_tiny_components(binary: np.ndarray, min_size: int) -> np.ndarray:
    labels, counts = _connected_components(binary)
    if not counts:
        return binary
    output = np.zeros_like(binary)
    for label, count in counts.items():
        if count >= min_size:
            output[labels == label] = 1
    return output


def _keep_largest_component(binary: np.ndarray, min_area: int) -> np.ndarray:
    if binary.sum() < min_area:
        return binary
    if _SKIMAGE_AVAILABLE:
        labeled = measure.label(binary, connectivity=2)
        if labeled.max() == 0:
            return binary
        counts = np.bincount(labeled.ravel())
        counts[0] = 0
        largest = counts.argmax()
        return (labeled == largest).astype(np.uint8)
    labels, counts = _connected_components(binary)
    if not counts:
        return binary
    largest_label = max(counts, key=counts.get)
    return (labels == largest_label).astype(np.uint8)


def _connected_components(binary: np.ndarray) -> Tuple[np.ndarray, dict[int, int]]:
    labels = np.zeros_like(binary, dtype=np.int32)
    counts: dict[int, int] = {}
    label = 0
    for i in range(binary.shape[0]):
        for j in range(binary.shape[1]):
            if binary[i, j] == 0 or labels[i, j] != 0:
                continue
            label += 1
            stack = [(i, j)]
            labels[i, j] = label
            counts[label] = 0
            while stack:
                x, y = stack.pop()
                counts[label] += 1
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        if dx == 0 and dy == 0:
                            continue
                        nx, ny = x + dx, y + dy
                        if (
                            0 <= nx < binary.shape[0]
                            and 0 <= ny < binary.shape[1]
                            and binary[nx, ny] == 1
                            and labels[nx, ny] == 0
                        ):
                            labels[nx, ny] = label
                            stack.append((nx, ny))
    return labels, counts


def _deskew_binary(binary: np.ndarray) -> np.ndarray:
    coords = np.column_stack(np.where(binary > 0))
    if coords.shape[0] < 10:
        return binary
    mean = coords.mean(axis=0)
    centered = coords - mean
    cov = np.cov(centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    principal = eigvecs[:, np.argmax(eigvals)]
    angle = np.degrees(np.arctan2(principal[0], principal[1]))
    rotated = Image.fromarray((binary * 255).astype(np.uint8)).rotate(
        -angle,
        resample=Image.BILINEAR,
        fillcolor=0,
    )
    return (np.array(rotated) > 0).astype(np.uint8)


def _crop_pad_resize(binary: np.ndarray, image_size: int) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    coords = np.column_stack(np.where(binary > 0))
    if coords.size == 0:
        canvas = np.zeros((image_size, image_size), dtype=np.float32)
        return canvas, (0, 0, image_size, image_size)
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    cropped = binary[y_min : y_max + 1, x_min : x_max + 1]
    height, width = cropped.shape
    if height == 0 or width == 0:
        canvas = np.zeros((image_size, image_size), dtype=np.float32)
        return canvas, (0, 0, image_size, image_size)
    side = max(height, width)
    margin = max(2, int(round(0.15 * side)))
    padded_size = side + 2 * margin
    canvas = np.zeros((padded_size, padded_size), dtype=np.uint8)
    y_offset = (padded_size - height) // 2
    x_offset = (padded_size - width) // 2
    canvas[y_offset : y_offset + height, x_offset : x_offset + width] = cropped
    resized = Image.fromarray((canvas * 255).astype(np.uint8)).resize(
        (image_size, image_size),
        resample=Image.BILINEAR,
    )
    final = (np.array(resized) > 0).astype(np.float32)
    return final, (int(x_min), int(y_min), int(x_max), int(y_max))


def _augment_binary(binary: np.ndarray, seed: Optional[int]) -> np.ndarray:
    rng = np.random.default_rng(seed)
    augmented = binary.copy()
    # Stroke thickness augmentation simulates pen width differences.
    augmented = _random_thickness(augmented, rng)
    augmented = _random_rotate(augmented, rng, max_degrees=10)
    augmented = _random_translate(augmented, rng, max_shift=3)
    return augmented


def _random_thickness(binary: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    mode = rng.choice(["none", "dilate", "erode"], p=[0.5, 0.25, 0.25])
    if mode == "none":
        return binary
    if _SKIMAGE_AVAILABLE:
        selem = morphology.footprint_rectangle((3, 3))
        if mode == "dilate":
            return morphology.dilation(binary.astype(bool), selem).astype(np.uint8)
        return morphology.erosion(binary.astype(bool), selem).astype(np.uint8)
    if mode == "dilate":
        return _binary_dilation(binary)
    return _binary_soft_erosion(binary)


def _binary_dilation(binary: np.ndarray) -> np.ndarray:
    padded = np.pad(binary, 1, mode="constant")
    output = np.zeros_like(binary)
    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            region = padded[i : i + 3, j : j + 3]
            output[i, j] = 1 if region.sum() > 0 else 0
    return output


def _binary_soft_erosion(binary: np.ndarray, threshold: int = 3) -> np.ndarray:
    padded = np.pad(binary, 1, mode="constant")
    output = np.zeros_like(binary)
    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            region = padded[i : i + 3, j : j + 3]
            output[i, j] = 1 if region.sum() >= threshold else 0
    return output


def _random_rotate(binary: np.ndarray, rng: np.random.Generator, max_degrees: float) -> np.ndarray:
    angle = float(rng.uniform(-max_degrees, max_degrees))
    rotated = Image.fromarray((binary * 255).astype(np.uint8)).rotate(
        angle,
        resample=Image.BILINEAR,
        fillcolor=0,
    )
    return (np.array(rotated) > 0).astype(np.uint8)


def _random_translate(binary: np.ndarray, rng: np.random.Generator, max_shift: int) -> np.ndarray:
    shift_x = int(rng.integers(-max_shift, max_shift + 1))
    shift_y = int(rng.integers(-max_shift, max_shift + 1))
    padded = np.pad(binary, max_shift, mode="constant")
    start_x = max_shift + shift_x
    start_y = max_shift + shift_y
    end_x = start_x + binary.shape[0]
    end_y = start_y + binary.shape[1]
    shifted = padded[start_x:end_x, start_y:end_y]
    return shifted.astype(np.uint8)


def _save_debug(result: PreprocessResult, debug_dir: Path, prefix: str) -> None:
    debug_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(result.original).save(debug_dir / f"{prefix}_original.png")
    Image.fromarray((result.binarized * 255).astype(np.uint8)).save(debug_dir / f"{prefix}_binarized.png")
    Image.fromarray((result.cleaned * 255).astype(np.uint8)).save(debug_dir / f"{prefix}_cleaned.png")
    Image.fromarray((result.final * 255).astype(np.uint8)).save(debug_dir / f"{prefix}_final.png")
