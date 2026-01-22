"""Dataset utilities for circuit symbols."""

from __future__ import annotations

import dataclasses
import math
import random
import shutil
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageDraw

from FH_Circuit.config import IMAGE_SIZE
from FH_Circuit.preprocess import preprocess


COARSE_GROUPS: dict[str, list[str]] = {
    "zigzag/coil": ["resistor", "inductor"],
    "parallel": ["capacitor"],
    "diode-like": ["diode"],
    "source": ["source"],
    "ground": ["ground"],
}

AMBIGUOUS_COARSE_GROUPS = {"zigzag/coil"}


@dataclasses.dataclass
class Sample:
    image: np.ndarray
    label: str


def _draw_resistor(draw: ImageDraw.ImageDraw, bbox: Tuple[int, int, int, int], thickness: int) -> None:
    x0, y0, x1, y1 = bbox
    mid_y = (y0 + y1) // 2
    step = (x1 - x0) // 6
    points = [
        (x0, mid_y),
        (x0 + step, mid_y - step),
        (x0 + 2 * step, mid_y + step),
        (x0 + 3 * step, mid_y - step),
        (x0 + 4 * step, mid_y + step),
        (x0 + 5 * step, mid_y - step),
        (x1, mid_y),
    ]
    draw.line(points, fill=255, width=thickness)


def _draw_capacitor(draw: ImageDraw.ImageDraw, bbox: Tuple[int, int, int, int], thickness: int) -> None:
    x0, y0, x1, y1 = bbox
    mid_x = (x0 + x1) // 2
    margin = (x1 - x0) // 5
    draw.line((x0, (y0 + y1) // 2, mid_x - margin, (y0 + y1) // 2), fill=255, width=thickness)
    draw.line((mid_x - margin, y0, mid_x - margin, y1), fill=255, width=thickness)
    draw.line((mid_x + margin, y0, mid_x + margin, y1), fill=255, width=thickness)
    draw.line((mid_x + margin, (y0 + y1) // 2, x1, (y0 + y1) // 2), fill=255, width=thickness)


def _draw_inductor(draw: ImageDraw.ImageDraw, bbox: Tuple[int, int, int, int], thickness: int) -> None:
    x0, y0, x1, y1 = bbox
    mid_y = (y0 + y1) // 2
    width = x1 - x0
    coils = 4
    coil_w = width // (coils + 1)
    draw.line((x0, mid_y, x0 + coil_w // 2, mid_y), fill=255, width=thickness)
    for i in range(coils):
        left = x0 + (i + 0.5) * coil_w
        right = left + coil_w
        draw.arc([left, mid_y - coil_w // 2, right, mid_y + coil_w // 2], 0, 180, fill=255, width=thickness)
    draw.line((x1 - coil_w // 2, mid_y, x1, mid_y), fill=255, width=thickness)


def _draw_source(draw: ImageDraw.ImageDraw, bbox: Tuple[int, int, int, int], thickness: int) -> None:
    x0, y0, x1, y1 = bbox
    mid_x = (x0 + x1) // 2
    mid_y = (y0 + y1) // 2
    radius = min(x1 - x0, y1 - y0) // 3
    draw.ellipse((mid_x - radius, mid_y - radius, mid_x + radius, mid_y + radius), outline=255, width=thickness)
    draw.line((x0, mid_y, mid_x - radius, mid_y), fill=255, width=thickness)
    draw.line((mid_x + radius, mid_y, x1, mid_y), fill=255, width=thickness)
    draw.line((mid_x, mid_y - radius // 2, mid_x, mid_y + radius // 2), fill=255, width=thickness)


def _draw_ground(draw: ImageDraw.ImageDraw, bbox: Tuple[int, int, int, int], thickness: int) -> None:
    x0, y0, x1, y1 = bbox
    mid_x = (x0 + x1) // 2
    top = y0 + (y1 - y0) // 3
    draw.line((mid_x, y0, mid_x, top), fill=255, width=thickness)
    widths = [0.7, 0.45, 0.2]
    for i, ratio in enumerate(widths):
        half = int((x1 - x0) * ratio / 2)
        y = top + i * thickness * 2
        draw.line((mid_x - half, y, mid_x + half, y), fill=255, width=thickness)


_DRAWERS = {
    "resistor": _draw_resistor,
    "capacitor": _draw_capacitor,
    "inductor": _draw_inductor,
    "source": _draw_source,
    "ground": _draw_ground,
}


def render_symbol(symbol: str, size: int = IMAGE_SIZE, thickness: int = 2) -> Image.Image:
    image = Image.new("L", (size, size), color=0)
    draw = ImageDraw.Draw(image)
    margin = size // 6
    bbox = (margin, margin, size - margin, size - margin)
    _DRAWERS[symbol](draw, bbox, thickness)
    return image


def add_noise(image: np.ndarray, amount: float) -> np.ndarray:
    noise = np.random.normal(0, amount, image.shape)
    noisy = np.clip(image + noise, 0, 255)
    return noisy.astype(np.uint8)


def random_affine(image: Image.Image, rotation: float, shear: float, scale: float) -> Image.Image:
    angle = random.uniform(-rotation, rotation)
    shear_x = random.uniform(-shear, shear)
    shear_y = random.uniform(-shear, shear)
    scale_factor = random.uniform(1 - scale, 1 + scale)
    matrix = (
        scale_factor,
        math.tan(math.radians(shear_x)),
        0,
        math.tan(math.radians(shear_y)),
        scale_factor,
        0,
    )
    transformed = image.transform(image.size, Image.AFFINE, matrix, resample=Image.BILINEAR)
    return transformed.rotate(angle, resample=Image.BILINEAR, fillcolor=0)


def synthesize_sample(symbol: str, size: int = IMAGE_SIZE) -> np.ndarray:
    thickness = random.randint(1, 4)
    image = render_symbol(symbol, size=size, thickness=thickness)
    image = random_affine(image, rotation=20, shear=5, scale=0.15)
    arr = np.array(image)
    arr = add_noise(arr, amount=random.uniform(5, 18))
    return arr


def load_training_dataset(
    dataset_dir: Path, size: int = IMAGE_SIZE
) -> Tuple[List[Sample], List[Sample], List[str]]:
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    train_dir = dataset_dir / "train"
    validation_dir = dataset_dir / "validation"
    val_dir = dataset_dir / "val"
    if not validation_dir.exists() and val_dir.exists():
        validation_dir = val_dir

    if train_dir.exists() and validation_dir.exists():
        train_samples, labels = _load_samples_from_root(train_dir, size=size)
        validation_samples, _ = _load_samples_from_root(validation_dir, size=size)
        return train_samples, validation_samples, labels

    samples: dict[str, List[Path]] = {}
    image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

    for class_dir in sorted(dataset_dir.iterdir(), key=lambda entry: entry.name.lower()):
        if not class_dir.is_dir() or class_dir.name in {"train", "validation", "val"}:
            continue
        label = class_dir.name.strip()
        if not label:
            continue
        for image_path in class_dir.iterdir():
            if image_path.suffix.lower() not in image_extensions:
                continue
            samples.setdefault(label, []).append(image_path)

    if not samples:
        raise ValueError("No samples found. Check dataset directory structure and labels.")
    _split_dataset_on_disk(dataset_dir, samples, image_extensions)
    train_samples, labels = _load_samples_from_root(train_dir, size=size)
    validation_samples, _ = _load_samples_from_root(validation_dir, size=size)
    return train_samples, validation_samples, labels


def _split_dataset_on_disk(
    dataset_dir: Path,
    samples: dict[str, List[Path]],
    image_extensions: set[str],
    train_ratio: float = 0.8,
) -> None:
    train_dir = dataset_dir / "train"
    validation_dir = dataset_dir / "validation"
    train_dir.mkdir(parents=True, exist_ok=True)
    validation_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(42)
    for label, image_paths in samples.items():
        rng.shuffle(image_paths)
        total = len(image_paths)
        val_count = 0 if total < 2 else max(1, int(round(total * (1 - train_ratio))))
        train_paths = image_paths[: total - val_count]
        val_paths = image_paths[total - val_count :]
        for split_dir, split_paths in ((train_dir, train_paths), (validation_dir, val_paths)):
            class_dir = split_dir / label
            class_dir.mkdir(parents=True, exist_ok=True)
            for image_path in split_paths:
                if image_path.suffix.lower() not in image_extensions:
                    continue
                dest_path = class_dir / image_path.name
                if not dest_path.exists():
                    shutil.copy2(image_path, dest_path)


def _load_samples_from_root(dataset_dir: Path, size: int) -> Tuple[List[Sample], List[str]]:
    samples: List[Sample] = []
    labels: List[str] = []
    image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    skipped = 0
    for class_dir in sorted(dataset_dir.iterdir(), key=lambda entry: entry.name.lower()):
        if not class_dir.is_dir():
            continue
        label = class_dir.name.strip()
        if not label:
            continue
        labels.append(label)
        for image_path in class_dir.iterdir():
            if image_path.suffix.lower() not in image_extensions:
                continue
            image = Image.open(image_path).convert("L")
            image = image.resize((size, size), resample=Image.BILINEAR)
            image_array = np.array(image)
            try:
                preprocess(image_array)
            except ValueError:
                skipped += 1
                continue
            samples.append(Sample(image=image_array, label=label))
    if not samples:
        raise ValueError("No samples found. Check dataset directory structure and labels.")
    random.shuffle(samples)
    if skipped:
        print(f"Skipped {skipped} unusable samples in {dataset_dir}.")
    return samples, labels


def save_samples(samples: List[Sample], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for idx, sample in enumerate(samples):
        image = Image.fromarray(sample.image)
        image.save(output_dir / f"{sample.label}_{idx:04d}.png")


def resolve_coarse_label(label: str) -> str:
    for group, members in COARSE_GROUPS.items():
        if label in members:
            return group
    return label


def list_coarse_labels(labels: List[str]) -> List[str]:
    coarse_labels: List[str] = []
    for label in labels:
        group = resolve_coarse_label(label)
        if group not in coarse_labels:
            coarse_labels.append(group)
    return coarse_labels


def labels_for_coarse_group(group: str) -> List[str]:
    return COARSE_GROUPS.get(group, [group])
