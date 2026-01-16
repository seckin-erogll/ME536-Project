"""Synthetic dataset generation for circuit symbols."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from PIL import Image, ImageDraw

from .config import IMAGE_SIZE, SYMBOLS


@dataclass
class Sample:
    image: np.ndarray
    label: str


def draw_resistor(draw: ImageDraw.ImageDraw, bbox: Tuple[int, int, int, int], thickness: int) -> None:
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


def draw_capacitor(draw: ImageDraw.ImageDraw, bbox: Tuple[int, int, int, int], thickness: int) -> None:
    x0, y0, x1, y1 = bbox
    mid_x = (x0 + x1) // 2
    margin = (x1 - x0) // 5
    draw.line((x0, (y0 + y1) // 2, mid_x - margin, (y0 + y1) // 2), fill=255, width=thickness)
    draw.line((mid_x - margin, y0, mid_x - margin, y1), fill=255, width=thickness)
    draw.line((mid_x + margin, y0, mid_x + margin, y1), fill=255, width=thickness)
    draw.line((mid_x + margin, (y0 + y1) // 2, x1, (y0 + y1) // 2), fill=255, width=thickness)


def draw_inductor(draw: ImageDraw.ImageDraw, bbox: Tuple[int, int, int, int], thickness: int) -> None:
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


def draw_source(draw: ImageDraw.ImageDraw, bbox: Tuple[int, int, int, int], thickness: int) -> None:
    x0, y0, x1, y1 = bbox
    mid_x = (x0 + x1) // 2
    mid_y = (y0 + y1) // 2
    radius = min(x1 - x0, y1 - y0) // 3
    draw.ellipse((mid_x - radius, mid_y - radius, mid_x + radius, mid_y + radius), outline=255, width=thickness)
    draw.line((x0, mid_y, mid_x - radius, mid_y), fill=255, width=thickness)
    draw.line((mid_x + radius, mid_y, x1, mid_y), fill=255, width=thickness)
    draw.line((mid_x, mid_y - radius // 2, mid_x, mid_y + radius // 2), fill=255, width=thickness)


def draw_ground(draw: ImageDraw.ImageDraw, bbox: Tuple[int, int, int, int], thickness: int) -> None:
    x0, y0, x1, y1 = bbox
    mid_x = (x0 + x1) // 2
    top = y0 + (y1 - y0) // 3
    draw.line((mid_x, y0, mid_x, top), fill=255, width=thickness)
    widths = [0.7, 0.45, 0.2]
    for i, ratio in enumerate(widths):
        half = int((x1 - x0) * ratio / 2)
        y = top + i * thickness * 2
        draw.line((mid_x - half, y, mid_x + half, y), fill=255, width=thickness)


DRAWERS: Dict[str, callable] = {
    "resistor": draw_resistor,
    "capacitor": draw_capacitor,
    "inductor": draw_inductor,
    "source": draw_source,
    "ground": draw_ground,
}


def render_symbol(symbol: str, size: int = IMAGE_SIZE, thickness: int = 2) -> Image.Image:
    image = Image.new("L", (size, size), color=0)
    draw = ImageDraw.Draw(image)
    margin = size // 6
    bbox = (margin, margin, size - margin, size - margin)
    DRAWERS[symbol](draw, bbox, thickness)
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


def build_dataset(count_per_class: int, size: int = IMAGE_SIZE) -> List[Sample]:
    samples: List[Sample] = []
    for symbol in SYMBOLS:
        for _ in range(count_per_class):
            samples.append(Sample(image=synthesize_sample(symbol, size=size), label=symbol))
    random.shuffle(samples)
    return samples


def save_samples(samples: Iterable[Sample], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for idx, sample in enumerate(samples):
        image = Image.fromarray(sample.image)
        image.save(output_dir / f"{sample.label}_{idx:04d}.png")
