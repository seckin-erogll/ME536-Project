"""Tkinter GUI for drawing and recognizing circuit components."""

from __future__ import annotations

import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw

from .config import DEFAULT_SAMPLES_PER_CLASS, IMAGE_SIZE
from .data import build_dataset
from .inference import classify_sketch
from .training import TrainingArtifacts, load_artifacts, save_artifacts, train_pipeline


@dataclass
class GuiState:
    artifacts: Optional[TrainingArtifacts] = None


def _create_canvas_image(size: int = 256) -> Image.Image:
    return Image.new("L", (size, size), color=0)


def run_gui() -> None:
    root = tk.Tk()
    root.title("Auto-Schematic: Draw a Component")

    canvas_size = 256
    state = GuiState()
    output_dir = Path("./artifacts")

    canvas = tk.Canvas(root, width=canvas_size, height=canvas_size, bg="white")
    canvas.grid(row=0, column=0, columnspan=3, padx=10, pady=10)

    status = tk.StringVar(value="Draw a symbol, then click Classify.")
    status_label = tk.Label(root, textvariable=status)
    status_label.grid(row=1, column=0, columnspan=3, padx=10, pady=5)

    buffer_image = _create_canvas_image(canvas_size)
    buffer_draw = ImageDraw.Draw(buffer_image)

    def clear_canvas() -> None:
        canvas.delete("all")
        nonlocal buffer_image, buffer_draw
        buffer_image = _create_canvas_image(canvas_size)
        buffer_draw = ImageDraw.Draw(buffer_image)
        status.set("Canvas cleared.")

    def on_draw(event: tk.Event) -> None:
        x, y = event.x, event.y
        r = 4
        canvas.create_oval(x - r, y - r, x + r, y + r, fill="black", outline="black")
        buffer_draw.ellipse((x - r, y - r, x + r, y + r), fill=255, outline=255)

    def train_model() -> None:
        status.set("Training on synthetic dataset...")
        root.update_idletasks()
        samples = build_dataset(DEFAULT_SAMPLES_PER_CLASS)
        artifacts = train_pipeline(samples)
        save_artifacts(artifacts, output_dir)
        state.artifacts = artifacts
        status.set("Training complete. Model ready.")

    def load_model() -> None:
        if (output_dir / "autoencoder.pt").exists():
            state.artifacts = load_artifacts(output_dir)
            status.set("Loaded trained model from artifacts/.")
        else:
            status.set("No saved model found. Please train first.")

    def classify() -> None:
        if state.artifacts is None:
            status.set("Model not loaded. Click Train or Load.")
            return
        resized = buffer_image.resize((IMAGE_SIZE, IMAGE_SIZE))
        sketch = np.array(resized)
        try:
            result = classify_sketch(state.artifacts.model, state.artifacts.pca, state.artifacts.kmeans, sketch)
        except ValueError as exc:
            status.set(str(exc))
            return
        status.set(result)

    train_button = tk.Button(root, text="Train", command=train_model)
    train_button.grid(row=2, column=0, padx=10, pady=10)

    load_button = tk.Button(root, text="Load", command=load_model)
    load_button.grid(row=2, column=1, padx=10, pady=10)

    classify_button = tk.Button(root, text="Classify", command=classify)
    classify_button.grid(row=2, column=2, padx=10, pady=10)

    clear_button = tk.Button(root, text="Clear", command=clear_canvas)
    clear_button.grid(row=3, column=0, columnspan=3, pady=5)

    canvas.bind("<B1-Motion>", on_draw)

    root.mainloop()
