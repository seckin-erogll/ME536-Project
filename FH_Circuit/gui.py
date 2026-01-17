"""Tkinter GUI for sketching circuit symbols."""

from __future__ import annotations

import random
import tkinter as tk
from pathlib import Path
from tkinter import ttk

import numpy as np
from PIL import Image, ImageDraw, ImageTk

from FH_Circuit.classify import classify_sketch, load_artifacts


class SketchGUI:
    def __init__(self, model_dir: Path, dataset_dir: Path, canvas_size: int = 256) -> None:
        self.model, self.pca, self.kmeans, self.labels = load_artifacts(model_dir)
        self.dataset_dir = dataset_dir
        self.canvas_size = canvas_size
        self.root = tk.Tk()
        self.root.title("FH Circuit Sketch")
        self.canvas = tk.Canvas(self.root, width=canvas_size, height=canvas_size, bg="black")
        self.canvas.grid(row=0, column=0, columnspan=3, padx=10, pady=10)

        self.example_label = ttk.Label(self.root, text="Example (random)")
        self.example_label.grid(row=0, column=3, padx=10, pady=(10, 0))
        self.example_image_label = ttk.Label(self.root)
        self.example_image_label.grid(row=1, column=3, rowspan=2, padx=10, pady=(0, 10))
        self.example_caption = ttk.Label(self.root, text="")
        self.example_caption.grid(row=3, column=3, padx=10, pady=(0, 10))

        self.status = tk.StringVar(value="Draw a circuit symbol.")
        self.status_label = ttk.Label(self.root, textvariable=self.status)
        self.status_label.grid(row=1, column=0, columnspan=3, pady=(0, 10))

        ttk.Button(self.root, text="Classify", command=self.on_classify).grid(row=2, column=0, padx=5, pady=5)
        ttk.Button(self.root, text="Clear", command=self.on_clear).grid(row=2, column=1, padx=5, pady=5)
        ttk.Button(self.root, text="Quit", command=self.root.destroy).grid(row=2, column=2, padx=5, pady=5)

        self.image = Image.new("L", (canvas_size, canvas_size), color=0)
        self.draw = ImageDraw.Draw(self.image)
        self.last_x = None
        self.last_y = None
        self.example_photo = None
        self.example_paths = self._load_example_paths()

        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self._show_random_example(None)

    def on_press(self, event: tk.Event) -> None:
        self.last_x = event.x
        self.last_y = event.y

    def on_drag(self, event: tk.Event) -> None:
        if self.last_x is None or self.last_y is None:
            return
        self.canvas.create_line(self.last_x, self.last_y, event.x, event.y, fill="white", width=6)
        self.draw.line((self.last_x, self.last_y, event.x, event.y), fill=255, width=6)
        self.last_x = event.x
        self.last_y = event.y

    def on_release(self, _event: tk.Event) -> None:
        self.last_x = None
        self.last_y = None

    def on_clear(self) -> None:
        self.canvas.delete("all")
        self.image = Image.new("L", (self.canvas_size, self.canvas_size), color=0)
        self.draw = ImageDraw.Draw(self.image)
        self.status.set("Canvas cleared.")

    def on_classify(self) -> None:
        resized = self.image.resize((64, 64), resample=Image.BILINEAR)
        sketch = np.array(resized)
        try:
            result = classify_sketch(self.model, self.pca, self.kmeans, sketch, self.labels)
        except ValueError as exc:
            result = str(exc)
        self.status.set(result)
        self._update_example_from_result(result)

    def run(self) -> None:
        self.root.mainloop()

    def _load_example_paths(self) -> dict[str, list[Path]]:
        examples: dict[str, list[Path]] = {}
        image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        if not self.dataset_dir.exists():
            return examples
        for label_dir in self.dataset_dir.iterdir():
            if not label_dir.is_dir():
                continue
            images = [
                path
                for path in label_dir.iterdir()
                if path.suffix.lower() in image_extensions and path.is_file()
            ]
            if images:
                examples[label_dir.name.strip()] = images
        return examples

    def _update_example_from_result(self, result: str) -> None:
        if result.startswith("Detected: "):
            label = result.replace("Detected: ", "", 1).strip()
            self._show_random_example(label)
        else:
            self._show_random_example(None)

    def _show_random_example(self, label: str | None) -> None:
        if label is None:
            all_labels = list(self.example_paths.keys())
            if not all_labels:
                self.example_caption.config(text="No examples found.")
                self.example_image_label.configure(image="")
                return
            label = random.choice(all_labels)
        candidates = self.example_paths.get(label, [])
        if not candidates:
            self.example_caption.config(text=f"No examples for '{label}'.")
            self.example_image_label.configure(image="")
            return
        image_path = random.choice(candidates)
        image = Image.open(image_path).convert("L").resize((128, 128), resample=Image.BILINEAR)
        self.example_photo = ImageTk.PhotoImage(image)
        self.example_image_label.configure(image=self.example_photo)
        self.example_caption.config(text=f"Label: {label}")


def launch_gui(model_dir: Path, dataset_dir: Path) -> None:
    app = SketchGUI(model_dir, dataset_dir)
    app.run()
