"""Tkinter GUI for sketching circuit symbols."""

from __future__ import annotations

import random
import tkinter as tk
from pathlib import Path
from tkinter import ttk

import numpy as np
from PIL import Image, ImageDraw, ImageTk

from FH_Circuit.classify import classify_sketch, load_artifacts


def propose_bboxes(image: Image.Image) -> list[tuple[int, int, int, int]]:
    """Return bounding boxes in original image coordinates.

    Replace this simple heuristic with your detector if available.
    """
    array = np.array(image)
    if array.ndim == 3:
        mask = np.any(array > 0, axis=-1)
    else:
        mask = array > 0
    if not np.any(mask):
        return []
    ys, xs = np.where(mask)
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    return [(x1, y1, x2, y2)]


class SketchGUI:
    def __init__(self, model_dir: Path, dataset_dir: Path, canvas_size: int = 256) -> None:
        self.artifacts = load_artifacts(model_dir)
        self.dataset_dir = dataset_dir
        self.canvas_size = canvas_size
        self.root = tk.Tk()
        self.root.title("FH Circuit Sketch")
        self.canvas = tk.Canvas(self.root, width=canvas_size, height=canvas_size, bg="black")
        self.canvas.grid(row=0, column=0, columnspan=3, padx=10, pady=10, sticky="nsew")
        for column in range(3):
            self.root.grid_columnconfigure(column, weight=1)
        self.root.grid_rowconfigure(0, weight=1)

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
        ttk.Button(self.root, text="Detect", command=self.on_detect).grid(row=2, column=1, padx=5, pady=5)
        ttk.Button(self.root, text="Clear", command=self.on_clear).grid(row=2, column=2, padx=5, pady=5)
        ttk.Button(self.root, text="Quit", command=self.root.destroy).grid(row=3, column=2, padx=5, pady=5)

        self.image = Image.new("L", (canvas_size, canvas_size), color=0)
        self.draw = ImageDraw.Draw(self.image)
        self.display_image: Image.Image | None = None
        self.photo_image: ImageTk.PhotoImage | None = None
        self.image_item: int | None = None
        self.scale = 1.0
        self.offset_x = 0.0
        self.offset_y = 0.0
        self.bboxes: list[tuple[int, int, int, int]] = []
        self.last_x = None
        self.last_y = None
        self.example_photo = None
        self.example_paths = self._load_example_paths()

        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.canvas.bind("<Configure>", self.on_canvas_resize)
        self.set_image(self.image)
        self._show_random_example(None)

    def on_press(self, event: tk.Event) -> None:
        self.last_x = event.x
        self.last_y = event.y

    def on_drag(self, event: tk.Event) -> None:
        if self.last_x is None or self.last_y is None:
            return
        self.draw.line((self.last_x, self.last_y, event.x, event.y), fill=255, width=6)
        self.set_image(self.image)
        self.last_x = event.x
        self.last_y = event.y

    def on_release(self, _event: tk.Event) -> None:
        self.last_x = None
        self.last_y = None

    def on_clear(self) -> None:
        self.canvas.delete("all")
        self.image = Image.new("L", (self.canvas_size, self.canvas_size), color=0)
        self.draw = ImageDraw.Draw(self.image)
        self.image_item = None
        self.photo_image = None
        self.bboxes = []
        self.set_image(self.image)
        self.status.set("Canvas cleared.")

    def on_classify(self) -> None:
        resized = self.image.resize((64, 64), resample=Image.BILINEAR)
        sketch = np.array(resized)
        try:
            result = classify_sketch(self.artifacts, sketch)
        except ValueError as exc:
            result = str(exc)
        self.status.set(result)
        self._update_example_from_result(result)

    def on_detect(self) -> None:
        bboxes = propose_bboxes(self.image)
        self.draw_bboxes(bboxes)
        if bboxes:
            self.status.set(f"Detected {len(bboxes)} bounding box(es).")
        else:
            self.status.set("No bounding boxes detected.")

    def on_canvas_resize(self, _event: tk.Event) -> None:
        if self.display_image is None:
            return
        self.set_image(self.display_image)

    def run(self) -> None:
        self.root.mainloop()

    def set_image(self, pil_image: Image.Image) -> None:
        self.display_image = pil_image
        self.image = pil_image
        self.draw = ImageDraw.Draw(self.image)
        canvas_width = max(self.canvas.winfo_width(), 1)
        canvas_height = max(self.canvas.winfo_height(), 1)
        image_width, image_height = pil_image.size
        if image_width == 0 or image_height == 0:
            return
        self.scale = min(canvas_width / image_width, canvas_height / image_height)
        display_width = max(int(image_width * self.scale), 1)
        display_height = max(int(image_height * self.scale), 1)
        self.offset_x = (canvas_width - display_width) / 2
        self.offset_y = (canvas_height - display_height) / 2
        resized = pil_image.resize((display_width, display_height), resample=Image.BILINEAR)
        self.photo_image = ImageTk.PhotoImage(resized)
        if self.image_item is None:
            self.image_item = self.canvas.create_image(
                self.offset_x,
                self.offset_y,
                image=self.photo_image,
                anchor="nw",
                tags=("img",),
            )
        else:
            self.canvas.itemconfigure(self.image_item, image=self.photo_image)
            self.canvas.coords(self.image_item, self.offset_x, self.offset_y)
        self.draw_bboxes(self.bboxes)

    def draw_bboxes(self, bboxes: list[tuple[int, int, int, int]]) -> None:
        self.bboxes = list(bboxes)
        self.clear_bboxes()
        if self.display_image is None:
            return
        for x1, y1, x2, y2 in bboxes:
            canvas_x1 = self.offset_x + self.scale * x1
            canvas_y1 = self.offset_y + self.scale * y1
            canvas_x2 = self.offset_x + self.scale * x2
            canvas_y2 = self.offset_y + self.scale * y2
            self.canvas.create_rectangle(
                canvas_x1,
                canvas_y1,
                canvas_x2,
                canvas_y2,
                outline="red",
                width=2,
                tags=("bbox",),
            )

    def clear_bboxes(self) -> None:
        self.canvas.delete("bbox")

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
