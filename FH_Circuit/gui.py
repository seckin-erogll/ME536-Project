"""Tkinter GUI for sketching circuit symbols."""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import ttk

import numpy as np
from PIL import Image, ImageDraw

from FH_Circuit.classify import classify_sketch, load_artifacts


class SketchGUI:
    def __init__(self, model_dir: Path, canvas_size: int = 256) -> None:
        self.model, self.pca, self.kmeans = load_artifacts(model_dir)
        self.canvas_size = canvas_size
        self.root = tk.Tk()
        self.root.title("FH Circuit Sketch")
        self.canvas = tk.Canvas(self.root, width=canvas_size, height=canvas_size, bg="black")
        self.canvas.grid(row=0, column=0, columnspan=3, padx=10, pady=10)

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

        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)

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
            result = classify_sketch(self.model, self.pca, self.kmeans, sketch)
        except ValueError as exc:
            result = str(exc)
        self.status.set(result)

    def run(self) -> None:
        self.root.mainloop()


def launch_gui(model_dir: Path) -> None:
    app = SketchGUI(model_dir)
    app.run()
