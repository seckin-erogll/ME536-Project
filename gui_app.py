"""Tkinter GUI for sketching circuits and recognizing components/wire nodes."""

from __future__ import annotations

import argparse
import tkinter as tk
from pathlib import Path
from tkinter import ttk

import cv2
import numpy as np
from PIL import Image, ImageDraw
from sklearn.cluster import DBSCAN
from skimage.morphology import skeletonize

from FH_Circuit.classify import classify_sketch, load_artifacts


class CircuitSketchApp:
    def __init__(self, model_dir: Path, canvas_size: int = 800) -> None:
        self.artifacts = load_artifacts(model_dir)
        self.canvas_size = canvas_size
        self.root = tk.Tk()
        self.root.title("FH Circuit Recognizer")

        self.canvas = tk.Canvas(self.root, width=canvas_size, height=canvas_size, bg="white")
        self.canvas.grid(row=0, column=0, columnspan=2, padx=10, pady=10)

        self.status = tk.StringVar(value="Draw a circuit, then click Recognize.")
        ttk.Label(self.root, textvariable=self.status).grid(row=1, column=0, columnspan=2, pady=(0, 10))

        ttk.Button(self.root, text="Clear", command=self.on_clear).grid(row=2, column=0, padx=5, pady=5)
        ttk.Button(self.root, text="Recognize", command=self.on_recognize).grid(row=2, column=1, padx=5, pady=5)

        self.image = Image.new("L", (canvas_size, canvas_size), color=255)
        self.draw = ImageDraw.Draw(self.image)
        self.last_x: int | None = None
        self.last_y: int | None = None

        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)

    def run(self) -> None:
        self.root.mainloop()

    def on_press(self, event: tk.Event) -> None:
        self.last_x = event.x
        self.last_y = event.y

    def on_drag(self, event: tk.Event) -> None:
        if self.last_x is None or self.last_y is None:
            return
        self.canvas.create_line(self.last_x, self.last_y, event.x, event.y, fill="black", width=4)
        self.draw.line((self.last_x, self.last_y, event.x, event.y), fill=0, width=4)
        self.last_x = event.x
        self.last_y = event.y

    def on_release(self, _event: tk.Event) -> None:
        self.last_x = None
        self.last_y = None

    def on_clear(self) -> None:
        self.canvas.delete("all")
        self.image = Image.new("L", (self.canvas_size, self.canvas_size), color=255)
        self.draw = ImageDraw.Draw(self.image)
        self.status.set("Canvas cleared.")

    def on_recognize(self) -> None:
        self.canvas.delete("overlay")
        gray = np.array(self.image)
        binary = (gray < 128).astype(np.uint8) * 255

        wires_only = self._extract_wires(binary)
        components_only = cv2.subtract(binary, wires_only)

        component_boxes = self._cluster_components(components_only)
        labels = self._classify_components(component_boxes, gray)
        wire_nodes = self._find_wire_nodes(wires_only)

        for (x_min, y_min, x_max, y_max), label in zip(component_boxes, labels):
            self.canvas.create_rectangle(
                x_min,
                y_min,
                x_max,
                y_max,
                outline="red",
                width=2,
                tags="overlay",
            )
            self.canvas.create_text(
                x_min,
                max(y_min - 10, 0),
                text=label,
                fill="red",
                anchor="sw",
                tags="overlay",
            )

        node_radius = 4
        for y, x in wire_nodes:
            self.canvas.create_oval(
                x - node_radius,
                y - node_radius,
                x + node_radius,
                y + node_radius,
                outline="blue",
                width=2,
                tags="overlay",
            )

        self.status.set(
            f"Detected {len(component_boxes)} component(s) and {len(wire_nodes)} wire node(s)."
        )

    def _extract_wires(self, binary: np.ndarray) -> np.ndarray:
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 20))
        horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
        vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)
        return cv2.bitwise_or(horizontal, vertical)

    def _cluster_components(self, components_only: np.ndarray) -> list[tuple[int, int, int, int]]:
        coords = np.column_stack(np.where(components_only > 0))
        if coords.size == 0:
            return []
        points = coords[:, [1, 0]]
        dbscan = DBSCAN(eps=10, min_samples=20)
        labels = dbscan.fit_predict(points)
        boxes: list[tuple[int, int, int, int]] = []
        for cluster_id in sorted(set(labels)):
            if cluster_id == -1:
                continue
            cluster_coords = coords[labels == cluster_id]
            y_min, x_min = cluster_coords.min(axis=0)
            y_max, x_max = cluster_coords.max(axis=0)
            padding = 4
            x_min = max(int(x_min) - padding, 0)
            y_min = max(int(y_min) - padding, 0)
            x_max = min(int(x_max) + padding, self.canvas_size - 1)
            y_max = min(int(y_max) + padding, self.canvas_size - 1)
            boxes.append((x_min, y_min, x_max, y_max))
        return boxes

    def _classify_components(
        self,
        component_boxes: list[tuple[int, int, int, int]],
        gray_image: np.ndarray,
    ) -> list[str]:
        labels: list[str] = []
        for x_min, y_min, x_max, y_max in component_boxes:
            crop = gray_image[y_min : y_max + 1, x_min : x_max + 1]
            try:
                result = classify_sketch(self.artifacts, crop)
                labels.append(self._format_label(result))
            except ValueError as exc:
                labels.append(str(exc))
        return labels

    def _find_wire_nodes(self, wires_only: np.ndarray) -> list[tuple[int, int]]:
        skeleton = skeletonize(wires_only > 0).astype(np.uint8)
        padded = np.pad(skeleton, 1, mode="constant")
        neighbors = (
            padded[:-2, :-2]
            + padded[:-2, 1:-1]
            + padded[:-2, 2:]
            + padded[1:-1, :-2]
            + padded[1:-1, 2:]
            + padded[2:, :-2]
            + padded[2:, 1:-1]
            + padded[2:, 2:]
        )
        center = padded[1:-1, 1:-1]
        node_mask = (center == 1) & ((neighbors == 1) | (neighbors > 2))
        return list(map(tuple, np.column_stack(np.where(node_mask))))

    @staticmethod
    def _format_label(result: str) -> str:
        prefix = "Detected: "
        if result.startswith(prefix):
            return result[len(prefix) :].strip()
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Circuit sketch recognizer GUI.")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "artifacts",
        help="Path to model artifacts directory.",
    )
    parser.add_argument("--canvas-size", type=int, default=800, help="Canvas size in pixels.")
    args = parser.parse_args()
    app = CircuitSketchApp(args.model_dir, canvas_size=args.canvas_size)
    app.run()


if __name__ == "__main__":
    main()
