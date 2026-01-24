"""Tkinter GUI for circuit segmentation + classification."""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import ttk

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageTk

from FH_Circuit.classify import classify_sketch, load_artifacts
from FH_Circuit.preprocess import segment_symbols


class CircuitSegmentationApp:
    def __init__(self, root: tk.Tk, model_dir: Path) -> None:
        self.root = root
        self.root.title("Circuit Segmentation + Classification")

        self.canvas_width = 800
        self.canvas_height = 600
        self.artifacts = load_artifacts(model_dir)

        self.image = Image.new("RGB", (self.canvas_width, self.canvas_height), "white")
        self.draw = ImageDraw.Draw(self.image)

        self.canvas = tk.Canvas(
            root,
            width=self.canvas_width,
            height=self.canvas_height,
            bg="white",
            cursor="pencil",
        )
        self.canvas.grid(row=0, column=0, columnspan=2, padx=10, pady=10)

        self.clear_button = ttk.Button(root, text="Clear", command=self.clear_canvas)
        self.clear_button.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 10))

        self.segment_button = ttk.Button(
            root, text="Segment + Classify", command=self.segment_components
        )
        self.segment_button.grid(row=1, column=1, sticky="ew", padx=10, pady=(0, 10))

        self.status = tk.StringVar(value="Draw a circuit and click Segment + Classify.")
        self.status_label = ttk.Label(root, textvariable=self.status)
        self.status_label.grid(row=2, column=0, columnspan=2, pady=(0, 10))

        self.last_x = None
        self.last_y = None

        self.canvas.bind("<ButtonPress-1>", self.start_draw)
        self.canvas.bind("<B1-Motion>", self.draw_line)
        self.canvas.bind("<ButtonRelease-1>", self.stop_draw)

    def start_draw(self, event: tk.Event) -> None:
        self.last_x = event.x
        self.last_y = event.y

    def draw_line(self, event: tk.Event) -> None:
        if self.last_x is None or self.last_y is None:
            return

        self.canvas.create_line(
            self.last_x,
            self.last_y,
            event.x,
            event.y,
            fill="black",
            width=4,
            capstyle=tk.ROUND,
            smooth=True,
        )
        self.draw.line(
            [self.last_x, self.last_y, event.x, event.y],
            fill="black",
            width=4,
        )
        self.last_x = event.x
        self.last_y = event.y

    def stop_draw(self, _event: tk.Event) -> None:
        self.last_x = None
        self.last_y = None

    def clear_canvas(self) -> None:
        self.canvas.delete("all")
        self.image = Image.new("RGB", (self.canvas_width, self.canvas_height), "white")
        self.draw = ImageDraw.Draw(self.image)
        self.status.set("Canvas cleared.")

    def segment_components(self) -> None:
        gray = np.array(self.image.convert("L"))

        inverted = 255 - gray
        bin_img = (inverted > 0).astype(np.uint8) * 255

        bboxes = segment_symbols(bin_img)

        boxed = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        statuses = []
        saw_novelty = False
        height, width = bin_img.shape[:2]
        for x, y, w, h in bboxes:
            x, y, w, h = self._refine_bbox_by_pixels(bin_img, x, y, w, h, pad=10)
            x0 = max(0, x)
            y0 = max(0, y)
            x1 = min(width, x + w)
            y1 = min(height, y + h)
            crop = bin_img[y0:y1, x0:x1]

            if crop.size == 0:
                continue

            skel = self._skeletonize(crop)
            endpoints, junctions = self._count_endpoints_and_junctions(skel)

            if endpoints >= 4 and junctions >= 1:
                core_box = self._tight_core_ignore_arms(crop)
                if core_box is None:
                    continue
                cx0, cy0, cx1, cy1 = core_box
                x = x0 + cx0
                y = y0 + cy0
                w = cx1 - cx0 + 1
                h = cy1 - cy0 + 1

            x, y, w, h = self._expand_bbox(x, y, w, h, 1.10, width, height)

            label, is_novelty, status = self._classify_crop(gray, x, y, w, h)
            statuses.append(status)
            saw_novelty = saw_novelty or is_novelty

            color = (0, 128, 0) if not is_novelty else (0, 0, 255)
            cv2.rectangle(boxed, (x, y), (x + w, y + h), color, 2)
            cv2.putText(
                boxed,
                label,
                (x, max(10, y - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )

        if statuses:
            if saw_novelty:
                self.status.set("New object detected. " + " | ".join(statuses))
            else:
                self.status.set(" | ".join(statuses))
        else:
            self.status.set("No components detected.")

        self._update_canvas_from_array(boxed)

    def _classify_crop(self, gray: np.ndarray, x: int, y: int, w: int, h: int) -> tuple[str, bool, str]:
        x0 = max(0, x)
        y0 = max(0, y)
        x1 = min(gray.shape[1], x + w)
        y1 = min(gray.shape[0], y + h)
        crop = gray[y0:y1, x0:x1]
        if crop.size == 0:
            return "Unknown", False, "Empty crop"
        # The classifier expects light foreground on a dark background.
        # Circuit GUI uses a white background with black strokes, so invert.
        crop = 255 - crop
        try:
            result = classify_sketch(self.artifacts, crop)
        except ValueError as exc:
            return "Unknown", True, str(exc)
        label, is_novelty = self._label_from_result(result)
        return label, is_novelty, result

    def _label_from_result(self, result: str) -> tuple[str, bool]:
        if result.startswith("Detected:"):
            return result.replace("Detected:", "", 1).strip(), False
        if result.startswith("Novelty detected"):
            return "Unknown", True
        if result.startswith("Ambiguity detected"):
            return "Ambiguous", False
        return "Unknown", True

    def _refine_bbox_by_pixels(self, bin_img, x, y, w, h, pad=25, close_k=5, close_iter=1):
        height, width = bin_img.shape[:2]
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(width, x + w + pad)
        y1 = min(height, y + h + pad)
        crop = bin_img[y0:y1, x0:x1]
        if crop.size == 0:
            return x, y, w, h

        kernel = np.ones((close_k, close_k), np.uint8)
        closed = cv2.morphologyEx(crop, cv2.MORPH_CLOSE, kernel, iterations=close_iter)
        ys, xs = np.where(closed > 0)
        if xs.size == 0 or ys.size == 0:
            return x, y, w, h

        crop_x0, crop_x1 = xs.min(), xs.max()
        crop_y0, crop_y1 = ys.min(), ys.max()
        ref_x0 = x0 + int(crop_x0)
        ref_y0 = y0 + int(crop_y0)
        ref_x1 = x0 + int(crop_x1)
        ref_y1 = y0 + int(crop_y1)
        ref_x0, ref_y0, ref_x1, ref_y1 = self._clamp_bbox(
            ref_x0, ref_y0, ref_x1, ref_y1, width, height
        )
        return ref_x0, ref_y0, ref_x1 - ref_x0 + 1, ref_y1 - ref_y0 + 1

    def _clamp_bbox(self, x0, y0, x1, y1, width, height):
        x0 = max(0, min(x0, width - 1))
        y0 = max(0, min(y0, height - 1))
        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        if x1 < x0:
            x0, x1 = x1, x0
        if y1 < y0:
            y0, y1 = y1, y0
        return x0, y0, x1, y1

    def _expand_bbox(self, x, y, w, h, scale, width, height):
        if w <= 0 or h <= 0:
            return x, y, w, h
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        cx = x + w / 2.0
        cy = y + h / 2.0
        x0 = int(round(cx - new_w / 2.0))
        y0 = int(round(cy - new_h / 2.0))
        x1 = x0 + new_w - 1
        y1 = y0 + new_h - 1
        x0, y0, x1, y1 = self._clamp_bbox(x0, y0, x1, y1, width, height)
        return x0, y0, x1 - x0 + 1, y1 - y0 + 1

    def _skeletonize(self, img_bin_255):
        img = (img_bin_255 > 0).astype(np.uint8) * 255
        if cv2.countNonZero(img) == 0:
            return np.zeros_like(img, dtype=np.uint8)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        img = cv2.dilate(img, element, iterations=1)
        skel = np.zeros_like(img)
        while True:
            eroded = cv2.erode(img, element)
            opened = cv2.dilate(eroded, element)
            temp = cv2.subtract(img, opened)
            skel = cv2.bitwise_or(skel, temp)
            img = eroded
            if cv2.countNonZero(img) == 0:
                break
        return (skel > 0).astype(np.uint8)

    def _count_endpoints_and_junctions(self, skel01):
        k = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
        neigh = cv2.filter2D(skel01, -1, k, borderType=cv2.BORDER_CONSTANT)
        endpoints = np.count_nonzero((skel01 == 1) & (neigh == 1))
        junctions = np.count_nonzero((skel01 == 1) & (neigh >= 3))
        return endpoints, junctions

    def _tight_core_ignore_arms(self, crop_bin_255):
        mask = (crop_bin_255 > 0).astype(np.uint8)
        if cv2.countNonZero(mask) == 0:
            return None
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
        if dist.max() <= 0:
            return None
        core = (dist >= 0.6 * dist.max()).astype(np.uint8)
        ys, xs = np.where(core > 0)
        if xs.size == 0:
            return None
        x0, x1 = xs.min(), xs.max()
        y0, y1 = ys.min(), ys.max()
        return int(x0), int(y0), int(x1), int(y1)

    def _update_canvas_from_array(self, array_bgr):
        array_rgb = cv2.cvtColor(array_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(array_rgb)
        self.image = pil_image
        self.draw = ImageDraw.Draw(self.image)

        self.canvas.delete("all")
        self._tk_image = ImageTk.PhotoImage(self.image)
        self.canvas.create_image(0, 0, anchor="nw", image=self._tk_image)


def launch_circuit_gui(model_dir: Path) -> None:
    root = tk.Tk()
    app = CircuitSegmentationApp(root, model_dir=model_dir)
    root.mainloop()


if __name__ == "__main__":
    launch_circuit_gui(Path("./artifacts"))
