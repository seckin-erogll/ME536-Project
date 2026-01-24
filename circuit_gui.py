"""Tkinter GUI for circuit segmentation + classification."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
import tkinter as tk

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageTk

from FH_Circuit.classify import ClassificationResult, classify_sketch_detailed, load_artifacts
from FH_Circuit.config import IMAGE_SIZE
from FH_Circuit.data import ensure_train_val_split
from FH_Circuit.train import incremental_update_pipeline

_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _as_uint8(image: np.ndarray) -> np.ndarray:
    return np.clip(image, 0, 255).astype(np.uint8)


class SymbolCollector(tk.Toplevel):
    def __init__(
        self,
        parent: tk.Tk,
        *,
        label: str,
        target_total: int,
        initial_total: int,
        normalize_fn,
        save_fn,
    ) -> None:
        super().__init__(parent)
        self.title(f"Collect samples: {label}")
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()

        self.label = label
        self.target_total = target_total
        self.normalize_fn = normalize_fn
        self.save_fn = save_fn

        self.canvas_size = 256
        self.image = Image.new("L", (self.canvas_size, self.canvas_size), color=255)
        self.draw = ImageDraw.Draw(self.image)

        self.saved_total = initial_total
        self.saved_in_session = 0

        self.status_var = tk.StringVar()
        self._update_status()

        self.canvas = tk.Canvas(
            self,
            width=self.canvas_size,
            height=self.canvas_size,
            bg="white",
            cursor="pencil",
        )
        self.canvas.grid(row=0, column=0, columnspan=3, padx=10, pady=10)

        ttk.Button(self, text="Clear", command=self.clear_canvas).grid(
            row=1, column=0, sticky="ew", padx=10, pady=(0, 8)
        )
        ttk.Button(self, text="Save Sample", command=self.save_sample).grid(
            row=1, column=1, sticky="ew", padx=10, pady=(0, 8)
        )
        ttk.Button(self, text="Done", command=self.finish).grid(
            row=1, column=2, sticky="ew", padx=10, pady=(0, 8)
        )

        ttk.Label(self, textvariable=self.status_var).grid(
            row=2, column=0, columnspan=3, padx=10, pady=(0, 10)
        )

        self.last_x: int | None = None
        self.last_y: int | None = None

        self.canvas.bind("<ButtonPress-1>", self.start_draw)
        self.canvas.bind("<B1-Motion>", self.draw_line)
        self.canvas.bind("<ButtonRelease-1>", self.stop_draw)

    def _update_status(self) -> None:
        remaining = max(0, self.target_total - self.saved_total)
        self.status_var.set(
            f"Saved: {self.saved_total}/{self.target_total} | Remaining: {remaining}"
        )

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
        self.draw.line((self.last_x, self.last_y, event.x, event.y), fill=0, width=4)
        self.last_x = event.x
        self.last_y = event.y

    def stop_draw(self, _event: tk.Event) -> None:
        self.last_x = None
        self.last_y = None

    def clear_canvas(self) -> None:
        self.canvas.delete("all")
        self.image = Image.new("L", (self.canvas_size, self.canvas_size), color=255)
        self.draw = ImageDraw.Draw(self.image)

    def save_sample(self) -> None:
        raw = np.array(self.image.convert("L"))
        inverted = 255 - raw
        normalized = self.normalize_fn(inverted)
        if np.count_nonzero(normalized) == 0:
            messagebox.showwarning("No strokes", "Draw a symbol before saving.", parent=self)
            return
        self.save_fn(self.label, normalized, hard_example=False)
        self.saved_total += 1
        self.saved_in_session += 1
        self._update_status()
        self.clear_canvas()
        if self.saved_total >= self.target_total:
            messagebox.showinfo(
                "Collection complete",
                f"Collected {self.saved_total} samples for '{self.label}'.",
                parent=self,
            )

    def finish(self) -> None:
        if self.saved_total < self.target_total:
            if not messagebox.askyesno(
                "Insufficient samples",
                "You have not reached the required sample count yet. Close anyway?",
                parent=self,
            ):
                return
        self.destroy()


class CircuitSegmentationApp:
    def __init__(self, root: tk.Tk, model_dir: Path, dataset_dir: Path, required_samples: int = 5) -> None:
        self.root = root
        self.root.title("Circuit Segmentation + Classification")

        self.model_dir = model_dir
        self.dataset_dir = dataset_dir
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        self.required_samples = max(1, required_samples)
        self.discarded_dir = self.model_dir / "_discarded"
        self.discarded_dir.mkdir(parents=True, exist_ok=True)

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

        self.last_x: int | None = None
        self.last_y: int | None = None
        self._save_counter = 0

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

        density_map = cv2.blur(inverted, (40, 40))

        _, dense_mask = cv2.threshold(density_map, 50, 255, cv2.THRESH_BINARY)

        kernel = np.ones((15, 15), np.uint8)
        refined = cv2.dilate(dense_mask, kernel, iterations=1)

        contours, _ = cv2.findContours(refined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boxed = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        statuses: list[str] = []
        saw_novelty = False
        height, width = bin_img.shape[:2]
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 500:
                continue
            x, y, w, h = cv2.boundingRect(contour)
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
        crop = self._prepare_crop(gray, x, y, w, h)
        if crop is None:
            return "Unknown", False, "Empty crop"
        result = classify_sketch_detailed(self.artifacts, crop)
        handled_label, updated = self._handle_result(crop, result)
        status_message = result.message
        if result.status == "ambiguous" and handled_label:
            if handled_label == "Noise":
                status_message = "Ambiguity resolved: marked as noise."
            else:
                status_message = f"Ambiguity resolved: {handled_label}"
        if result.status == "noisy" and handled_label == "Noise":
            status_message = "Marked as noise."
        if updated:
            self.artifacts = load_artifacts(self.model_dir)
            result = classify_sketch_detailed(self.artifacts, crop)
            handled_label = handled_label or (result.label if result.label else "Unknown")
            status_message = result.message
        label = handled_label or (result.label if result.label else self._label_from_status(result.status))
        is_novelty = result.status in {"novel", "noisy"} or label in {"Unknown", "Noise"}
        return label, is_novelty, status_message

    def _handle_result(self, crop: np.ndarray, result: ClassificationResult) -> tuple[str | None, bool]:
        if result.status == "ambiguous":
            return self._handle_ambiguity(crop, result), False
        if result.status in {"novel", "noisy"}:
            return self._handle_novelty_or_noise(crop, result)
        return None, False

    def _handle_ambiguity(self, crop: np.ndarray, result: ClassificationResult) -> str | None:
        top_candidates = result.candidates[:2]
        if not top_candidates:
            top_candidates = [(label, 0.0) for label in self.artifacts.labels[:2]]

        dialog = tk.Toplevel(self.root)
        dialog.title("Ambiguity detected")
        dialog.transient(self.root)
        dialog.grab_set()
        dialog.resizable(False, False)

        preview = self._make_preview_image(crop, size=192)
        preview_label = ttk.Label(dialog, image=preview)
        preview_label.image = preview
        preview_label.grid(row=0, column=0, columnspan=2, padx=12, pady=(12, 6))

        ttk.Label(dialog, text="Select the correct label:").grid(
            row=1, column=0, columnspan=2, padx=12, pady=(0, 8)
        )

        selection = {"label": None}

        def choose(label_name: str | None) -> None:
            selection["label"] = label_name
            dialog.destroy()

        for idx, (label_name, prob) in enumerate(top_candidates):
            text = f"{label_name} ({prob * 100:.1f}%)"
            ttk.Button(dialog, text=text, command=lambda name=label_name: choose(name)).grid(
                row=2 + idx, column=0, columnspan=2, sticky="ew", padx=12, pady=4
            )

        ttk.Button(dialog, text="Noise / Bad crop", command=lambda: choose(None)).grid(
            row=4, column=0, columnspan=2, sticky="ew", padx=12, pady=(6, 12)
        )

        self.root.wait_window(dialog)

        if selection["label"]:
            saved_path = self._save_dataset_sample(selection["label"], crop, hard_example=True)
            self.status.set(f"Saved hard example: {saved_path.name}")
            return selection["label"]
        discarded_path = self._save_discarded_sample(crop)
        self.status.set(f"Discarded ambiguous crop: {discarded_path.name}")
        return "Noise"

    def _handle_novelty_or_noise(self, crop: np.ndarray, result: ClassificationResult) -> tuple[str | None, bool]:
        if result.status == "noisy":
            self._show_noise_dialog(crop, result.message)
            return "Noise", False

        class_name = self._prompt_new_class_name(crop)
        if not class_name:
            return "Unknown", False

        self._save_dataset_sample(class_name, crop, hard_example=False)
        total_before = self._count_label_samples(class_name)
        if total_before < self.required_samples:
            self._collect_samples_for_label(class_name, total_before)
        total_after = self._count_label_samples(class_name)
        if total_after < self.required_samples:
            messagebox.showwarning(
                "Not enough samples",
                f"Need at least {self.required_samples} samples to update the model.",
                parent=self.root,
            )
            return class_name, False

        self.status.set("Updating model with new class...")
        self.root.update_idletasks()
        try:
            incremental_update_pipeline(self.dataset_dir, self.model_dir)
        except Exception as exc:  # pragma: no cover - defensive UI guard
            messagebox.showerror("Update failed", str(exc), parent=self.root)
            self.status.set("Incremental update failed.")
            return class_name, False
        self.status.set(f"Model updated with class '{class_name}'.")
        return class_name, True

    def _collect_samples_for_label(self, label: str, initial_total: int) -> None:
        while True:
            current_total = self._count_label_samples(label)
            if current_total >= self.required_samples:
                return
            mode = self._choose_collection_mode(label, current_total)
            if mode is None:
                return
            if mode == "draw":
                collector = SymbolCollector(
                    self.root,
                    label=label,
                    target_total=self.required_samples,
                    initial_total=current_total,
                    normalize_fn=self._normalize_crop,
                    save_fn=self._save_dataset_sample,
                )
                self.root.wait_window(collector)
            elif mode == "import":
                imported = self._import_samples(label)
                if imported == 0:
                    messagebox.showinfo(
                        "No images imported",
                        "No valid images were imported from the selected folder.",
                        parent=self.root,
                    )
            if self._count_label_samples(label) >= self.required_samples:
                return

    def _choose_collection_mode(self, label: str, current_total: int) -> str | None:
        dialog = tk.Toplevel(self.root)
        dialog.title("Collect samples")
        dialog.transient(self.root)
        dialog.grab_set()
        dialog.resizable(False, False)

        remaining = max(0, self.required_samples - current_total)
        ttk.Label(
            dialog,
            text=(
                f"Class '{label}' has {current_total} samples.\n"
                f"Collect {remaining} more to reach {self.required_samples}."
            ),
            justify="center",
        ).grid(row=0, column=0, columnspan=2, padx=14, pady=(14, 10))

        selection = {"mode": None}

        def choose(mode: str | None) -> None:
            selection["mode"] = mode
            dialog.destroy()

        ttk.Button(dialog, text="Draw samples now", command=lambda: choose("draw")).grid(
            row=1, column=0, sticky="ew", padx=(14, 7), pady=(0, 12)
        )
        ttk.Button(dialog, text="Import folder", command=lambda: choose("import")).grid(
            row=1, column=1, sticky="ew", padx=(7, 14), pady=(0, 12)
        )
        ttk.Button(dialog, text="Cancel", command=lambda: choose(None)).grid(
            row=2, column=0, columnspan=2, sticky="ew", padx=14, pady=(0, 14)
        )

        self.root.wait_window(dialog)
        return selection["mode"]

    def _show_noise_dialog(self, crop: np.ndarray, message: str) -> None:
        dialog = tk.Toplevel(self.root)
        dialog.title("Noisy crop")
        dialog.transient(self.root)
        dialog.grab_set()
        dialog.resizable(False, False)

        preview = self._make_preview_image(crop, size=192)
        preview_label = ttk.Label(dialog, image=preview)
        preview_label.image = preview
        preview_label.grid(row=0, column=0, padx=12, pady=(12, 6))

        ttk.Label(dialog, text=message, wraplength=220, justify="center").grid(
            row=1, column=0, padx=12, pady=(0, 8)
        )

        def discard() -> None:
            discarded_path = self._save_discarded_sample(crop)
            self.status.set(f"Discarded noisy crop: {discarded_path.name}")
            dialog.destroy()

        ttk.Button(dialog, text="Mark as Noise", command=discard).grid(
            row=2, column=0, sticky="ew", padx=12, pady=(0, 12)
        )

        self.root.wait_window(dialog)

    def _prompt_new_class_name(self, crop: np.ndarray) -> str | None:
        dialog = tk.Toplevel(self.root)
        dialog.title("Unknown component")
        dialog.transient(self.root)
        dialog.grab_set()
        dialog.resizable(False, False)

        preview = self._make_preview_image(crop, size=192)
        preview_label = ttk.Label(dialog, image=preview)
        preview_label.image = preview
        preview_label.grid(row=0, column=0, columnspan=2, padx=12, pady=(12, 6))

        ttk.Label(dialog, text="Enter a new class name:").grid(
            row=1, column=0, columnspan=2, padx=12, pady=(0, 6)
        )

        entry = ttk.Entry(dialog, width=28)
        entry.grid(row=2, column=0, columnspan=2, padx=12, pady=(0, 10))
        entry.focus_set()

        result = {"name": None}

        def submit() -> None:
            name = entry.get().strip()
            if not name:
                messagebox.showwarning("Missing name", "Please enter a class name.", parent=dialog)
                return
            result["name"] = name
            dialog.destroy()

        ttk.Button(dialog, text="Continue", command=submit).grid(
            row=3, column=0, sticky="ew", padx=(12, 6), pady=(0, 12)
        )
        ttk.Button(dialog, text="Cancel", command=dialog.destroy).grid(
            row=3, column=1, sticky="ew", padx=(6, 12), pady=(0, 12)
        )

        self.root.wait_window(dialog)
        return result["name"]

    def _import_samples(self, label: str) -> int:
        folder = filedialog.askdirectory(parent=self.root, title="Select folder of samples")
        if not folder:
            return 0
        folder_path = Path(folder)
        saved = 0
        for image_path in sorted(folder_path.iterdir()):
            if image_path.suffix.lower() not in _IMAGE_EXTENSIONS:
                continue
            try:
                image = Image.open(image_path).convert("L")
            except OSError:
                continue
            arr = np.array(image)
            if arr.mean() > 127:
                arr = 255 - arr
            normalized = self._normalize_crop(arr)
            if np.count_nonzero(normalized) == 0:
                continue
            self._save_dataset_sample(label, normalized, hard_example=False)
            saved += 1
        if saved:
            messagebox.showinfo(
                "Import complete",
                f"Imported {saved} samples into '{label}'.",
                parent=self.root,
            )
        return saved

    def _label_from_status(self, status: str) -> str:
        if status == "ambiguous":
            return "Ambiguous"
        if status in {"novel", "noisy"}:
            return "Unknown"
        return "Unknown"

    def _prepare_crop(self, gray: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray | None:
        x0 = max(0, x)
        y0 = max(0, y)
        x1 = min(gray.shape[1], x + w)
        y1 = min(gray.shape[0], y + h)
        crop = gray[y0:y1, x0:x1]
        if crop.size == 0:
            return None
        crop = 255 - crop
        return self._normalize_crop(crop)

    def _normalize_crop(self, crop_light: np.ndarray) -> np.ndarray:
        crop_light = _as_uint8(crop_light)
        ys, xs = np.where(crop_light > 0)
        if xs.size == 0 or ys.size == 0:
            return np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
        tight = crop_light[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
        height, width = tight.shape
        scale = (IMAGE_SIZE * 0.8) / max(height, width)
        new_w = max(1, int(round(width * scale)))
        new_h = max(1, int(round(height * scale)))
        resized = Image.fromarray(tight).resize((new_w, new_h), resample=Image.BILINEAR)
        canvas = Image.new("L", (IMAGE_SIZE, IMAGE_SIZE), color=0)
        offset_x = (IMAGE_SIZE - new_w) // 2
        offset_y = (IMAGE_SIZE - new_h) // 2
        canvas.paste(resized, (offset_x, offset_y))
        return np.array(canvas)

    def _make_preview_image(self, crop: np.ndarray, size: int = 192) -> ImageTk.PhotoImage:
        image = Image.fromarray(_as_uint8(crop))
        image = image.resize((size, size), resample=Image.NEAREST)
        return ImageTk.PhotoImage(image)

    def _next_filename(self, label: str, hard_example: bool) -> str:
        self._save_counter += 1
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        suffix = "_hard" if hard_example else ""
        safe_label = label.replace(" ", "_")
        return f"{safe_label}_{stamp}_{self._save_counter:03d}{suffix}.png"

    def _save_dataset_sample(self, label: str, image: np.ndarray, hard_example: bool) -> Path:
        label_dir = self.dataset_dir / label
        label_dir.mkdir(parents=True, exist_ok=True)
        filename = self._next_filename(label, hard_example)
        path = label_dir / filename
        Image.fromarray(_as_uint8(image)).save(path)
        ensure_train_val_split(self.dataset_dir)
        return path

    def _save_discarded_sample(self, image: np.ndarray) -> Path:
        filename = self._next_filename("discarded", hard_example=False)
        path = self.discarded_dir / filename
        Image.fromarray(_as_uint8(image)).save(path)
        return path

    def _count_label_samples(self, label: str) -> int:
        label_dir = self.dataset_dir / label
        if not label_dir.exists():
            return 0
        count = 0
        for path in label_dir.iterdir():
            if path.is_file() and path.suffix.lower() in _IMAGE_EXTENSIONS:
                count += 1
        for split_name in ("train", "validation"):
            split_dir = label_dir / split_name
            if not split_dir.exists():
                continue
            for path in split_dir.iterdir():
                if path.is_file() and path.suffix.lower() in _IMAGE_EXTENSIONS:
                    count += 1
        return count

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


def launch_circuit_gui(model_dir: Path, dataset_dir: Path) -> None:
    root = tk.Tk()
    app = CircuitSegmentationApp(root, model_dir=model_dir, dataset_dir=dataset_dir)
    root.mainloop()


if __name__ == "__main__":
    launch_circuit_gui(Path("./artifacts"), Path("FH_Circuit/Training_Data"))
