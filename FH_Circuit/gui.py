"""Tkinter GUI for sketching circuit symbols."""

from __future__ import annotations

import random
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np
from PIL import Image, ImageDraw, ImageTk

from FH_Circuit.classify import ClassificationResult, classify_sketch_detailed, load_artifacts
from FH_Circuit.config import IMAGE_SIZE
from FH_Circuit.data import ensure_train_val_split
from FH_Circuit.train import incremental_update_pipeline

_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


class SampleCollector(tk.Toplevel):
    def __init__(
        self,
        parent: tk.Tk,
        *,
        label: str,
        target_total: int,
        initial_total: int,
        save_fn,
        canvas_size: int = 256,
    ) -> None:
        super().__init__(parent)
        self.title(f"Collect samples: {label}")
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()

        self.label = label
        self.target_total = target_total
        self.save_fn = save_fn
        self.canvas_size = canvas_size

        self.image = Image.new("L", (canvas_size, canvas_size), color=0)
        self.draw = ImageDraw.Draw(self.image)

        self.saved_total = initial_total
        self.saved_in_session = 0

        self.status_var = tk.StringVar()
        self._update_status()

        self.canvas = tk.Canvas(self, width=canvas_size, height=canvas_size, bg="black")
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
            fill="white",
            width=6,
            capstyle=tk.ROUND,
            smooth=True,
        )
        self.draw.line((self.last_x, self.last_y, event.x, event.y), fill=255, width=6)
        self.last_x = event.x
        self.last_y = event.y

    def stop_draw(self, _event: tk.Event) -> None:
        self.last_x = None
        self.last_y = None

    def clear_canvas(self) -> None:
        self.canvas.delete("all")
        self.image = Image.new("L", (self.canvas_size, self.canvas_size), color=0)
        self.draw = ImageDraw.Draw(self.image)

    def save_sample(self) -> None:
        raw = np.array(self.image)
        if np.count_nonzero(raw) == 0:
            messagebox.showwarning("No strokes", "Draw a symbol before saving.", parent=self)
            return
        self.save_fn(self.label, raw)
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


class SketchGUI:
    def __init__(
        self,
        model_dir: Path,
        dataset_dir: Path,
        canvas_size: int = 256,
        required_samples: int = 5,
    ) -> None:
        self.artifacts = load_artifacts(model_dir)
        self.dataset_dir = dataset_dir
        self.model_dir = model_dir
        self.canvas_size = canvas_size
        self.required_samples = max(1, required_samples)
        self.discarded_dir = self.model_dir / "_discarded"
        self.discarded_dir.mkdir(parents=True, exist_ok=True)
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
        resized = self.image.resize((IMAGE_SIZE, IMAGE_SIZE), resample=Image.BILINEAR)
        sketch = np.array(resized)
        try:
            result = classify_sketch_detailed(self.artifacts, sketch)
        except ValueError as exc:
            self.status.set(str(exc))
            self._update_example_from_label(None)
            return

        label = result.label
        status_message = result.message
        updated = False

        if result.status == "ambiguous":
            label = self._handle_ambiguity(sketch, result)
            if label == "Noise":
                status_message = "Ambiguity resolved: marked as noise."
            elif label:
                status_message = f"Ambiguity resolved: {label}"
        elif result.novelty_label in {"UNKNOWN", "BAD_CROP"}:
            label, updated, status_message = self._handle_novelty_or_noise(sketch, result)

        if updated:
            self.artifacts = load_artifacts(self.model_dir)
        self.status.set(status_message)
        self._update_example_from_label(label)

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

    def _update_example_from_label(self, label: str | None) -> None:
        if label and label not in {"Noise", "Unknown", "Ambiguous"}:
            self._show_random_example(label)
        else:
            self._show_random_example(None)

    def _handle_ambiguity(self, sketch: np.ndarray, result: ClassificationResult) -> str | None:
        top_candidates = result.candidates[:2]
        if not top_candidates:
            top_candidates = [(label, 0.0) for label in self.artifacts.labels[:2]]

        dialog = tk.Toplevel(self.root)
        dialog.title("Ambiguity detected")
        dialog.transient(self.root)
        dialog.grab_set()
        dialog.resizable(False, False)

        preview = self._make_preview_image(sketch, size=192)
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
            self._save_dataset_sample(selection["label"], sketch)
            return selection["label"]

        self._save_discarded_sample(sketch)
        return "Noise"

    def _handle_novelty_or_noise(
        self,
        sketch: np.ndarray,
        result: ClassificationResult,
    ) -> tuple[str | None, bool, str]:
        if result.novelty_label == "BAD_CROP":
            self._show_noise_dialog(sketch, result.message)
            return "Noise", False, "Marked as noise."

        choice = self._prompt_novelty_choice(sketch)
        if choice == "noise":
            self._save_discarded_sample(sketch)
            return "Noise", False, "Marked as noise."
        if choice != "new":
            return "Unknown", False, result.message

        class_name = self._prompt_new_class_name(sketch)
        if not class_name:
            return "Unknown", False, result.message

        self._save_dataset_sample(class_name, sketch)
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
            return class_name, False, "Saved new component samples."

        self.status.set("Updating model with new class...")
        self.root.update_idletasks()
        try:
            incremental_update_pipeline(self.dataset_dir, self.model_dir)
        except Exception as exc:  # pragma: no cover - defensive UI guard
            messagebox.showerror("Update failed", str(exc), parent=self.root)
            return class_name, False, "Incremental update failed."
        return class_name, True, f"Model updated with class '{class_name}'."

    def _show_noise_dialog(self, sketch: np.ndarray, message: str) -> None:
        dialog = tk.Toplevel(self.root)
        dialog.title("Noisy sketch")
        dialog.transient(self.root)
        dialog.grab_set()
        dialog.resizable(False, False)

        preview = self._make_preview_image(sketch, size=192)
        preview_label = ttk.Label(dialog, image=preview)
        preview_label.image = preview
        preview_label.grid(row=0, column=0, padx=12, pady=(12, 6))

        ttk.Label(dialog, text=message, wraplength=220, justify="center").grid(
            row=1, column=0, padx=12, pady=(0, 8)
        )

        ttk.Button(dialog, text="Mark as Noise", command=dialog.destroy).grid(
            row=2, column=0, sticky="ew", padx=12, pady=(0, 12)
        )

        self.root.wait_window(dialog)

    def _prompt_novelty_choice(self, sketch: np.ndarray) -> str | None:
        dialog = tk.Toplevel(self.root)
        dialog.title("Unknown component")
        dialog.transient(self.root)
        dialog.grab_set()
        dialog.resizable(False, False)

        preview = self._make_preview_image(sketch, size=192)
        preview_label = ttk.Label(dialog, image=preview)
        preview_label.image = preview
        preview_label.grid(row=0, column=0, columnspan=2, padx=12, pady=(12, 6))

        ttk.Label(dialog, text="Is this noise or a new component?").grid(
            row=1, column=0, columnspan=2, padx=12, pady=(0, 8)
        )

        selection = {"choice": None}

        def choose(choice: str | None) -> None:
            selection["choice"] = choice
            dialog.destroy()

        ttk.Button(dialog, text="New component", command=lambda: choose("new")).grid(
            row=2, column=0, sticky="ew", padx=(12, 6), pady=(0, 12)
        )
        ttk.Button(dialog, text="Noise", command=lambda: choose("noise")).grid(
            row=2, column=1, sticky="ew", padx=(6, 12), pady=(0, 12)
        )
        ttk.Button(dialog, text="Cancel", command=lambda: choose(None)).grid(
            row=3, column=0, columnspan=2, sticky="ew", padx=12, pady=(0, 12)
        )

        self.root.wait_window(dialog)
        return selection["choice"]

    def _prompt_new_class_name(self, sketch: np.ndarray) -> str | None:
        dialog = tk.Toplevel(self.root)
        dialog.title("New component")
        dialog.transient(self.root)
        dialog.grab_set()
        dialog.resizable(False, False)

        preview = self._make_preview_image(sketch, size=192)
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

    def _collect_samples_for_label(self, label: str, initial_total: int) -> None:
        while True:
            current_total = self._count_label_samples(label)
            if current_total >= self.required_samples:
                return
            mode = self._choose_collection_mode(label, current_total)
            if mode is None:
                return
            if mode == "draw":
                collector = SampleCollector(
                    self.root,
                    label=label,
                    target_total=self.required_samples,
                    initial_total=current_total,
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
            if arr.mean() < 127:
                arr = 255 - arr
            if np.count_nonzero(arr) == 0:
                continue
            self._save_dataset_sample(label, arr)
            saved += 1
        if saved:
            messagebox.showinfo(
                "Import complete",
                f"Imported {saved} samples into '{label}'.",
                parent=self.root,
            )
        return saved

    def _save_dataset_sample(self, label: str, image: np.ndarray) -> Path:
        label_dir = self.dataset_dir / label
        label_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{label.replace(' ', '_')}_{random.randint(0, 1_000_000):06d}.png"
        path = label_dir / filename
        resized = Image.fromarray(image).resize((IMAGE_SIZE, IMAGE_SIZE), resample=Image.BILINEAR)
        resized.save(path)
        ensure_train_val_split(self.dataset_dir)
        return path

    def _save_discarded_sample(self, image: np.ndarray) -> Path:
        filename = f"discarded_{random.randint(0, 1_000_000):06d}.png"
        path = self.discarded_dir / filename
        resized = Image.fromarray(image).resize((IMAGE_SIZE, IMAGE_SIZE), resample=Image.BILINEAR)
        resized.save(path)
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

    def _make_preview_image(self, sketch: np.ndarray, size: int = 192) -> ImageTk.PhotoImage:
        image = Image.fromarray(sketch)
        image = image.resize((size, size), resample=Image.NEAREST)
        return ImageTk.PhotoImage(image)

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
