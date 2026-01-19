"""Tkinter GUI for drawing and analyzing full circuits."""

from __future__ import annotations

import json
import tkinter as tk
from pathlib import Path
from tkinter import ttk

import numpy as np
from PIL import Image, ImageDraw

from FH_Circuit.circuit_analyze import analyze_circuit, serialize_analysis


class CircuitDrawGUI:
    def __init__(self, canvas_width: int = 900, canvas_height: int = 600) -> None:
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.root = tk.Tk()
        self.root.title("Circuit Draw GUI")

        self.canvas = tk.Canvas(self.root, width=canvas_width, height=canvas_height, bg="black")
        self.canvas.grid(row=0, column=0, padx=10, pady=10, rowspan=6)

        control_frame = ttk.Frame(self.root)
        control_frame.grid(row=0, column=1, sticky="n", padx=10, pady=10)

        ttk.Label(control_frame, text="Pen thickness").grid(row=0, column=0, sticky="w")
        self.pen_width = tk.IntVar(value=6)
        ttk.Scale(
            control_frame,
            from_=2,
            to=16,
            orient="horizontal",
            variable=self.pen_width,
        ).grid(row=1, column=0, sticky="ew", pady=(0, 10))

        ttk.Button(control_frame, text="Analyze Circuit", command=self.on_analyze).grid(
            row=2, column=0, sticky="ew", pady=5
        )
        ttk.Button(control_frame, text="Clear", command=self.on_clear).grid(
            row=3, column=0, sticky="ew", pady=5
        )
        ttk.Button(control_frame, text="Save image", command=self.on_save).grid(
            row=4, column=0, sticky="ew", pady=5
        )

        self.show_bbox = tk.BooleanVar(value=True)
        self.show_terminals = tk.BooleanVar(value=True)
        self.show_nodes = tk.BooleanVar(value=True)
        self.show_corners = tk.BooleanVar(value=True)
        ttk.Checkbutton(control_frame, text="Show bboxes", variable=self.show_bbox, command=self.draw_overlays).grid(
            row=5, column=0, sticky="w"
        )
        ttk.Checkbutton(
            control_frame, text="Show terminals", variable=self.show_terminals, command=self.draw_overlays
        ).grid(row=6, column=0, sticky="w")
        ttk.Checkbutton(control_frame, text="Show wire nodes", variable=self.show_nodes, command=self.draw_overlays).grid(
            row=7, column=0, sticky="w"
        )
        ttk.Checkbutton(
            control_frame, text="Show wire corners", variable=self.show_corners, command=self.draw_overlays
        ).grid(row=8, column=0, sticky="w")

        self.status = tk.StringVar(value="Draw a full circuit, then analyze.")
        ttk.Label(control_frame, textvariable=self.status).grid(row=9, column=0, sticky="w", pady=(10, 0))

        self.details_text = tk.Text(self.root, width=40, height=20, state="disabled")
        self.details_text.grid(row=6, column=0, columnspan=2, padx=10, pady=(0, 10))

        self.image = Image.new("L", (canvas_width, canvas_height), color=0)
        self.draw = ImageDraw.Draw(self.image)
        self.last_x = None
        self.last_y = None
        self.analysis_result: dict | None = None
        self.output_dir = Path("artifacts/debug_circuit")

        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)

    def on_press(self, event: tk.Event) -> None:
        self.last_x = event.x
        self.last_y = event.y

    def on_drag(self, event: tk.Event) -> None:
        if self.last_x is None or self.last_y is None:
            return
        width = int(self.pen_width.get())
        self.canvas.create_line(self.last_x, self.last_y, event.x, event.y, fill="white", width=width)
        self.draw.line((self.last_x, self.last_y, event.x, event.y), fill=255, width=width)
        self.last_x = event.x
        self.last_y = event.y

    def on_release(self, _event: tk.Event) -> None:
        self.last_x = None
        self.last_y = None

    def on_clear(self) -> None:
        self.canvas.delete("all")
        self.image = Image.new("L", (self.canvas_width, self.canvas_height), color=0)
        self.draw = ImageDraw.Draw(self.image)
        self.analysis_result = None
        self._update_details("")
        self.status.set("Canvas cleared.")

    def on_save(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / "circuit_sketch.png"
        self.image.save(path)
        self.status.set(f"Saved image to {path}.")

    def on_analyze(self) -> None:
        sketch = np.array(self.image)
        self.analysis_result = analyze_circuit(sketch, debug_dir=self.output_dir)
        self.draw_overlays()
        serialized = serialize_analysis(self.analysis_result, include_crops=False)
        self._update_details(json.dumps(serialized, indent=2))
        self.status.set("Analysis complete (see overlays + coordinates).")

    def draw_overlays(self) -> None:
        self.canvas.delete("overlay")
        if not self.analysis_result:
            return
        if self.show_bbox.get():
            for candidate in self.analysis_result.get("symbol_candidates", []):
                x0, y0, x1, y1 = candidate["bbox"]
                self.canvas.create_rectangle(
                    x0, y0, x1, y1, outline="yellow", width=2, tags="overlay"
                )
        if self.show_terminals.get():
            for candidate in self.analysis_result.get("symbol_candidates", []):
                for x, y in candidate["terminals"]:
                    self.canvas.create_oval(
                        x - 4, y - 4, x + 4, y + 4, outline="cyan", width=2, tags="overlay"
                    )
        if self.show_nodes.get():
            for node in self.analysis_result.get("wire_graph", {}).get("nodes", []):
                x, y = node["coord"]
                color = "lime" if node["kind"] == "junction" else "orange"
                self.canvas.create_oval(
                    x - 4, y - 4, x + 4, y + 4, outline=color, width=2, tags="overlay"
                )
        if self.show_corners.get():
            for x, y in self.analysis_result.get("wire_corners", []):
                self.canvas.create_rectangle(
                    x - 3, y - 3, x + 3, y + 3, outline="red", width=2, tags="overlay"
                )

    def _update_details(self, text: str) -> None:
        self.details_text.configure(state="normal")
        self.details_text.delete("1.0", tk.END)
        self.details_text.insert(tk.END, text)
        self.details_text.configure(state="disabled")

    def run(self) -> None:
        self.root.mainloop()


def launch_circuit_gui() -> None:
    app = CircuitDrawGUI()
    app.run()
