import tkinter as tk
from tkinter import ttk

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageTk


class CircuitSegmentationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Circuit Sketch Segmentation")

        self.canvas_width = 800
        self.canvas_height = 600

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

        self.segment_button = ttk.Button(root, text="Segment", command=self.segment_components)
        self.segment_button.grid(row=1, column=1, sticky="ew", padx=10, pady=(0, 10))

        self.last_x = None
        self.last_y = None

        self.canvas.bind("<ButtonPress-1>", self.start_draw)
        self.canvas.bind("<B1-Motion>", self.draw_line)
        self.canvas.bind("<ButtonRelease-1>", self.stop_draw)

    def start_draw(self, event):
        self.last_x = event.x
        self.last_y = event.y

    def draw_line(self, event):
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

    def stop_draw(self, _event):
        self.last_x = None
        self.last_y = None

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("RGB", (self.canvas_width, self.canvas_height), "white")
        self.draw = ImageDraw.Draw(self.image)
        cv2.destroyAllWindows()

    def segment_components(self):
        gray = np.array(self.image.convert("L"))

        inverted = 255 - gray
        density_map = cv2.blur(inverted, (40, 40))

        grabcut_mask = np.full(gray.shape, cv2.GC_PR_BGD, dtype=np.uint8)
        grabcut_mask[density_map < 10] = cv2.GC_BGD
        grabcut_mask[density_map > 50] = cv2.GC_PR_FGD

        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        color_input = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        cv2.grabCut(
            color_input,
            grabcut_mask,
            None,
            bgd_model,
            fgd_model,
            5,
            cv2.GC_INIT_WITH_MASK,
        )

        final_mask = np.where(
            (grabcut_mask == cv2.GC_FGD) | (grabcut_mask == cv2.GC_PR_FGD),
            255,
            0,
        ).astype(np.uint8)

        contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boxed = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 500:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(boxed, (x, y), (x + w, y + h), (0, 0, 255), 2)

        seed_vis = np.full(gray.shape, 127, dtype=np.uint8)
        seed_vis[grabcut_mask == cv2.GC_BGD] = 0
        seed_vis[grabcut_mask == cv2.GC_PR_FGD] = 255
        seed_vis[grabcut_mask == cv2.GC_FGD] = 255

        cv2.imshow("1. Density Input", density_map)
        cv2.imshow("2. GrabCut Seeding", seed_vis)
        cv2.imshow("3. Final Cut", final_mask)
        cv2.waitKey(1)

        self._update_canvas_from_array(boxed)

    def _update_canvas_from_array(self, array_bgr):
        array_rgb = cv2.cvtColor(array_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(array_rgb)
        self.image = pil_image
        self.draw = ImageDraw.Draw(self.image)

        self.canvas.delete("all")
        self._tk_image = ImageTk.PhotoImage(self.image)
        self.canvas.create_image(0, 0, anchor="nw", image=self._tk_image)


if __name__ == "__main__":
    root = tk.Tk()
    app = CircuitSegmentationApp(root)
    root.mainloop()
