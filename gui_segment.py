import tkinter as tk
from tkinter import ttk

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageTk

HOUGH_CANNY1 = 50
HOUGH_CANNY2 = 150
HOUGH_THRESH = 60
MIN_LINE_LEN = 80
MAX_LINE_GAP = 10
ANGLE_TOL_DEG = 12
WIRE_THICKNESS = 8


def detect_wires_hough(bw: np.ndarray):
    kernel = np.ones((3, 3), np.uint8)
    closed = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=1)
    edges = cv2.Canny(closed, HOUGH_CANNY1, HOUGH_CANNY2)

    raw_lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=HOUGH_THRESH,
        minLineLength=MIN_LINE_LEN,
        maxLineGap=MAX_LINE_GAP,
    )

    line_mask = np.zeros_like(bw)
    kept_lines = []
    if raw_lines is None:
        return line_mask, kept_lines

    for line in raw_lines:
        x1, y1, x2, y2 = line[0]
        dx = x2 - x1
        dy = y2 - y1
        length = np.hypot(dx, dy)
        if length < MIN_LINE_LEN:
            continue
        angle = abs(np.degrees(np.arctan2(dy, dx)))
        if angle > 90:
            angle = 180 - angle
        if angle < ANGLE_TOL_DEG or abs(angle - 90) < ANGLE_TOL_DEG:
            kept_lines.append((x1, y1, x2, y2))
            cv2.line(line_mask, (x1, y1), (x2, y2), 255, WIRE_THICKNESS)

    return line_mask, kept_lines


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

        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        line_mask, wire_lines = detect_wires_hough(bw)
        bw_no_wires = cv2.bitwise_and(bw, cv2.bitwise_not(line_mask))

        contours, _ = cv2.findContours(bw_no_wires, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boxed = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 500:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(boxed, (x, y), (x + w, y + h), (0, 0, 255), 2)

        overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        for x1, y1, x2, y2 in wire_lines:
            cv2.line(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.imshow("BW", bw)
        cv2.imshow("Wire Mask", line_mask)
        cv2.imshow("BW no wires", bw_no_wires)
        cv2.imshow("Wire Overlay", overlay)
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
