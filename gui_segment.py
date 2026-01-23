import tkinter as tk
from collections import deque
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

        _, dense_mask = cv2.threshold(density_map, 50, 255, cv2.THRESH_BINARY)

        kernel = np.ones((15, 15), np.uint8)
        refined = cv2.dilate(dense_mask, kernel, iterations=1)

        contours, _ = cv2.findContours(refined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boxed = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)[1]
        image_h, image_w = gray.shape
        stub_len_px = 45
        stub_r = 25
        pad_factor = 1.10
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 500:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            roi_bin = binary[y : y + h, x : x + w]
            if roi_bin.size == 0:
                continue
            roi_skel = self._skeletonize(roi_bin)
            exit_points = self._get_exit_points(roi_skel)
            protect_mask = self._stub_mask_from_exits(roi_skel, exit_points, stub_len_px, stub_r)
            roi_clean = self._remove_long_lines(roi_bin, roi_skel, protect_mask)
            tight_bbox = self._tight_bbox_from_masks(roi_clean, protect_mask, x, y, w, h)
            exit_points_full = [(x + ex, y + ey) for ex, ey in exit_points]
            padded_bbox = self._pad_bbox(tight_bbox, pad_factor, (image_w, image_h), exit_points_full)
            x_final, y_final, w_final, h_final = padded_bbox
            cv2.rectangle(
                boxed,
                (x_final, y_final),
                (x_final + w_final, y_final + h_final),
                (0, 0, 255),
                2,
            )

        contour_display = cv2.cvtColor(density_map, cv2.COLOR_GRAY2BGR)
        cv2.imshow("Density Map", contour_display)
        cv2.waitKey(1)

        self._update_canvas_from_array(boxed)

    def _skeletonize(self, binary_u8):
        working = binary_u8.copy()
        working[working > 0] = 255
        skeleton = np.zeros_like(working)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        while True:
            eroded = cv2.erode(working, element)
            temp = cv2.dilate(eroded, element)
            temp = cv2.subtract(working, temp)
            skeleton = cv2.bitwise_or(skeleton, temp)
            working = eroded.copy()
            if cv2.countNonZero(working) == 0:
                break
        return skeleton

    def _get_exit_points(self, skeleton_roi_u8):
        if skeleton_roi_u8 is None or cv2.countNonZero(skeleton_roi_u8) == 0:
            return []
        height, width = skeleton_roi_u8.shape
        border_mask = np.zeros_like(skeleton_roi_u8)
        border_mask[0, :] = 255
        border_mask[-1, :] = 255
        border_mask[:, 0] = 255
        border_mask[:, -1] = 255
        border_pixels = cv2.bitwise_and(skeleton_roi_u8, border_mask)
        if cv2.countNonZero(border_pixels) == 0:
            return []
        dilated = cv2.dilate(border_pixels, np.ones((3, 3), np.uint8), iterations=1)
        num_labels, labels = cv2.connectedComponents((dilated > 0).astype(np.uint8))
        exit_points = []
        for label in range(1, num_labels):
            ys, xs = np.where(labels == label)
            if xs.size == 0:
                continue
            cx = int(np.round(xs.mean()))
            cy = int(np.round(ys.mean()))
            exit_points.append((cx, cy))
        return exit_points

    def _stub_mask_from_exits(self, skel_roi_u8, exit_points, stub_len_px, stub_r):
        height, width = skel_roi_u8.shape
        mask = np.zeros_like(skel_roi_u8)
        if not exit_points:
            return mask
        skeleton = skel_roi_u8 > 0
        distances = np.full((height, width), -1, dtype=np.int32)
        queue = deque()
        for x, y in exit_points:
            if 0 <= x < width and 0 <= y < height:
                cv2.circle(mask, (x, y), stub_r, 255, -1)
                if skeleton[y, x]:
                    distances[y, x] = 0
                    queue.append((x, y))
        while queue:
            x, y = queue.popleft()
            dist = distances[y, x]
            if dist >= stub_len_px:
                continue
            mask[y, x] = 255
            for ny in range(y - 1, y + 2):
                for nx in range(x - 1, x + 2):
                    if nx == x and ny == y:
                        continue
                    if 0 <= nx < width and 0 <= ny < height:
                        if skeleton[ny, nx] and distances[ny, nx] == -1:
                            distances[ny, nx] = dist + 1
                            queue.append((nx, ny))
        return mask

    def _remove_long_lines(self, roi_bin, roi_skel, protect_mask):
        height, width = roi_bin.shape
        line_mask = np.zeros_like(roi_bin)
        hough_input = roi_skel if roi_skel is not None else roi_bin
        min_line_length = max(60, int(0.5 * max(height, width)))
        threshold = max(30, int(0.4 * min_line_length))
        lines = cv2.HoughLinesP(
            hough_input,
            1,
            np.pi / 180,
            threshold=threshold,
            minLineLength=min_line_length,
            maxLineGap=10,
        )
        if lines is not None:
            thickness = 6
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(line_mask, (x1, y1), (x2, y2), 255, thickness=thickness)
        if protect_mask is None:
            protect_mask = np.zeros_like(roi_bin)
        erase_mask = cv2.bitwise_and(line_mask, cv2.bitwise_not(protect_mask))
        roi_clean = roi_bin.copy()
        roi_clean[erase_mask > 0] = 0
        return roi_clean

    def _tight_bbox_from_masks(self, roi_clean, protect_mask, x, y, w, h):
        combined = cv2.bitwise_or(roi_clean, protect_mask)
        ys, xs = np.where(combined > 0)
        if xs.size == 0:
            return x, y, w, h
        x0 = x + int(xs.min())
        y0 = y + int(ys.min())
        x1 = x + int(xs.max())
        y1 = y + int(ys.max())
        return x0, y0, x1 - x0 + 1, y1 - y0 + 1

    def _pad_bbox(self, bbox, pad_factor, bounds, required_points=None):
        x, y, w, h = bbox
        if w <= 0 or h <= 0:
            return bbox
        center_x = x + w / 2
        center_y = y + h / 2
        new_w = max(1, int(round(w * pad_factor)))
        new_h = max(1, int(round(h * pad_factor)))
        x0 = int(round(center_x - new_w / 2))
        y0 = int(round(center_y - new_h / 2))
        x1 = x0 + new_w
        y1 = y0 + new_h
        if required_points:
            for px, py in required_points:
                x0 = min(x0, px)
                y0 = min(y0, py)
                x1 = max(x1, px)
                y1 = max(y1, py)
        max_w, max_h = bounds
        x0 = max(0, x0)
        y0 = max(0, y0)
        x1 = min(max_w - 1, x1)
        y1 = min(max_h - 1, y1)
        if x1 <= x0 or y1 <= y0:
            return bbox
        return x0, y0, x1 - x0, y1 - y0

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
