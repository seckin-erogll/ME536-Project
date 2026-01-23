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
        bin_img = (inverted > 0).astype(np.uint8) * 255

        density_map = cv2.blur(inverted, (40, 40))

        _, dense_mask = cv2.threshold(density_map, 50, 255, cv2.THRESH_BINARY)

        kernel = np.ones((15, 15), np.uint8)
        refined = cv2.dilate(dense_mask, kernel, iterations=1)

        contours, _ = cv2.findContours(refined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boxed = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 500:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            x, y, w, h = self._refine_bbox_by_pixels(bin_img, x, y, w, h, pad=10)
            height, width = bin_img.shape[:2]
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
            cv2.rectangle(boxed, (x, y), (x + w, y + h), (0, 0, 255), 2)

        contour_display = cv2.cvtColor(density_map, cv2.COLOR_GRAY2BGR)
        cv2.imshow("Density Map", contour_display)
        cv2.waitKey(1)

        self._update_canvas_from_array(boxed)

    def _refine_bbox_by_pixels(self, bin_img, x, y, w, h, pad=10, close_k=5, close_iter=1):
        """
        Take ROI from bin_img (0/255). Pad it.
        Apply morphological closing (cv2.morphologyEx with MORPH_CLOSE) using a small kernel (close_k x close_k).
        Compute tight bbox from actual nonzero pixels (cv2.findNonZero or np.where).
        Return refined bbox in FULL-image coordinates (x,y,w,h).
        If no pixels, return original bbox.
        """
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

    def _refine_bbox_by_projections(self, bin_img, x, y, w, h, pad=10):
        height, width = bin_img.shape[:2]
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(width, x + w + pad)
        y1 = min(height, y + h + pad)
        crop = bin_img[y0:y1, x0:x1]
        if crop.size == 0:
            return x, y, w, h

        active_mask = crop > 0
        col_sum = np.count_nonzero(active_mask, axis=0)
        row_sum = np.count_nonzero(active_mask, axis=1)

        col_thresh = max(3, int(0.02 * crop.shape[0]))
        row_thresh = max(3, int(0.02 * crop.shape[1]))

        active_cols = np.where(col_sum >= col_thresh)[0]
        active_rows = np.where(row_sum >= row_thresh)[0]

        if active_cols.size == 0 or active_rows.size == 0:
            return x, y, w, h

        crop_x0 = int(active_cols[0])
        crop_x1 = int(active_cols[-1])
        crop_y0 = int(active_rows[0])
        crop_y1 = int(active_rows[-1])

        ref_x0 = x0 + crop_x0
        ref_y0 = y0 + crop_y0
        ref_x1 = x0 + crop_x1
        ref_y1 = y0 + crop_y1

        ref_x0, ref_y0, ref_x1, ref_y1 = self._clamp_bbox(
            ref_x0, ref_y0, ref_x1, ref_y1, width, height
        )
        return ref_x0, ref_y0, ref_x1 - ref_x0 + 1, ref_y1 - ref_y0 + 1

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


if __name__ == "__main__":
    root = tk.Tk()
    app = CircuitSegmentationApp(root)
    root.mainloop()
