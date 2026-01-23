import tkinter as tk
from tkinter import ttk

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageTk
from sklearn.cluster import DBSCAN


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
        eps = 25
        min_samples = 6
        max_expand_steps = 10
        corner_density_low = 0.0004
        corner_density_high = 0.0012
        stub_len_px = 30
        pad_factor = 1.10
        core_margin = int(1.5 * eps)

        gray = np.array(self.image.convert("L"))
        binary = self._binarize(gray)
        skeleton = self._skeletonize(binary)
        junction_mask = self._junction_mask(skeleton)

        corners = cv2.goodFeaturesToTrack(
            binary,
            maxCorners=400,
            qualityLevel=0.01,
            minDistance=8,
            blockSize=5,
            useHarrisDetector=False,
        )
        points = []
        if corners is not None:
            points = [(int(pt[0][0]), int(pt[0][1])) for pt in corners]

        boxed = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        if points:
            pts = np.array(points)
            clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(pts)
            labels = clustering.labels_
            unique_labels = {label for label in labels if label != -1}
            height, width = gray.shape

            for label in unique_labels:
                cluster_points = pts[labels == label]
                if cluster_points.size == 0:
                    continue
                min_x = int(cluster_points[:, 0].min())
                max_x = int(cluster_points[:, 0].max())
                min_y = int(cluster_points[:, 1].min())
                max_y = int(cluster_points[:, 1].max())
                bbox, exit_count = self._expand_bbox_until_exits(
                    (min_x, min_y, max_x, max_y),
                    skeleton,
                    eps,
                    max_expand_steps,
                    (width, height),
                )
                x1, y1, x2, y2 = bbox
                area = max(1, (x2 - x1 + 1) * (y2 - y1 + 1))
                corners_in_box = self._count_corners_in_bbox(points, bbox)
                corner_density = corners_in_box / area
                has_center_junction = self._has_center_junction(junction_mask, bbox)

                is_junction = (
                    exit_count >= 3
                    and has_center_junction
                    and corner_density < corner_density_low
                )
                is_component = (
                    exit_count in {2, 3, 4} and corner_density > corner_density_high
                )

                if is_junction or not is_component:
                    continue

                seed_bbox = (min_x, min_y, max_x, max_y)
                core_bbox = self._refine_core_bbox_by_local_cc(
                    binary, seed_bbox, core_margin, (width, height)
                )
                roi_skel = skeleton[y1 : y2 + 1, x1 : x2 + 1]
                exit_points = self._get_exit_points(roi_skel)
                exit_points = self._pick_two_farthest_points(exit_points)
                if len(exit_points) == 2:
                    exit_points_img = [(x1 + x, y1 + y) for x, y in exit_points]
                    sides = self._exit_sides_from_points(
                        exit_points, roi_skel.shape[1], roi_skel.shape[0]
                    )
                    extended = self._extend_bbox_by_sides(
                        core_bbox,
                        sides,
                        stub_len_px,
                        (x1, y1, x2, y2),
                        (width, height),
                    )
                    final_bbox = self._pad_bbox_centered(
                        extended, pad_factor, (width, height), exit_points_img
                    )
                else:
                    final_bbox = self._pad_bbox_centered(
                        core_bbox, pad_factor, (width, height)
                    )
                fx1, fy1, fx2, fy2 = final_bbox
                cv2.rectangle(boxed, (fx1, fy1), (fx2, fy2), (0, 0, 255), 2)

        cv2.imshow("Binary", binary)
        cv2.imshow("Skeleton", skeleton)
        cv2.imshow("Junctions", junction_mask)
        cv2.waitKey(1)

        self._update_canvas_from_array(boxed)

    def _binarize(self, gray):
        binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            21,
            5,
        )
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
        return binary

    def _skeletonize(self, binary):
        skeleton = np.zeros_like(binary)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        working = binary.copy()
        while True:
            eroded = cv2.erode(working, element)
            opened = cv2.dilate(eroded, element)
            temp = cv2.subtract(working, opened)
            skeleton = cv2.bitwise_or(skeleton, temp)
            working = eroded.copy()
            if cv2.countNonZero(working) == 0:
                break
        return skeleton

    def _junction_mask(self, skeleton):
        skel = (skeleton > 0).astype(np.uint8)
        kernel = np.ones((3, 3), np.uint8)
        neighbor_count = cv2.filter2D(skel, -1, kernel)
        neighbor_count = neighbor_count - skel
        junctions = ((skel == 1) & (neighbor_count >= 3)).astype(np.uint8) * 255
        junctions = cv2.dilate(junctions, kernel, iterations=1)
        return junctions

    def _count_border_exits(self, roi_skeleton):
        if roi_skeleton.size == 0:
            return 0
        h, w = roi_skeleton.shape
        border_mask = np.zeros((h, w), dtype=np.uint8)
        border_mask[0, :] = 255
        border_mask[-1, :] = 255
        border_mask[:, 0] = 255
        border_mask[:, -1] = 255
        border_pixels = cv2.bitwise_and(roi_skeleton, border_mask)
        if cv2.countNonZero(border_pixels) == 0:
            return 0
        border_pixels = cv2.dilate(border_pixels, np.ones((3, 3), np.uint8), iterations=1)
        count, _ = cv2.connectedComponents((border_pixels > 0).astype(np.uint8))
        return max(0, count - 1)

    def _get_exit_points(self, roi_skeleton):
        if roi_skeleton.size == 0:
            return []
        h, w = roi_skeleton.shape
        border_mask = np.zeros((h, w), dtype=np.uint8)
        border_mask[0, :] = 255
        border_mask[-1, :] = 255
        border_mask[:, 0] = 255
        border_mask[:, -1] = 255
        border_pixels = cv2.bitwise_and(roi_skeleton, border_mask)
        if cv2.countNonZero(border_pixels) == 0:
            return []
        border_pixels = cv2.dilate(border_pixels, np.ones((3, 3), np.uint8), iterations=1)
        count, labels = cv2.connectedComponents((border_pixels > 0).astype(np.uint8))
        points = []
        for label in range(1, count):
            ys, xs = np.where(labels == label)
            if ys.size == 0:
                continue
            cx = int(np.mean(xs))
            cy = int(np.mean(ys))
            points.append((cx, cy))
        return points

    def _refine_core_bbox_by_local_cc(self, binary, seed_bbox, core_margin, image_size):
        img_w, img_h = image_size
        x1, y1, x2, y2 = seed_bbox
        wx1 = max(0, x1 - core_margin)
        wy1 = max(0, y1 - core_margin)
        wx2 = min(img_w - 1, x2 + core_margin)
        wy2 = min(img_h - 1, y2 + core_margin)
        roi = binary[wy1 : wy2 + 1, wx1 : wx2 + 1]
        if cv2.countNonZero(roi) == 0:
            return seed_bbox
        count, labels = cv2.connectedComponents((roi > 0).astype(np.uint8))
        if count <= 1:
            return seed_bbox
        seed_mask = np.zeros_like(roi, dtype=np.uint8)
        sx1 = x1 - wx1
        sy1 = y1 - wy1
        sx2 = x2 - wx1
        sy2 = y2 - wy1
        sx1 = max(0, min(sx1, roi.shape[1] - 1))
        sx2 = max(0, min(sx2, roi.shape[1] - 1))
        sy1 = max(0, min(sy1, roi.shape[0] - 1))
        sy2 = max(0, min(sy2, roi.shape[0] - 1))
        seed_mask[sy1 : sy2 + 1, sx1 : sx2 + 1] = 1
        best_label = None
        best_overlap = 0
        for label in range(1, count):
            overlap = int(np.sum((labels == label) & (seed_mask > 0)))
            if overlap > best_overlap:
                best_overlap = overlap
                best_label = label
        if best_label is None or best_overlap == 0:
            return seed_bbox
        ys, xs = np.where(labels == best_label)
        if xs.size == 0 or ys.size == 0:
            return seed_bbox
        min_x = int(xs.min()) + wx1
        max_x = int(xs.max()) + wx1
        min_y = int(ys.min()) + wy1
        max_y = int(ys.max()) + wy1
        return (min_x, min_y, max_x, max_y)

    def _pick_two_farthest_points(self, pts):
        if len(pts) < 2:
            return []
        max_dist = -1
        pair = (pts[0], pts[1])
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                dx = pts[i][0] - pts[j][0]
                dy = pts[i][1] - pts[j][1]
                dist = dx * dx + dy * dy
                if dist > max_dist:
                    max_dist = dist
                    pair = (pts[i], pts[j])
        return [pair[0], pair[1]]

    def _exit_sides_from_points(self, exit_pts, roi_w, roi_h):
        sides = []
        for x, y in exit_pts:
            if x <= 1:
                sides.append("left")
            elif x >= roi_w - 2:
                sides.append("right")
            elif y <= 1:
                sides.append("top")
            elif y >= roi_h - 2:
                sides.append("bottom")
        return sides

    def _extend_bbox_by_sides(
        self, core_bbox, sides, stub_len_px, expanded_bbox, image_size
    ):
        img_w, img_h = image_size
        core_x1, core_y1, core_x2, core_y2 = core_bbox
        ex1, ey1, ex2, ey2 = expanded_bbox
        fx1, fy1, fx2, fy2 = core_bbox
        margin_x = int(0.15 * max(1, core_x2 - core_x1 + 1))
        margin_y = int(0.15 * max(1, core_y2 - core_y1 + 1))

        if "left" in sides:
            fx1 = max(ex1, core_x1 - stub_len_px)
        if "right" in sides:
            fx2 = min(ex2, core_x2 + stub_len_px)
        if "top" in sides:
            fy1 = max(ey1, core_y1 - stub_len_px)
        if "bottom" in sides:
            fy2 = min(ey2, core_y2 + stub_len_px)

        if ("left" in sides or "right" in sides) and not (
            "top" in sides or "bottom" in sides
        ):
            fy1 = max(ey1, core_y1 - margin_y)
            fy2 = min(ey2, core_y2 + margin_y)
        if ("top" in sides or "bottom" in sides) and not (
            "left" in sides or "right" in sides
        ):
            fx1 = max(ex1, core_x1 - margin_x)
            fx2 = min(ex2, core_x2 + margin_x)

        fx1 = max(0, min(fx1, img_w - 1))
        fy1 = max(0, min(fy1, img_h - 1))
        fx2 = max(0, min(fx2, img_w - 1))
        fy2 = max(0, min(fy2, img_h - 1))
        return (fx1, fy1, fx2, fy2)

    def _pad_bbox_centered(self, bbox, pad_factor, image_size, required_points=None):
        x1, y1, x2, y2 = bbox
        img_w, img_h = image_size
        width = max(1, x2 - x1 + 1)
        height = max(1, y2 - y1 + 1)
        cx = x1 + width / 2.0
        cy = y1 + height / 2.0
        new_w = width * pad_factor
        new_h = height * pad_factor
        fx1 = int(max(0, cx - new_w / 2.0))
        fy1 = int(max(0, cy - new_h / 2.0))
        fx2 = int(min(img_w - 1, cx + new_w / 2.0))
        fy2 = int(min(img_h - 1, cy + new_h / 2.0))
        if required_points:
            for ex, ey in required_points:
                if ex < fx1:
                    fx1 = ex
                if ex > fx2:
                    fx2 = ex
                if ey < fy1:
                    fy1 = ey
                if ey > fy2:
                    fy2 = ey
        fx1 = max(0, min(fx1, img_w - 1))
        fy1 = max(0, min(fy1, img_h - 1))
        fx2 = max(0, min(fx2, img_w - 1))
        fy2 = max(0, min(fy2, img_h - 1))
        return (fx1, fy1, fx2, fy2)

    def _expand_bbox_until_exits(self, bbox, skeleton, eps, max_steps, image_size):
        x1, y1, x2, y2 = bbox
        width = max(1, x2 - x1 + 1)
        height = max(1, y2 - y1 + 1)
        max_margin = int(2.0 * max(width, height))
        img_w, img_h = image_size
        last_exit_count = 0
        for step in range(1, max_steps + 1):
            margin = min(int(step * eps), max_margin)
            ex1 = max(0, x1 - margin)
            ey1 = max(0, y1 - margin)
            ex2 = min(img_w - 1, x2 + margin)
            ey2 = min(img_h - 1, y2 + margin)
            roi = skeleton[ey1 : ey2 + 1, ex1 : ex2 + 1]
            exit_count = self._count_border_exits(roi)
            last_exit_count = exit_count
            if exit_count > 0:
                return (ex1, ey1, ex2, ey2), exit_count
        return (x1, y1, x2, y2), last_exit_count

    def _count_corners_in_bbox(self, points, bbox):
        x1, y1, x2, y2 = bbox
        return sum(1 for x, y in points if x1 <= x <= x2 and y1 <= y <= y2)

    def _has_center_junction(self, junction_mask, bbox):
        x1, y1, x2, y2 = bbox
        width = max(1, x2 - x1 + 1)
        height = max(1, y2 - y1 + 1)
        cx1 = int(x1 + 0.35 * width)
        cx2 = int(x2 - 0.35 * width)
        cy1 = int(y1 + 0.35 * height)
        cy2 = int(y2 - 0.35 * height)
        roi = junction_mask[cy1 : cy2 + 1, cx1 : cx2 + 1]
        if roi.size == 0:
            return False
        return cv2.countNonZero(roi) > 0

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
