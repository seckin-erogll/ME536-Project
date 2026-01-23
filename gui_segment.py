import tkinter as tk
from tkinter import ttk

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageTk
from sklearn.cluster import DBSCAN

transition_list = []


def initial_filtering(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 21
    )
    transition_list.append(img)
    k3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, k3, iterations=2)
    k3_2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    transition_list.append(img)
    img = cv2.dilate(img, k3_2, iterations=2)
    return img


def cluster_anchor_points(img, original_img):
    anchor_points = cv2.goodFeaturesToTrack(
        image=img, maxCorners=100, qualityLevel=0.40, minDistance=20, blockSize=15
    )
    if anchor_points is None:
        transition_list.append(original_img.copy())
        transition_list.append(original_img.copy())
        transition_list.append(original_img.copy())
        return np.empty((0, 2)), np.array([]), np.array([]), np.array([])
    anchor_points = np.float32(anchor_points.reshape((-1, 2)))
    clustering = DBSCAN(eps=80, min_samples=4).fit(anchor_points)
    labels = clustering.labels_
    unique_labels, counts = np.unique(labels, return_counts=True)

    cluster_plot = original_img.copy()
    for label in unique_labels:
        if label == -1:
            continue
        px = [int(i[0]) for i in anchor_points[labels == label]]
        py = [int(i[1]) for i in anchor_points[labels == label]]
        color = np.random.choice(range(256), size=3)
        color = (int(color[0]), int(color[1]), int(color[2]))
        for idx in range(len(px)):
            cv2.circle(cluster_plot, (px[idx], py[idx]), 5, color=tuple(color), thickness=10)

    transition_list.append(cluster_plot)

    cluster_centers = []
    cluster_labels = []
    for label in unique_labels:
        if label == -1:
            continue
        cluster_points = anchor_points[labels == label]
        cluster_center = np.mean(cluster_points, axis=0)
        cluster_centers.append(cluster_center)
        cluster_labels.append(label)

    fast = cv2.FastFeatureDetector_create(
        threshold=20, nonmaxSuppression=False, type=cv2.FAST_FEATURE_DETECTOR_TYPE_9_16
    )
    kp = fast.detect(img, None)

    transition_list.append(cv2.drawKeypoints(original_img.copy(), kp, None, color=(255, 0, 0)))

    fast_features = [np.array([int(kp[i].pt[0]), int(kp[i].pt[1])]) for i in range(len(kp))]
    final_keypoints = anchor_points.copy()
    final_labels = labels.copy()

    for feature in fast_features:
        for j in range(len(cluster_centers)):
            if np.linalg.norm(feature - cluster_centers[j]) < 60:
                final_keypoints = np.vstack((final_keypoints, [feature]))
                final_labels = np.append(final_labels, cluster_labels[j])
                break

    final_cluster = original_img.copy()
    for label in unique_labels:
        if label == -1:
            continue
        px = [int(i[0]) for i in final_keypoints[final_labels == label]]
        py = [int(i[1]) for i in final_keypoints[final_labels == label]]
        color = np.random.choice(range(256), size=3)
        color = (int(color[0]), int(color[1]), int(color[2]))
        for idx in range(len(px)):
            cv2.circle(final_cluster, (px[idx], py[idx]), 5, color=tuple(color), thickness=10)

    transition_list.append(final_cluster)

    return final_keypoints, final_labels, unique_labels, counts


def remove_components(img, anchor_points, labels, unique_labels, counts):
    components_ext = []
    rects = []
    rects_contour = []

    ext_img = img.copy()
    for idx, _ in enumerate(counts):
        if idx == 0:
            continue
        px = [int(i[0]) for i in anchor_points[labels == unique_labels[idx]]]
        py = [int(i[1]) for i in anchor_points[labels == unique_labels[idx]]]
        if abs(max(py) - min(py)) * abs(max(px) - min(px)) > 700:
            components_ext.append(
                ext_img[min(py) - 15 : max(py) + 15, min(px) - 15 : max(px) + 15]
            )
            img[min(py) - 15 : max(py) + 15, min(px) - 15 : max(px) + 15] = 0
            rects_contour.append(
                np.array(
                    [
                        [min(px) - 15, min(py) - 15],
                        [max(px) + 15, min(py) - 15],
                        [max(px) + 15, max(py) + 15],
                        [min(px) - 15, max(py) + 15],
                    ]
                )
            )
            rects.append([min(px) - 22, max(px) + 22, min(py) - 22, max(py) + 22])

    transition_list.append(img)

    return img, rects, rects_contour, components_ext


def wire_mapping(img, rects, rects_contour, original_img):
    cnts = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_cnts = [c for c in cnts[0] if cv2.contourArea(c) > 600]

    wiring_dict = {}
    last_key = 0
    for contour in filtered_cnts:
        for rect in rects:
            for point in contour:
                if (
                    rect[0] <= point[0][0] <= rect[1]
                    and rect[2] <= point[0][1] <= rect[3]
                ):
                    wiring_dict.setdefault(last_key, []).append(rect)
                    break
        last_key += 1

    key = 0
    dict_len = len(wiring_dict)
    del_wire = []
    while key < dict_len:
        if len(wiring_dict[key]) == 1:
            del_wire.append(key)
            del wiring_dict[key]
        key += 1

    for idx in sorted(del_wire, reverse=True):
        del filtered_cnts[idx]

    cv2.drawContours(original_img, tuple(rects_contour), -1, (255, 0, 0), 3)
    cv2.drawContours(original_img, filtered_cnts, -1, (0, 255, 0), 10)

    return original_img


def driver_preprocess(img):
    transition_list.clear()
    original_img = img.copy()
    img = initial_filtering(img)
    anchor_points, labels, unique_labels, counts = cluster_anchor_points(img.copy(), original_img)
    img, rects, rects_contour, components = remove_components(
        img, anchor_points, labels, unique_labels, counts
    )
    mapped = wire_mapping(img, rects, rects_contour, original_img)
    return mapped, transition_list, components, rects


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
        rgb = np.array(self.image.convert("RGB"))
        mapped, transitions, _components, rects = driver_preprocess(rgb)

        boxed = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        for rect in rects:
            x_min, x_max, y_min, y_max = rect
            cv2.rectangle(boxed, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)

        cv2.imshow("Wire Mapping", cv2.cvtColor(mapped, cv2.COLOR_RGB2BGR))
        for idx, frame in enumerate(transitions):
            title = f"Transition {idx + 1}"
            if frame.ndim == 2:
                cv2.imshow(title, frame)
            else:
                cv2.imshow(title, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
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
