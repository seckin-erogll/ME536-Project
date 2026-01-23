from __future__ import annotations

import cv2
import numpy as np
from sklearn.cluster import DBSCAN


CORNER_QUALITY = 0.05
CORNER_MAX = 300
CORNER_MIN_DISTANCE_FRAC = 0.02
CORNER_BLOCK_FRAC = 0.02
DBSCAN_EPS_FRAC = 0.08
DBSCAN_MIN_SAMPLES = 3
FAST_RADIUS_FRAC = 0.06
AREA_FRAC = 0.001
WIRE_AREA_FRAC = 0.0005


def _min_dim(img: np.ndarray) -> int:
    return min(img.shape[0], img.shape[1])


def _ensure_odd(value: int) -> int:
    return value if value % 2 == 1 else value + 1


def initial_filtering(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    min_dim = _min_dim(gray)
    block_size = _ensure_odd(max(11, int(min_dim * 0.05)))
    bw = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block_size,
        5,
    )
    kernel_size = max(3, int(min_dim * 0.01))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    bw = cv2.dilate(bw, dilate_kernel, iterations=1)
    return bw


def cluster_anchor_points(
    bw: np.ndarray,
    original_bgr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[int], dict[int, int], list[np.ndarray]]:
    min_dim = _min_dim(bw)
    min_distance = max(10, int(min_dim * CORNER_MIN_DISTANCE_FRAC))
    block_size = _ensure_odd(max(7, int(min_dim * CORNER_BLOCK_FRAC)))
    corners = cv2.goodFeaturesToTrack(
        bw,
        maxCorners=CORNER_MAX,
        qualityLevel=CORNER_QUALITY,
        minDistance=min_distance,
        blockSize=block_size,
    )

    debug_imgs: list[np.ndarray] = []
    corner_debug = original_bgr.copy()

    if corners is None:
        return (
            np.empty((0, 2), dtype=np.float32),
            np.empty((0,), dtype=int),
            [],
            {},
            [corner_debug],
        )

    pts = corners.reshape(-1, 2)
    for x, y in pts:
        cv2.circle(corner_debug, (int(x), int(y)), 3, (0, 0, 255), -1)
    debug_imgs.append(corner_debug)

    eps = max(5, int(min_dim * DBSCAN_EPS_FRAC))
    clusterer = DBSCAN(eps=eps, min_samples=DBSCAN_MIN_SAMPLES)
    labels = clusterer.fit_predict(pts)

    cluster_debug = original_bgr.copy()
    unique_labels = sorted({label for label in labels if label >= 0})
    for label in unique_labels:
        mask = labels == label
        color = tuple(int(c) for c in np.random.default_rng(label).integers(0, 255, size=3))
        for x, y in pts[mask]:
            cv2.circle(cluster_debug, (int(x), int(y)), 3, color, -1)
    debug_imgs.append(cluster_debug)

    fast = cv2.FastFeatureDetector_create()
    keypoints = fast.detect(bw, None)
    fast_debug = original_bgr.copy()
    for kp in keypoints:
        cv2.circle(fast_debug, (int(kp.pt[0]), int(kp.pt[1])), 3, (255, 0, 0), -1)
    debug_imgs.append(fast_debug)

    cluster_centers: list[tuple[int, np.ndarray]] = []
    counts: dict[int, int] = {}
    for label in unique_labels:
        cluster_points = pts[labels == label]
        center = cluster_points.mean(axis=0)
        cluster_centers.append((label, center))
        counts[label] = cluster_points.shape[0]

    final_pts = pts.tolist()
    final_labels = labels.tolist()
    radius = max(5, int(min_dim * FAST_RADIUS_FRAC))
    for kp in keypoints:
        pt = np.array(kp.pt, dtype=np.float32)
        nearest_label = None
        nearest_dist = None
        for label, center in cluster_centers:
            dist = np.linalg.norm(pt - center)
            if nearest_dist is None or dist < nearest_dist:
                nearest_dist = dist
                nearest_label = label
        if nearest_dist is not None and nearest_dist <= radius and nearest_label is not None:
            final_pts.append(pt.tolist())
            final_labels.append(nearest_label)
            counts[nearest_label] = counts.get(nearest_label, 0) + 1

    final_pts_arr = np.array(final_pts, dtype=np.float32)
    final_labels_arr = np.array(final_labels, dtype=int)

    final_debug = original_bgr.copy()
    for label in unique_labels:
        mask = final_labels_arr == label
        color = tuple(int(c) for c in np.random.default_rng(label + 100).integers(0, 255, size=3))
        for x, y in final_pts_arr[mask]:
            cv2.circle(final_debug, (int(x), int(y)), 3, color, -1)
    debug_imgs.append(final_debug)

    return final_pts_arr, final_labels_arr, unique_labels, counts, debug_imgs


def remove_components(
    bw: np.ndarray,
    final_pts: np.ndarray,
    final_labels: np.ndarray,
    cluster_labels: list[int],
    counts: dict[int, int],
    pad_px: int = 15,
) -> tuple[np.ndarray, list[list[int]], list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    h, w = bw.shape[:2]
    bw_wo = bw.copy()
    rects: list[list[int]] = []
    rect_contours: list[np.ndarray] = []
    crops: list[np.ndarray] = []
    debug_imgs: list[np.ndarray] = []

    area_threshold = AREA_FRAC * (w * h)
    for label in cluster_labels:
        if label not in counts:
            continue
        cluster_points = final_pts[final_labels == label]
        if cluster_points.size == 0:
            continue
        xs = cluster_points[:, 0]
        ys = cluster_points[:, 1]
        x_min = max(0, int(xs.min()) - pad_px)
        x_max = min(w - 1, int(xs.max()) + pad_px)
        y_min = max(0, int(ys.min()) - pad_px)
        y_max = min(h - 1, int(ys.max()) + pad_px)
        if (x_max - x_min) * (y_max - y_min) < area_threshold:
            continue
        crop = bw[y_min : y_max + 1, x_min : x_max + 1].copy()
        bw_wo[y_min : y_max + 1, x_min : x_max + 1] = 0
        rects.append([x_min, x_max, y_min, y_max])
        rect_contours.append(
            np.array(
                [[[x_min, y_min]], [[x_max, y_min]], [[x_max, y_max]], [[x_min, y_max]]],
                dtype=np.int32,
            )
        )
        crops.append(crop)

    debug_imgs.append(cv2.cvtColor(bw_wo, cv2.COLOR_GRAY2BGR))
    return bw_wo, rects, rect_contours, crops, debug_imgs


def wire_mapping(
    bw_wo_components: np.ndarray,
    rects: list[list[int]],
    rect_contours: list[np.ndarray],
    original_bgr: np.ndarray,
) -> np.ndarray:
    h, w = bw_wo_components.shape[:2]
    contours, _ = cv2.findContours(bw_wo_components, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area_threshold = WIRE_AREA_FRAC * (w * h)
    kept_contours = []
    for contour in contours:
        if cv2.contourArea(contour) < area_threshold:
            continue
        touched = set()
        for idx, rect in enumerate(rects):
            x_min, x_max, y_min, y_max = rect
            for point in contour:
                px, py = point[0]
                if x_min <= px <= x_max and y_min <= py <= y_max:
                    touched.add(idx)
                    break
        if len(touched) >= 2:
            kept_contours.append(contour)

    overlay = original_bgr.copy()
    if rect_contours:
        cv2.drawContours(overlay, rect_contours, -1, (0, 0, 255), 2)
    if kept_contours:
        cv2.drawContours(overlay, kept_contours, -1, (0, 255, 0), 2)
    return overlay


def driver_preprocess(
    img_bgr: np.ndarray,
) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray], list[list[int]]]:
    bw = initial_filtering(img_bgr)
    debug_images = [cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)]
    final_pts, final_labels, cluster_labels, counts, cluster_debugs = cluster_anchor_points(
        bw, img_bgr
    )
    debug_images.extend(cluster_debugs)
    bw_wo, rects, rect_contours, components, remove_debugs = remove_components(
        bw,
        final_pts,
        final_labels,
        cluster_labels,
        counts,
    )
    debug_images.extend(remove_debugs)
    overlay = wire_mapping(bw_wo, rects, rect_contours, img_bgr)
    debug_images.append(overlay.copy())
    return overlay, debug_images, components, rects
