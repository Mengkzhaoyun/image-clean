from __future__ import annotations

import cv2
import numpy as np


def build_vertical_dark_text_mask(
    image: np.ndarray,
    anchor_mask: np.ndarray,
    enabled: bool,
    dilate: int,
    min_area: int,
    max_area_ratio: float,
    edge_ratio: float,
    anchor_radius: int,
) -> tuple[np.ndarray, int]:
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    if not enabled:
        return mask, 0

    height, width = image.shape[:2]
    candidate = _adaptive_dark_stroke_candidates(image)
    anchor = _build_anchor_mask(anchor_mask, radius=anchor_radius)

    num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(candidate, connectivity=8)
    count = 0
    max_area = int(height * width * max_area_ratio)
    edge_margin = int(width * edge_ratio)

    for label in range(1, num_labels):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])
        if not _is_likely_vertical_text_component(
            image_shape=image.shape,
            x=x,
            y=y,
            w=w,
            h=h,
            area=area,
            min_area=min_area,
            max_area=max_area,
            edge_margin=edge_margin,
        ):
            continue
        if not np.any(anchor[y : y + h, x : x + w]):
            continue

        component = (labels == label).astype(np.uint8) * 255
        mask = cv2.bitwise_or(mask, component)
        count += 1

    if not np.any(mask):
        return mask, 0

    close_h = max(9, dilate * 3)
    close_w = max(3, dilate)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (close_w, close_h))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    if dilate > 0:
        kernel_size = dilate * 2 + 1
        mask = cv2.dilate(
            mask,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)),
            iterations=1,
        )

    return mask, count


def build_vertical_text_column_mask(
    image: np.ndarray,
    enabled: bool,
    dilate: int,
    min_height_ratio: float,
    max_width_ratio: float,
    edge_ratio: float,
) -> tuple[np.ndarray, int]:
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    if not enabled:
        return mask, 0

    height, width = image.shape[:2]
    candidate = _dark_stroke_candidates(image)
    candidate = cv2.morphologyEx(
        candidate,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (9, 21)),
        iterations=1,
    )
    col_counts = np.count_nonzero(candidate, axis=0)
    threshold = max(3, int(height * 0.012))
    active_cols = col_counts >= threshold

    count = 0
    raw_runs = _runs(active_cols)
    candidate_runs: list[tuple[int, int]] = []
    for x1, x2 in raw_runs:
        col_w = x2 - x1
        if col_w < 6 or col_w > int(width * max_width_ratio):
            continue

        roi = candidate[:, x1:x2]
        rows = np.where(np.any(roi > 0, axis=1))[0]
        if rows.size == 0:
            continue
        y1 = int(rows[0])
        y2 = int(rows[-1]) + 1
        col_h = y2 - y1
        if col_h < int(height * min_height_ratio):
            continue

        center_x = (x1 + x2) / 2
        in_text_band = center_x <= width * edge_ratio or center_x >= width * (1 - edge_ratio)
        if not in_text_band:
            continue
        candidate_runs.append((x1, x2))

    max_width = int(width * max_width_ratio)
    for x1, x2 in _merge_runs_limited(candidate_runs, max_gap=max(16, dilate * 3), max_width=max_width):
        x_pad = max(dilate * 2, 8)
        y_pad = max(dilate * 2, 8)
        roi = candidate[:, x1:x2]
        rows = np.where(np.any(roi > 0, axis=1))[0]
        if rows.size == 0:
            continue
        y1 = int(rows[0])
        y2 = int(rows[-1]) + 1
        bx1 = max(x1 - x_pad, 0)
        bx2 = min(x2 + x_pad, width)
        by1 = max(y1 - y_pad, 0)
        by2 = min(y2 + y_pad, height)
        mask[by1:by2, bx1:bx2] = 255
        count += 1

    if count == 0:
        return mask, 0

    if dilate > 0:
        kernel_size = dilate * 2 + 1
        mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)), iterations=1)

    return mask, count


def _dark_stroke_candidates(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]

    dark = cv2.inRange(gray, 0, 95)
    low_saturation = cv2.inRange(saturation, 0, 120)
    candidate = cv2.bitwise_and(dark, low_saturation)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, kernel, iterations=1)
    return candidate


def _adaptive_dark_stroke_candidates(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    adaptive = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        13,
    )
    not_too_light = cv2.inRange(gray, 0, 180)
    candidate = cv2.bitwise_and(adaptive, not_too_light)
    candidate = cv2.morphologyEx(
        candidate,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)),
        iterations=1,
    )
    return candidate


def _is_likely_vertical_text_component(
    image_shape: tuple[int, ...],
    x: int,
    y: int,
    w: int,
    h: int,
    area: int,
    min_area: int,
    max_area: int,
    edge_margin: int,
) -> bool:
    height, width = image_shape[:2]
    if area < min_area or area > max_area:
        return False
    if w < 3 or h < 8:
        return False
    if w > width * 0.22 or h > height * 0.45:
        return False

    fill_ratio = area / max(w * h, 1)
    if fill_ratio > 0.78:
        return False

    center_x = x + w / 2
    center_y = y + h / 2
    in_edge_or_poster_band = center_x <= edge_margin or center_x >= width - edge_margin or center_y <= height * 0.2
    if not in_edge_or_poster_band:
        return False

    aspect = max(w / max(h, 1), h / max(w, 1))
    return aspect <= 8.0


def _build_anchor_mask(anchor_mask: np.ndarray, radius: int) -> np.ndarray:
    if not np.any(anchor_mask):
        return np.zeros(anchor_mask.shape, dtype=np.uint8)

    radius = max(int(radius), 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
    return cv2.dilate(anchor_mask, kernel, iterations=1)


def _runs(active: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(active.tolist()):
        if value and start is None:
            start = index
        elif not value and start is not None:
            runs.append((start, index))
            start = None
    if start is not None:
        runs.append((start, len(active)))
    return runs


def _merge_runs(runs: list[tuple[int, int]], max_gap: int) -> list[tuple[int, int]]:
    if not runs:
        return []

    merged: list[tuple[int, int]] = []
    current_start, current_end = runs[0]
    for start, end in runs[1:]:
        if start - current_end <= max_gap:
            current_end = end
        else:
            merged.append((current_start, current_end))
            current_start, current_end = start, end
    merged.append((current_start, current_end))
    return merged


def _merge_runs_limited(runs: list[tuple[int, int]], max_gap: int, max_width: int) -> list[tuple[int, int]]:
    if not runs:
        return []

    merged: list[tuple[int, int]] = []
    current_start, current_end = runs[0]
    for start, end in runs[1:]:
        if start - current_end <= max_gap and end - current_start <= max_width:
            current_end = end
        else:
            merged.append((current_start, current_end))
            current_start, current_end = start, end
    merged.append((current_start, current_end))
    return merged
