from __future__ import annotations

import cv2
import numpy as np

from .models import TextDetection


def build_text_mask(shape: tuple[int, ...], detections: list[TextDetection], dilate: int) -> np.ndarray:
    mask = np.zeros(shape[:2], dtype=np.uint8)
    for detection in detections:
        points = np.array(detection.box, dtype=np.int32)
        cv2.fillPoly(mask, [points], 255)

    if dilate > 0 and np.any(mask):
        kernel_size = dilate * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        mask = cv2.dilate(mask, kernel, iterations=1)

    return mask


def build_text_block_expansion_mask(
    image: np.ndarray,
    anchor_mask: np.ndarray,
    enabled: bool,
    dilate: int,
    min_area: int,
    max_area_ratio: float,
) -> tuple[np.ndarray, int]:
    mask = np.zeros(anchor_mask.shape[:2], dtype=np.uint8)
    if not enabled or not np.any(anchor_mask):
        return mask, 0

    height, width = anchor_mask.shape[:2]
    image_area = height * width
    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (max(3, dilate * 4 + 1), max(3, dilate * 2 + 1)),
    )
    grouped = cv2.morphologyEx(anchor_mask, cv2.MORPH_CLOSE, close_kernel, iterations=1)
    num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats((grouped > 0).astype(np.uint8), connectivity=8)
    max_area = image_area * max_area_ratio
    count = 0
    contrast = _local_text_stroke_candidates(image)

    for label in range(1, num_labels):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area or area > max_area:
            continue
        if w < 4 or h < 4:
            continue

        center_x = x + w / 2
        center_y = y + h / 2
        in_lower_title_band = center_y >= height * 0.56 and w >= width * 0.16
        in_side_vertical_band = (
            (center_x <= width * 0.28 or center_x >= width * 0.72)
            and h >= height * 0.08
            and h >= w * 1.2
        )
        in_large_title = area >= image_area * 0.008 or w >= width * 0.28
        if not (in_lower_title_band or in_side_vertical_band or in_large_title):
            continue

        pad_x = max(dilate * 2, int(round(w * 0.12)))
        pad_y = max(dilate * 2, int(round(h * 0.18)))
        if in_lower_title_band:
            pad_x = max(pad_x, int(round(width * 0.035)))
            pad_y = max(pad_y, int(round(height * 0.025)))
        if in_side_vertical_band:
            pad_x = max(pad_x, int(round(width * 0.025)))
            pad_y = max(pad_y, int(round(height * 0.02)))

        x1 = max(x - pad_x, 0)
        y1 = max(y - pad_y, 0)
        x2 = min(x + w + pad_x, width)
        y2 = min(y + h + pad_y, height)
        roi_contrast = contrast[y1:y2, x1:x2]
        roi_candidate = roi_contrast
        if in_lower_title_band:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(7, dilate * 3 + 1), max(3, dilate // 2 + 1)))
        elif in_side_vertical_band:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(3, dilate // 2 + 1), max(7, dilate * 3 + 1)))
        else:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(3, dilate + 1), max(3, dilate + 1)))
        roi_candidate = cv2.morphologyEx(roi_candidate, cv2.MORPH_CLOSE, kernel, iterations=1)
        roi_candidate = cv2.dilate(
            roi_candidate,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(3, dilate // 2 + 1), max(3, dilate // 2 + 1))),
            iterations=1,
        )
        mask[y1:y2, x1:x2] = cv2.bitwise_or(mask[y1:y2, x1:x2], roi_candidate)
        count += 1

    if np.any(mask):
        mask = cv2.bitwise_and(mask, cv2.bitwise_not(anchor_mask))

    return mask, count


def build_vertical_bright_text_mask(
    image: np.ndarray,
    enabled: bool,
    dilate: int,
    min_components: int,
    min_span_ratio: float,
    max_width_ratio: float,
) -> tuple[np.ndarray, int]:
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    if not enabled:
        return mask, 0

    height, width = image.shape[:2]
    candidate = _local_bright_text_candidates(image)
    candidate[: int(height * 0.08), :] = 0
    candidate[int(height * 0.72) :, :] = 0
    candidate[:, int(width * 0.42) : int(width * 0.62)] = 0
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)), iterations=1)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(candidate, connectivity=8)
    components: list[tuple[int, int, int, int, int, float]] = []
    for label in range(1, num_labels):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < 4 or area > height * width * 0.006:
            continue
        if w < 2 or h < 2 or w > width * 0.12 or h > height * 0.16:
            continue
        fill_ratio = area / max(w * h, 1)
        if fill_ratio > 0.82:
            continue
        components.append((label, x, y, w, h, float(centroids[label][0])))

    count = 0
    used: set[int] = set()
    max_column_width = max(10, int(width * max_width_ratio))
    min_span = max(24, int(height * min_span_ratio))
    for label, _x, _y, _w, _h, center_x in components:
        if label in used:
            continue
        column = [
            item
            for item in components
            if abs(item[5] - center_x) <= max_column_width and item[0] not in used
        ]
        if len(column) < min_components:
            continue
        xs: list[int] = []
        ys: list[int] = []
        y2s: list[int] = []
        for _label, x, y, w, h, _center_x in column:
            xs.extend([x, x + w])
            ys.append(y)
            y2s.append(y + h)
        span = max(y2s) - min(ys)
        col_width = max(xs) - min(xs)
        if span < min_span or col_width > max_column_width * 3:
            continue
        x1 = max(min(xs) - max(dilate, 6), 0)
        x2 = min(max(xs) + max(dilate, 6), width)
        y1 = max(min(ys) - max(dilate * 2, 10), 0)
        y2 = min(max(y2s) + max(dilate * 2, 10), height)
        roi = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
        for selected_label, *_rest in column:
            roi[labels[y1:y2, x1:x2] == selected_label] = 255
            used.add(selected_label)
        roi = cv2.morphologyEx(
            roi,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (max(3, dilate), max(9, dilate * 4 + 1))),
            iterations=1,
        )
        roi = cv2.dilate(
            roi,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(3, dilate + 1), max(3, dilate + 1))),
            iterations=1,
        )
        mask[y1:y2, x1:x2] = cv2.bitwise_or(mask[y1:y2, x1:x2], roi)
        count += 1

    return mask, count


def build_corner_watermark_mask(
    image: np.ndarray,
    text_mask: np.ndarray,
    enabled: bool,
    corner_ratio: float,
    min_area: int,
    dilate: int,
    text_near_radius: int,
) -> tuple[np.ndarray, int]:
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    if not enabled:
        return mask, 0

    height, width = image.shape[:2]
    corner_w = max(1, int(width * corner_ratio))
    corner_h = max(1, int(height * corner_ratio))
    corners = [
        (0, 0, corner_w, corner_h),
        (width - corner_w, 0, corner_w, corner_h),
        (0, height - corner_h, corner_w, corner_h),
        (width - corner_w, height - corner_h, corner_w, corner_h),
    ]

    count = 0
    for x, y, w, h in corners:
        text_roi = text_mask[y : y + h, x : x + w]
        if not np.any(text_roi):
            continue

        roi = image[y : y + h, x : x + w]
        roi_mask = detect_colored_watermark_region(roi, min_area=min_area)
        if not np.any(roi_mask):
            continue

        nearby_text = cv2.dilate(
            text_roi,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (text_near_radius * 2 + 1, text_near_radius * 2 + 1),
            ),
            iterations=1,
        )
        roi_mask = cv2.bitwise_and(roi_mask, nearby_text)
        if not np.any(roi_mask):
            continue

        mask[y : y + h, x : x + w] = cv2.bitwise_or(mask[y : y + h, x : x + w], roi_mask)
        count += 1

    if dilate > 0 and np.any(mask):
        kernel_size = dilate * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        mask = cv2.dilate(mask, kernel, iterations=1)

    return mask, count


def build_colored_sticker_mask(
    image: np.ndarray,
    text_mask: np.ndarray,
    enabled: bool,
    min_area: int,
    dilate: int,
    text_near_radius: int,
) -> tuple[np.ndarray, int]:
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    if not enabled or not np.any(text_mask):
        return mask, 0

    colored_mask = detect_colored_watermark_region(image, min_area=min_area, max_area_ratio=0.08)
    if not np.any(colored_mask):
        return mask, 0

    anchor_text = cv2.dilate(
        text_mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (text_near_radius * 2 + 1, text_near_radius * 2 + 1)),
        iterations=1,
    )

    num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(colored_mask, connectivity=8)
    count = 0
    for label in range(1, num_labels):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        if not is_likely_sticker_position(image.shape, x=x, y=y, w=w, h=h):
            continue

        text_inside_bbox = anchor_text[y : y + h, x : x + w]
        if not np.any(text_inside_bbox):
            continue

        aspect = max(w / max(h, 1), h / max(w, 1))
        if aspect > 4.0:
            continue

        component = (labels == label).astype(np.uint8) * 255
        mask = cv2.bitwise_or(mask, component)
        count += 1

    if dilate > 0 and np.any(mask):
        kernel_size = dilate * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        mask = cv2.dilate(mask, kernel, iterations=1)

    return mask, count


def build_dark_stroke_refinement_mask(
    image: np.ndarray,
    anchor_mask: np.ndarray,
    enabled: bool,
    edge_ratio: float,
    anchor_radius: int,
    min_area: int,
    max_area_ratio: float,
    dilate: int,
) -> tuple[np.ndarray, int]:
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    if not enabled or not np.any(anchor_mask):
        return mask, 0

    height, width = image.shape[:2]
    edge_w = max(1, int(width * edge_ratio))
    edge_band = np.zeros((height, width), dtype=np.uint8)
    edge_band[:, :edge_w] = 255
    edge_band[:, width - edge_w :] = 255

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    dark = cv2.bitwise_and(cv2.inRange(gray, 0, 120), cv2.inRange(saturation, 0, 90))
    dark = cv2.bitwise_and(dark, cv2.inRange(value, 0, 180))
    dark = cv2.bitwise_and(dark, edge_band)

    anchor = cv2.dilate(
        anchor_mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (anchor_radius * 2 + 1, anchor_radius * 2 + 1)),
        iterations=1,
    )
    dark = cv2.bitwise_and(dark, anchor)
    if not np.any(dark):
        return mask, 0

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, kernel, iterations=1)
    num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(dark, connectivity=8)
    max_area = height * width * max_area_ratio
    count = 0
    for label in range(1, num_labels):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area or area > max_area:
            continue
        if h < 3 or w < 2:
            continue
        if x > edge_w and x + w < width - edge_w:
            continue

        component = (labels == label).astype(np.uint8) * 255
        mask = cv2.bitwise_or(mask, component)
        count += 1

    if dilate > 0 and np.any(mask):
        kernel_size = max(3, dilate * 2 + 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        mask = cv2.dilate(mask, kernel, iterations=1)

    return mask, count


def build_edge_column_refinement_mask(
    image: np.ndarray,
    anchor_mask: np.ndarray,
    enabled: bool,
    edge_ratio: float,
    anchor_radius: int,
    min_area: int,
    max_area_ratio: float,
    dilate: int,
) -> tuple[np.ndarray, int]:
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    if not enabled or not np.any(anchor_mask):
        return mask, 0

    height, width = image.shape[:2]
    edge_w = max(1, int(width * edge_ratio))
    edge_band = np.zeros((height, width), dtype=np.uint8)
    edge_band[:, :edge_w] = 255
    edge_band[:, width - edge_w :] = 255

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    dark = cv2.bitwise_and(cv2.inRange(gray, 0, 135), cv2.inRange(saturation, 0, 110))
    dark = cv2.bitwise_and(dark, cv2.inRange(value, 0, 190))
    dark = cv2.bitwise_and(dark, edge_band)

    anchor = cv2.dilate(
        anchor_mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (anchor_radius * 2 + 1, anchor_radius * 2 + 1)),
        iterations=1,
    )
    dark = cv2.bitwise_and(dark, anchor)
    dark = cv2.bitwise_and(dark, cv2.bitwise_not(anchor_mask))
    if not np.any(dark):
        return mask, 0

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, kernel, iterations=1)
    num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(dark, connectivity=8)
    max_area = height * width * max_area_ratio
    count = 0
    for label in range(1, num_labels):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area or area > max_area:
            continue
        if w < 2 or h < 2:
            continue
        if x > edge_w and x + w < width - edge_w:
            continue
        aspect = max(w / max(h, 1), h / max(w, 1))
        if aspect > 18:
            continue

        pad = max(anchor_radius // 2, 8)
        x1 = max(x - pad, 0)
        y1 = max(y - pad, 0)
        x2 = min(x + w + pad, width)
        y2 = min(y + h + pad, height)
        nearby_anchor_pixels = int(np.count_nonzero(anchor_mask[y1:y2, x1:x2]))
        if nearby_anchor_pixels < max(20, area):
            continue

        component = (labels == label).astype(np.uint8) * 255
        mask = cv2.bitwise_or(mask, component)
        count += 1

    if dilate > 0 and np.any(mask):
        kernel_size = max(3, dilate * 2 + 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        mask = cv2.dilate(mask, kernel, iterations=1)

    return mask, count


def build_bright_text_refinement_mask(
    image: np.ndarray,
    anchor_mask: np.ndarray,
    enabled: bool,
    edge_ratio: float,
    anchor_radius: int,
    min_area: int,
    max_area_ratio: float,
    dilate: int,
) -> tuple[np.ndarray, int]:
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    if not enabled or not np.any(anchor_mask):
        return mask, 0

    height, width = image.shape[:2]
    edge_w = max(1, int(width * edge_ratio))
    edge_band = np.zeros((height, width), dtype=np.uint8)
    edge_band[:, :edge_w] = 255
    edge_band[:, width - edge_w :] = 255
    edge_band[int(height * 0.58) :, :] = 255

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    bright = cv2.bitwise_and(cv2.inRange(gray, 168, 255), cv2.inRange(value, 175, 255))
    bright = cv2.bitwise_and(bright, cv2.inRange(saturation, 0, 145))
    bright = cv2.bitwise_and(bright, edge_band)

    anchor = cv2.dilate(
        anchor_mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (anchor_radius * 2 + 1, anchor_radius * 2 + 1)),
        iterations=1,
    )
    bright = cv2.bitwise_and(bright, anchor)
    bright = cv2.bitwise_and(bright, cv2.bitwise_not(anchor_mask))
    if not np.any(bright):
        return mask, 0

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, kernel, iterations=1)
    num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(bright, connectivity=8)
    max_area = height * width * max_area_ratio
    count = 0
    for label in range(1, num_labels):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area or area > max_area:
            continue
        if w < 2 or h < 2:
            continue
        aspect = max(w / max(h, 1), h / max(w, 1))
        if aspect > 22:
            continue

        pad = max(anchor_radius // 2, 8)
        x1 = max(x - pad, 0)
        y1 = max(y - pad, 0)
        x2 = min(x + w + pad, width)
        y2 = min(y + h + pad, height)
        nearby_anchor_pixels = int(np.count_nonzero(anchor_mask[y1:y2, x1:x2]))
        if nearby_anchor_pixels < max(16, area // 2):
            continue

        component = (labels == label).astype(np.uint8) * 255
        mask = cv2.bitwise_or(mask, component)
        count += 1

    if dilate > 0 and np.any(mask):
        kernel_size = max(3, dilate * 2 + 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        mask = cv2.dilate(mask, kernel, iterations=1)

    return mask, count


def build_text_stroke_completion_mask(
    image: np.ndarray,
    anchor_mask: np.ndarray,
    enabled: bool,
    anchor_radius: int,
    min_area: int,
    max_area_ratio: float,
    dilate: int,
) -> tuple[np.ndarray, int]:
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    if not enabled or not np.any(anchor_mask):
        return mask, 0

    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    bright = cv2.bitwise_and(cv2.inRange(gray, 155, 255), cv2.inRange(value, 165, 255))
    bright = cv2.bitwise_and(bright, cv2.inRange(saturation, 0, 165))
    dark = cv2.bitwise_and(cv2.inRange(gray, 0, 125), cv2.inRange(value, 0, 175))
    dark = cv2.bitwise_and(dark, cv2.inRange(saturation, 0, 155))
    high_contrast = cv2.bitwise_or(bright, dark)

    anchor = cv2.dilate(
        anchor_mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (anchor_radius * 2 + 1, anchor_radius * 2 + 1)),
        iterations=1,
    )
    candidate = cv2.bitwise_and(high_contrast, anchor)
    candidate = cv2.bitwise_and(candidate, cv2.bitwise_not(anchor_mask))
    if not np.any(candidate):
        return mask, 0

    candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)), iterations=1)
    num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(candidate, connectivity=8)
    max_area = height * width * max_area_ratio
    count = 0
    for label in range(1, num_labels):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area or area > max_area:
            continue
        if w < 2 or h < 2:
            continue
        aspect = max(w / max(h, 1), h / max(w, 1))
        if aspect > 28:
            continue

        pad = max(anchor_radius // 3, 10)
        x1 = max(x - pad, 0)
        y1 = max(y - pad, 0)
        x2 = min(x + w + pad, width)
        y2 = min(y + h + pad, height)
        nearby_anchor_pixels = int(np.count_nonzero(anchor_mask[y1:y2, x1:x2]))
        if nearby_anchor_pixels < max(10, area // 3):
            continue

        component = (labels == label).astype(np.uint8) * 255
        mask = cv2.bitwise_or(mask, component)
        count += 1

    if dilate > 0 and np.any(mask):
        kernel_size = max(3, dilate * 2 + 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        mask = cv2.dilate(mask, kernel, iterations=1)

    return mask, count


def build_post_dark_residual_mask(
    image: np.ndarray,
    anchor_mask: np.ndarray,
    enabled: bool,
    edge_ratio: float,
    anchor_radius: int,
    min_area: int,
    max_area_ratio: float,
    dilate: int,
) -> tuple[np.ndarray, int]:
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    if not enabled or not np.any(anchor_mask):
        return mask, 0

    height, width = image.shape[:2]
    edge_w = max(1, int(width * edge_ratio))
    edge_band = np.zeros((height, width), dtype=np.uint8)
    edge_band[:, :edge_w] = 255
    edge_band[:, width - edge_w :] = 255

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    dark = cv2.bitwise_and(cv2.inRange(gray, 0, 120), cv2.inRange(saturation, 0, 105))
    dark = cv2.bitwise_and(dark, cv2.inRange(value, 0, 180))
    dark = cv2.bitwise_and(dark, edge_band)

    anchor = cv2.dilate(
        anchor_mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (anchor_radius * 2 + 1, anchor_radius * 2 + 1)),
        iterations=1,
    )
    dark = cv2.bitwise_and(dark, anchor)
    if not np.any(dark):
        return mask, 0

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, kernel, iterations=1)
    num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(dark, connectivity=8)
    max_area = height * width * max_area_ratio
    count = 0
    for label in range(1, num_labels):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area or area > max_area:
            continue
        if w < 2 or h < 2:
            continue
        if x > edge_w and x + w < width - edge_w:
            continue
        aspect = max(w / max(h, 1), h / max(w, 1))
        if aspect > 16:
            continue

        pad = max(anchor_radius // 2, 8)
        x1 = max(x - pad, 0)
        y1 = max(y - pad, 0)
        x2 = min(x + w + pad, width)
        y2 = min(y + h + pad, height)
        nearby_anchor_pixels = int(np.count_nonzero(anchor_mask[y1:y2, x1:x2]))
        if nearby_anchor_pixels < max(20, area):
            continue

        box_pad = max(dilate, 4)
        x1 = max(x - box_pad, 0)
        y1 = max(y - box_pad, 0)
        x2 = min(x + w + box_pad, width)
        y2 = min(y + h + box_pad, height)
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, thickness=-1)
        count += 1

    if dilate > 0 and np.any(mask):
        kernel_size = max(3, dilate * 2 + 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        mask = cv2.dilate(mask, kernel, iterations=1)

    return mask, count


def build_empty_edge_dark_text_mask(
    image: np.ndarray,
    enabled: bool,
    edge_ratio: float,
    min_area: int,
    dilate: int,
) -> tuple[np.ndarray, int]:
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    if not enabled:
        return mask, 0

    height, width = image.shape[:2]
    edge_w = max(1, int(width * edge_ratio))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    value = hsv[:, :, 2]
    dark = cv2.bitwise_or(cv2.inRange(gray, 0, 115), cv2.inRange(value, 0, 135))

    count = 0
    for x_offset, roi_dark in ((0, dark[:, :edge_w]), (width - edge_w, dark[:, width - edge_w :])):
        roi_mask, roi_count = _build_dark_vertical_columns_from_edge_roi(
            roi_dark=roi_dark,
            min_area=min_area,
            min_span=max(18, int(height * 0.24)),
            max_column_width=max(12, int(width * 0.12)),
        )
        if roi_count == 0:
            continue
        mask[:, x_offset : x_offset + edge_w] = cv2.bitwise_or(mask[:, x_offset : x_offset + edge_w], roi_mask)
        count += roi_count

    if dilate > 0 and np.any(mask):
        kernel_size = max(3, dilate * 2 + 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        mask = cv2.dilate(mask, kernel, iterations=1)

    return mask, count


def _build_dark_vertical_columns_from_edge_roi(
    roi_dark: np.ndarray,
    min_area: int,
    min_span: int,
    max_column_width: int,
) -> tuple[np.ndarray, int]:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 3))
    candidate = cv2.morphologyEx(roi_dark, cv2.MORPH_OPEN, kernel, iterations=1)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(candidate, connectivity=8)
    components: list[tuple[int, int, int, int, int, float]] = []
    for label in range(1, num_labels):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area or area > roi_dark.size * 0.08:
            continue
        if w < 1 or h < 2:
            continue
        if w > max_column_width or h > roi_dark.shape[0] * 0.45:
            continue
        components.append((label, x, y, w, h, float(centroids[label][0])))

    selected = np.zeros_like(roi_dark)
    used_labels: set[int] = set()
    count = 0
    for label, x, y, w, h, center_x in components:
        if label in used_labels:
            continue
        column_labels = []
        xs = []
        ys = []
        y_ends = []
        total_area = 0
        for other_label, other_x, other_y, other_w, other_h, other_center_x in components:
            if abs(other_center_x - center_x) > max_column_width:
                continue
            column_labels.append(other_label)
            xs.extend([other_x, other_x + other_w])
            ys.append(other_y)
            y_ends.append(other_y + other_h)
            total_area += int(stats[other_label, cv2.CC_STAT_AREA])

        if len(column_labels) < 3:
            continue
        span = max(y_ends) - min(ys)
        column_width = max(xs) - min(xs)
        if span < min_span or column_width > max_column_width * 2:
            continue
        if total_area < min_area * 4:
            continue

        for selected_label in column_labels:
            selected[labels == selected_label] = 255
            used_labels.add(selected_label)
        count += 1

    return selected, count


def is_likely_sticker_position(shape: tuple[int, ...], x: int, y: int, w: int, h: int) -> bool:
    height, width = shape[:2]
    center_x = x + w / 2
    center_y = y + h / 2

    margin_x = width * 0.18
    margin_y = height * 0.18
    in_edge_band = (
        center_x <= margin_x
        or center_x >= width - margin_x
        or center_y <= margin_y
        or center_y >= height - margin_y
    )
    in_lower_half = center_y >= height * 0.55

    return in_edge_band or in_lower_half


def detect_colored_watermark_region(roi: np.ndarray, min_area: int, max_area_ratio: float = 0.45) -> np.ndarray:
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    candidate = cv2.inRange(saturation, 70, 255)
    candidate = cv2.bitwise_and(candidate, cv2.inRange(value, 80, 255))

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, kernel, iterations=2)

    num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(candidate, connectivity=8)
    mask = np.zeros(candidate.shape, dtype=np.uint8)
    roi_area = roi.shape[0] * roi.shape[1]
    max_area = roi_area * max_area_ratio

    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if min_area <= area <= max_area:
            mask[labels == label] = 255

    return mask


def _high_contrast_text_candidates(image: np.ndarray) -> np.ndarray:
    bright = _local_bright_text_candidates(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    dark = cv2.bitwise_and(cv2.inRange(gray, 0, 135), cv2.inRange(value, 0, 185))
    dark = cv2.bitwise_and(dark, cv2.inRange(saturation, 0, 165))
    return cv2.bitwise_or(bright, dark)


def _local_text_stroke_candidates(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
    top_hat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    black_hat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    bright = cv2.bitwise_and(cv2.inRange(top_hat, 18, 255), cv2.inRange(saturation, 0, 190))
    dark = cv2.bitwise_and(cv2.inRange(black_hat, 18, 255), cv2.inRange(saturation, 0, 190))

    gradient_x = cv2.convertScaleAbs(cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3))
    gradient_y = cv2.convertScaleAbs(cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=3))
    gradient = cv2.addWeighted(gradient_x, 0.5, gradient_y, 0.5, 0)
    edges = cv2.inRange(gradient, 28, 255)

    candidate = cv2.bitwise_or(bright, dark)
    candidate = cv2.bitwise_and(candidate, cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1))
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)), iterations=1)
    return candidate


def _local_bright_text_candidates(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    top_hat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    bright = cv2.bitwise_and(cv2.inRange(top_hat, 16, 255), cv2.inRange(saturation, 0, 185))
    return bright


def _bright_low_saturation_candidates(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    bright = cv2.bitwise_and(cv2.inRange(gray, 150, 255), cv2.inRange(value, 160, 255))
    return cv2.bitwise_and(bright, cv2.inRange(saturation, 0, 170))


def merge_masks(*masks: np.ndarray) -> np.ndarray:
    merged = np.zeros(masks[0].shape, dtype=np.uint8)
    for mask in masks:
        merged = cv2.bitwise_or(merged, mask)
    return merged
