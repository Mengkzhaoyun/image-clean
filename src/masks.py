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


def merge_masks(*masks: np.ndarray) -> np.ndarray:
    merged = np.zeros(masks[0].shape, dtype=np.uint8)
    for mask in masks:
        merged = cv2.bitwise_or(merged, mask)
    return merged
