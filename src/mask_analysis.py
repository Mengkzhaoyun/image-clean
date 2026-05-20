from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class MaskSource:
    name: str
    mask: np.ndarray


def analyze_mask_components(
    mask: np.ndarray,
    sources: list[MaskSource],
    protected_mask: np.ndarray | None = None,
    min_component_area: int = 8,
) -> dict[str, object]:
    if not np.any(mask):
        return {
            "mask_component_count": 0,
            "mask_class_counts": {},
            "mask_class_pixels": {},
            "mask_zone_counts": {},
            "mask_collision_count": 0,
            "mask_collision_pixels": 0,
            "mask_large_component_count": 0,
            "mask_unknown_component_count": 0,
            "mask_top_classes": "",
        }

    height, width = mask.shape[:2]
    image_area = max(height * width, 1)
    protected = _normalize_mask(protected_mask, mask.shape) if protected_mask is not None else None
    normalized_sources = [MaskSource(source.name, _normalize_mask(source.mask, mask.shape)) for source in sources]

    num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
    class_counts: Counter[str] = Counter()
    class_pixels: defaultdict[str, int] = defaultdict(int)
    zone_counts: Counter[str] = Counter()
    collision_count = 0
    collision_pixels = 0
    large_count = 0
    unknown_count = 0
    component_count = 0

    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_component_area:
            continue

        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        component = labels[y : y + h, x : x + w] == label
        component_count += 1

        source_name = _classify_source(component, normalized_sources, x=x, y=y)
        class_counts[source_name] += 1
        class_pixels[source_name] += area
        if source_name == "unknown":
            unknown_count += 1

        zone = _classify_zone(x=x, y=y, w=w, h=h, width=width, height=height)
        zone_counts[zone] += 1
        if area >= image_area * 0.03:
            large_count += 1

        if protected is not None:
            protected_roi = protected[y : y + h, x : x + w] > 0
            overlap_pixels = int(np.count_nonzero(component & protected_roi))
            if overlap_pixels:
                collision_count += 1
                collision_pixels += overlap_pixels

    return {
        "mask_component_count": component_count,
        "mask_class_counts": dict(sorted(class_counts.items())),
        "mask_class_pixels": dict(sorted(class_pixels.items())),
        "mask_zone_counts": dict(sorted(zone_counts.items())),
        "mask_collision_count": collision_count,
        "mask_collision_pixels": collision_pixels,
        "mask_large_component_count": large_count,
        "mask_unknown_component_count": unknown_count,
        "mask_top_classes": _format_top_classes(class_pixels),
    }


def split_mask_by_protection(mask: np.ndarray, protected_mask: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    safe = np.zeros(mask.shape[:2], dtype=np.uint8)
    restricted = np.zeros(mask.shape[:2], dtype=np.uint8)
    if not np.any(mask):
        return safe, restricted

    protected = _normalize_mask(protected_mask, mask.shape) if protected_mask is not None else np.zeros(mask.shape[:2], dtype=np.uint8)
    num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
    for label in range(1, num_labels):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        component = labels[y : y + h, x : x + w] == label
        protected_roi = protected[y : y + h, x : x + w] > 0
        target = restricted if np.any(component & protected_roi) else safe
        target[y : y + h, x : x + w][component] = 255

    return safe, restricted


def _classify_source(component: np.ndarray, sources: list[MaskSource], x: int, y: int) -> str:
    best_name = "unknown"
    best_pixels = 0
    component_pixels = max(int(np.count_nonzero(component)), 1)
    mixed_pixels = 0

    for source in sources:
        roi = source.mask[y : y + component.shape[0], x : x + component.shape[1]] > 0
        overlap = int(np.count_nonzero(component & roi))
        if overlap:
            mixed_pixels += overlap
        if overlap > best_pixels:
            best_pixels = overlap
            best_name = source.name

    if best_pixels == 0:
        return "unknown"
    if mixed_pixels >= component_pixels * 1.25 and best_pixels < component_pixels * 0.72:
        return "mixed"
    return best_name


def _classify_zone(x: int, y: int, w: int, h: int, width: int, height: int) -> str:
    center_x = x + w / 2
    center_y = y + h / 2
    edge_x = width * 0.18
    edge_y = height * 0.18
    if center_y >= height * 0.62:
        return "lower"
    if center_x <= edge_x or center_x >= width - edge_x:
        return "side"
    if center_y <= edge_y:
        return "top"
    return "center"


def _format_top_classes(class_pixels: dict[str, int]) -> str:
    if not class_pixels:
        return ""
    ranked = sorted(class_pixels.items(), key=lambda item: item[1], reverse=True)
    return ", ".join(f"{name}:{pixels}" for name, pixels in ranked[:3])


def _normalize_mask(mask: np.ndarray | None, shape: tuple[int, ...]) -> np.ndarray:
    if mask is None:
        return np.zeros(shape[:2], dtype=np.uint8)
    normalized = mask
    if normalized.shape[:2] != shape[:2]:
        normalized = cv2.resize(normalized, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return (normalized > 0).astype(np.uint8) * 255
