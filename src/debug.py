from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .io_utils import write_image


def write_debug_panel(
    path: Path,
    original: np.ndarray,
    mask: np.ndarray,
    result: np.ndarray,
    result_label: str | None = None,
) -> None:
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    overlay = original.copy()
    overlay[mask > 0] = (0, 0, 255)
    overlay = cv2.addWeighted(original, 0.65, overlay, 0.35, 0)
    result_panel = result.copy()
    if result_label:
        result_panel = add_panel_label(result_panel, result_label)
    panel = np.concatenate([original, overlay, result_panel, mask_bgr], axis=1)
    write_image(path, panel)


def write_mask_review_panel(
    path: Path,
    original: np.ndarray,
    mask: np.ndarray,
    label: str | None = None,
    color: tuple[int, int, int] = (0, 0, 255),
) -> None:
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    overlay = original.copy()
    overlay[mask > 0] = color
    overlay = cv2.addWeighted(original, 0.65, overlay, 0.35, 0)

    original_panel = original.copy()
    overlay_panel = overlay
    mask_panel = mask_bgr
    if label:
        overlay_panel = add_panel_label(overlay_panel, label)

    panel = np.concatenate([original_panel, overlay_panel, mask_panel], axis=1)
    write_image(path, panel)


def add_panel_label(image: np.ndarray, label: str) -> np.ndarray:
    labeled = image.copy()
    height, width = labeled.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(min(width, height) / 900, 0.45)
    thickness = max(int(round(font_scale * 2)), 1)
    padding = max(int(round(font_scale * 12)), 8)
    (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
    box_w = min(text_w + padding * 2, width)
    box_h = text_h + baseline + padding * 2
    cv2.rectangle(labeled, (0, 0), (box_w, box_h), (0, 0, 180), thickness=-1)
    cv2.putText(
        labeled,
        label,
        (padding, padding + text_h),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )
    return labeled
