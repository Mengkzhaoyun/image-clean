from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .io_utils import write_image


def write_debug_panel(path: Path, original: np.ndarray, mask: np.ndarray, result: np.ndarray) -> None:
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    overlay = original.copy()
    overlay[mask > 0] = (0, 0, 255)
    overlay = cv2.addWeighted(original, 0.65, overlay, 0.35, 0)
    panel = np.concatenate([original, overlay, result, mask_bgr], axis=1)
    write_image(path, panel)

