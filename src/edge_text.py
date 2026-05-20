from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import numpy as np

from .io_utils import write_image
from .models import TextDetection
from .ocr import PaddleTextDetector


def detect_edge_text(
    image_path: Path,
    image: np.ndarray,
    detector: PaddleTextDetector,
    enabled: bool,
    edge_ratio: float,
    upscale: float,
    min_score: float,
) -> list[TextDetection]:
    if not enabled:
        return []

    height, width = image.shape[:2]
    band = max(24, int(min(height, width) * edge_ratio))
    band = min(band, max(height, width))
    crops = [
        ("top", 0, 0, width, min(band, height)),
        ("bottom", 0, max(0, height - band), width, min(band, height)),
        ("left", 0, 0, min(band, width), height),
        ("right", max(0, width - band), 0, min(band, width), height),
    ]

    detections: list[TextDetection] = []
    seen: set[tuple[int, int, int, int, str]] = set()
    for edge, x, y, w, h in crops:
        crop = image[y : y + h, x : x + w]
        if crop.size == 0:
            continue

        crop_detections = _detect_crop(
            source_path=image_path,
            crop=crop,
            edge=edge,
            detector=detector,
            upscale=max(upscale, 1.0),
        )
        for detection in crop_detections:
            if detection.score < min_score:
                continue

            mapped = _map_detection(detection, offset_x=x, offset_y=y, width=width, height=height)
            if not _touches_edge(mapped, width=width, height=height, margin=max(8, band // 6)):
                continue

            key = _dedupe_key(mapped)
            if key in seen:
                continue

            seen.add(key)
            detections.append(mapped)

    return detections


def _detect_crop(
    source_path: Path,
    crop: np.ndarray,
    edge: str,
    detector: PaddleTextDetector,
    upscale: float,
) -> list[TextDetection]:
    if upscale > 1:
        crop_for_ocr = cv2.resize(crop, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)
    else:
        crop_for_ocr = crop

    with tempfile.NamedTemporaryFile(prefix=f"{source_path.stem}.edge-{edge}-", suffix=".png", delete=False) as temp_file:
        temp_path = Path(temp_file.name)

    write_image(temp_path, crop_for_ocr)
    try:
        detections = detector.detect(temp_path)
    finally:
        temp_path.unlink(missing_ok=True)

    if upscale > 1:
        return [_scale_detection(detection, scale=upscale, shape=crop.shape) for detection in detections]
    return detections


def _scale_detection(detection: TextDetection, scale: float, shape: tuple[int, ...]) -> TextDetection:
    height, width = shape[:2]
    box: list[tuple[int, int]] = []
    for x, y in detection.box:
        box.append(
            (
                min(max(int(round(x / scale)), 0), width - 1),
                min(max(int(round(y / scale)), 0), height - 1),
            )
        )
    return TextDetection(box=box, text=detection.text, score=detection.score)


def _map_detection(
    detection: TextDetection,
    offset_x: int,
    offset_y: int,
    width: int,
    height: int,
) -> TextDetection:
    box: list[tuple[int, int]] = []
    for x, y in detection.box:
        box.append(
            (
                min(max(x + offset_x, 0), width - 1),
                min(max(y + offset_y, 0), height - 1),
            )
        )
    return TextDetection(box=box, text=detection.text, score=detection.score)


def _touches_edge(detection: TextDetection, width: int, height: int, margin: int) -> bool:
    xs = [point[0] for point in detection.box]
    ys = [point[1] for point in detection.box]
    return min(xs) <= margin or max(xs) >= width - margin or min(ys) <= margin or max(ys) >= height - margin


def _dedupe_key(detection: TextDetection) -> tuple[int, int, int, int, str]:
    xs = [point[0] for point in detection.box]
    ys = [point[1] for point in detection.box]
    return (
        round(min(xs) / 8) * 8,
        round(min(ys) / 8) * 8,
        round(max(xs) / 8) * 8,
        round(max(ys) / 8) * 8,
        detection.text.strip(),
    )
