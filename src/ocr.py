from __future__ import annotations

import inspect
import os
import tempfile
from pathlib import Path
from typing import Any

import cv2

from .io_utils import read_image, write_image
from .models import TextDetection


class PaddleTextDetector:
    def __init__(
        self,
        lang: str,
        use_angle_cls: bool,
        ocr_version: str,
        upscale_small: int,
        min_score: float,
    ) -> None:
        os.environ.setdefault("FLAGS_use_onednn", "0")
        try:
            from paddleocr import PaddleOCR
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "PaddleOCR is not installed. Install dependencies with "
                "`python -m pip install -r requirements.txt`."
            ) from exc

        init_params = inspect.signature(PaddleOCR).parameters
        kwargs: dict[str, Any] = {
            "lang": lang,
            "device": "cpu",
            "enable_mkldnn": False,
            "cpu_threads": 4,
        }
        if ocr_version:
            kwargs["ocr_version"] = ocr_version
        if "use_doc_orientation_classify" in init_params:
            kwargs["use_doc_orientation_classify"] = False
        if "use_doc_unwarping" in init_params:
            kwargs["use_doc_unwarping"] = False
        if "use_textline_orientation" in init_params:
            kwargs["use_textline_orientation"] = use_angle_cls
        elif "use_angle_cls" in init_params:
            kwargs["use_angle_cls"] = use_angle_cls

        self._use_angle_cls = use_angle_cls
        self._upscale_small = max(int(upscale_small), 0)
        self._min_score = float(min_score)
        self._ocr = PaddleOCR(**kwargs)

    def detect(self, image_path: Path) -> list[TextDetection]:
        image = read_image(image_path)
        short_side = min(image.shape[:2])
        if self._upscale_small and 0 < short_side < self._upscale_small:
            scale = self._upscale_small / short_side
            return self._filter_detections(self._detect_resized(image_path=image_path, image=image, scale=scale))

        return self._filter_detections(self._detect_path(image_path))

    def _detect_path(self, image_path: Path) -> list[TextDetection]:
        if hasattr(self._ocr, "predict"):
            raw_result = self._ocr.predict(
                str(image_path),
                use_textline_orientation=self._use_angle_cls,
            )
        else:
            raw_result = self._ocr.ocr(str(image_path), cls=self._use_angle_cls)
        return normalize_paddle_result(raw_result)

    def _detect_resized(self, image_path: Path, image, scale: float) -> list[TextDetection]:
        resized = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        with tempfile.NamedTemporaryFile(prefix="image-clean-ocr-", suffix=".png", delete=False) as temp_file:
            temp_path = Path(temp_file.name)
        write_image(temp_path, resized)
        try:
            detections = self._detect_path(temp_path)
        finally:
            temp_path.unlink(missing_ok=True)

        return [scale_detection(detection, scale=scale, shape=image.shape) for detection in detections]

    def _filter_detections(self, detections: list[TextDetection]) -> list[TextDetection]:
        return [detection for detection in detections if detection.score >= self._min_score]


def normalize_paddle_result(raw_result: Any) -> list[TextDetection]:
    detections: list[TextDetection] = []

    if isinstance(raw_result, list) and raw_result and isinstance(raw_result[0], dict):
        for page in raw_result:
            detections.extend(normalize_paddle3_page(page))
        return detections

    pages = raw_result or []
    if pages and isinstance(pages, list) and len(pages) == 1 and isinstance(pages[0], list):
        pages = pages[0]

    for item in pages:
        if not item or not isinstance(item, (list, tuple)) or len(item) < 2:
            continue

        box_raw = item[0]
        text_raw = item[1]
        text = ""
        score = 0.0

        if isinstance(text_raw, (list, tuple)) and text_raw:
            text = str(text_raw[0]) if len(text_raw) >= 1 else ""
            try:
                score = float(text_raw[1]) if len(text_raw) >= 2 else 0.0
            except (TypeError, ValueError):
                score = 0.0
        else:
            text = str(text_raw)

        try:
            box = [(int(round(float(x))), int(round(float(y)))) for x, y in box_raw]
        except (TypeError, ValueError):
            continue

        if len(box) >= 3:
            detections.append(TextDetection(box=box, text=text, score=score))

    return detections


def normalize_paddle3_page(page: dict[str, Any]) -> list[TextDetection]:
    detections: list[TextDetection] = []
    boxes = page.get("dt_polys") or page.get("rec_polys") or []
    texts = page.get("rec_texts") or []
    scores = page.get("rec_scores") or []

    for index, box_raw in enumerate(boxes):
        try:
            box = [(int(round(float(x))), int(round(float(y)))) for x, y in box_raw]
        except (TypeError, ValueError):
            continue

        if len(box) < 3:
            continue

        text = str(texts[index]) if index < len(texts) else ""
        try:
            score = float(scores[index]) if index < len(scores) else 0.0
        except (TypeError, ValueError):
            score = 0.0

        detections.append(TextDetection(box=box, text=text, score=score))

    return detections


def scale_detection(detection: TextDetection, scale: float, shape: tuple[int, ...]) -> TextDetection:
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

