from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from .models import TextDetection


class FlorenceTextDetector:
    def __init__(
        self,
        model_id: str,
        task: str,
        max_new_tokens: int,
        max_side: int,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoProcessor
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Florence vision detection needs optional dependencies. Install them with "
                "`python -m pip install torch transformers pillow`."
            ) from exc

        self._torch = torch
        self._task = task
        self._max_new_tokens = max(32, int(max_new_tokens))
        self._max_side = max(256, int(max_side))
        self._processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self._model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=torch.float32,
            attn_implementation="eager",
        ).eval()

    def detect(self, image_path: Path, image: np.ndarray) -> list[TextDetection]:
        original_h, original_w = image.shape[:2]
        image_for_model, scale = _resize_for_model(image, max_side=self._max_side)
        rgb = cv2.cvtColor(image_for_model, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)
        inputs = self._processor(text=self._task, images=pil_image, return_tensors="pt")

        with self._torch.no_grad():
            generated_ids = self._model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=self._max_new_tokens,
                num_beams=1,
                use_cache=False,
            )

        generated_text = self._processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed = self._processor.post_process_generation(
            generated_text,
            task=self._task,
            image_size=(pil_image.width, pil_image.height),
        )
        detections = normalize_florence_result(parsed, task=self._task, width=pil_image.width, height=pil_image.height)
        if scale != 1.0:
            detections = [_scale_detection(detection, scale=scale, width=original_w, height=original_h) for detection in detections]
        return detections


class NoopVisionDetector:
    def detect(self, image_path: Path, image: np.ndarray) -> list[TextDetection]:
        return []


def normalize_florence_result(parsed: Any, task: str, width: int, height: int) -> list[TextDetection]:
    payload = parsed.get(task, parsed) if isinstance(parsed, dict) else parsed
    detections: list[TextDetection] = []

    if isinstance(payload, dict):
        detections.extend(_from_quad_payload(payload, width=width, height=height))
        detections.extend(_from_bbox_payload(payload, width=width, height=height))

    return detections


def _from_quad_payload(payload: dict[str, Any], width: int, height: int) -> list[TextDetection]:
    quad_boxes = payload.get("quad_boxes") or payload.get("quad_bboxes") or []
    labels = payload.get("labels") or payload.get("texts") or payload.get("rec_texts") or []
    detections: list[TextDetection] = []

    for index, raw_box in enumerate(quad_boxes):
        if not isinstance(raw_box, (list, tuple)) or len(raw_box) < 8:
            continue
        points = []
        for point_index in range(0, 8, 2):
            points.append(
                (
                    _clamp_int(raw_box[point_index], 0, width - 1),
                    _clamp_int(raw_box[point_index + 1], 0, height - 1),
                )
            )
        text = str(labels[index]) if index < len(labels) else "vision_text"
        detections.append(TextDetection(box=points, text=text, score=1.0))

    return detections


def _from_bbox_payload(payload: dict[str, Any], width: int, height: int) -> list[TextDetection]:
    boxes = payload.get("bboxes") or payload.get("boxes") or []
    labels = payload.get("labels") or payload.get("texts") or []
    detections: list[TextDetection] = []

    for index, raw_box in enumerate(boxes):
        if not isinstance(raw_box, (list, tuple)) or len(raw_box) < 4:
            continue
        x1 = _clamp_int(raw_box[0], 0, width - 1)
        y1 = _clamp_int(raw_box[1], 0, height - 1)
        x2 = _clamp_int(raw_box[2], 0, width - 1)
        y2 = _clamp_int(raw_box[3], 0, height - 1)
        if x2 <= x1 or y2 <= y1:
            continue
        text = str(labels[index]) if index < len(labels) else "vision_text"
        detections.append(TextDetection(box=[(x1, y1), (x2, y1), (x2, y2), (x1, y2)], text=text, score=1.0))

    return detections


def _clamp_int(value: Any, low: int, high: int) -> int:
    try:
        number = int(round(float(value)))
    except (TypeError, ValueError):
        number = low
    return min(max(number, low), high)


def _resize_for_model(image: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    height, width = image.shape[:2]
    largest = max(height, width)
    if largest <= max_side:
        return image, 1.0

    scale = max_side / largest
    resized = cv2.resize(image, (int(round(width * scale)), int(round(height * scale))), interpolation=cv2.INTER_AREA)
    return resized, scale


def _scale_detection(detection: TextDetection, scale: float, width: int, height: int) -> TextDetection:
    box: list[tuple[int, int]] = []
    for x, y in detection.box:
        box.append(
            (
                min(max(int(round(x / scale)), 0), width - 1),
                min(max(int(round(y / scale)), 0), height - 1),
            )
        )
    return TextDetection(box=box, text=detection.text, score=detection.score)
