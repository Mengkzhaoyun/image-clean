from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


@dataclass(frozen=True)
class TextDetection:
    box: list[tuple[int, int]]
    text: str
    score: float


@dataclass(frozen=True)
class ProcessResult:
    path: Path
    status: str
    output: Path | None = None
    mask: Path | None = None
    text_count: int = 0
    watermark_count: int = 0
    route: str | None = None
    message: str | None = None


@dataclass(frozen=True)
class InpaintResult:
    image: np.ndarray
    route: str


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

    def _detect_resized(self, image_path: Path, image: np.ndarray, scale: float) -> list[TextDetection]:
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


class BodyOverlapDetector:
    """Placeholder for the future person segmentation route."""

    def overlaps(self, _image: np.ndarray, _mask: np.ndarray) -> bool:
        return False


class OpenCVInpainter:
    def __init__(self, radius: float, method: str) -> None:
        self._radius = max(float(radius), 0.1)
        self._method = method

    def inpaint(self, image: np.ndarray, mask: np.ndarray) -> InpaintResult:
        cleaned = cv2.inpaint(
            image,
            mask,
            inpaintRadius=self._radius,
            flags=inpaint_flag(self._method),
        )
        return InpaintResult(image=cleaned, route=f"opencv-inpaint:{self._method}")


class LamaOnnxInpainter:
    def __init__(self, model_path: Path, device: str) -> None:
        try:
            import onnxruntime as ort
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "onnxruntime-directml is not installed. Install dependencies with "
                "`python -m pip install -r requirements.txt`."
            ) from exc

        if not model_path.exists():
            raise RuntimeError(f"LaMa ONNX model does not exist: {model_path}")

        self._ort = ort
        self._model_path = model_path
        self._directml_failed = False

        providers = self._ort.get_available_providers()
        provider_order: list[str] = []
        if device == "directml" and "DmlExecutionProvider" in providers:
            provider_order.append("DmlExecutionProvider")
        if "CPUExecutionProvider" in providers:
            provider_order.append("CPUExecutionProvider")
        if not provider_order:
            provider_order = providers

        self._providers = provider_order
        self._session = self._create_session(provider_order)
        self._input_size = self._detect_input_size()

    def inpaint(self, image: np.ndarray, mask: np.ndarray) -> InpaintResult:
        original_h, original_w = image.shape[:2]
        model_h, model_w = self._input_size

        resized_image = cv2.resize(image, (model_w, model_h), interpolation=cv2.INTER_AREA)
        resized_mask = cv2.resize(mask, (model_w, model_h), interpolation=cv2.INTER_NEAREST)

        rgb = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        image_tensor = np.transpose(rgb, (2, 0, 1))[None, ...]
        mask_tensor = (resized_mask.astype(np.float32) / 255.0)[None, None, ...]

        output = self._run_with_fallback(image_tensor=image_tensor, mask_tensor=mask_tensor)
        cleaned = normalize_lama_output(output)
        cleaned = cv2.resize(cleaned, (original_w, original_h), interpolation=cv2.INTER_LINEAR)
        merged = composite_inpaint_result(original=image, cleaned=cleaned, mask=mask)

        provider = self._session.get_providers()[0] if self._session.get_providers() else "unknown"
        return InpaintResult(image=merged, route=f"lama-onnx:{provider}:{model_w}x{model_h}")

    def _create_session(self, providers: list[str]) -> Any:
        options = self._ort.SessionOptions()
        options.log_severity_level = 3
        return self._ort.InferenceSession(str(self._model_path), sess_options=options, providers=providers)

    def _run_with_fallback(self, image_tensor: np.ndarray, mask_tensor: np.ndarray) -> np.ndarray:
        feed = self._build_feed(image_tensor=image_tensor, mask_tensor=mask_tensor)
        try:
            return self._session.run(None, feed)[0]
        except Exception as exc:
            if "DmlExecutionProvider" not in self._session.get_providers() or self._directml_failed:
                raise

            self._directml_failed = True
            print(
                "LaMa DirectML inference failed; retrying with CPUExecutionProvider. "
                f"Reason: {exc}",
                file=sys.stderr,
            )
            self._session = self._create_session(["CPUExecutionProvider"])
            feed = self._build_feed(image_tensor=image_tensor, mask_tensor=mask_tensor)
            return self._session.run(None, feed)[0]

    def _detect_input_size(self) -> tuple[int, int]:
        inputs = self._session.get_inputs()
        image_input = next(
            (
                input_info
                for input_info in inputs
                if "mask" not in input_info.name.lower() and len(input_info.shape) == 4
            ),
            inputs[0],
        )

        shape = image_input.shape
        if len(shape) != 4:
            raise RuntimeError(f"LaMa image input must be NCHW or NHWC, got shape: {shape}")

        if shape[1] == 3:
            height, width = shape[2], shape[3]
        else:
            height, width = shape[1], shape[2]

        if not isinstance(height, int) or not isinstance(width, int):
            raise RuntimeError(
                "Dynamic LaMa ONNX input size is not supported yet. "
                f"Model input shape: {shape}"
            )

        return height, width

    def _build_feed(self, image_tensor: np.ndarray, mask_tensor: np.ndarray) -> dict[str, np.ndarray]:
        inputs = self._session.get_inputs()
        if len(inputs) < 2:
            raise RuntimeError("LaMa ONNX model must expose at least image and mask inputs.")

        feed: dict[str, np.ndarray] = {}
        for input_info in inputs:
            name = input_info.name.lower()
            if "mask" in name:
                feed[input_info.name] = mask_tensor
            elif "image" in name or "img" in name:
                feed[input_info.name] = image_tensor

        if len(feed) < 2:
            feed = {
                inputs[0].name: image_tensor,
                inputs[1].name: mask_tensor,
            }

        return feed


def normalize_paddle_result(raw_result: Any) -> list[TextDetection]:
    detections: list[TextDetection] = []

    if isinstance(raw_result, list) and raw_result and isinstance(raw_result[0], dict):
        for page in raw_result:
            detections.extend(normalize_paddle3_page(page))
        return detections

    # PaddleOCR 2.x usually returns: [[ [box, (text, score)], ... ]]
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


def iter_images(input_dir: Path, recursive: bool) -> Iterable[Path]:
    pattern = "**/*" if recursive else "*"
    for path in input_dir.glob(pattern):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def read_image(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to read image: {path}")
    return image


def write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix or ".png"
    ok, encoded = cv2.imencode(ext, image)
    if not ok:
        raise ValueError(f"Failed to encode image: {path}")
    encoded.tofile(str(path))


def pad_to_multiple(array: np.ndarray, multiple: int) -> np.ndarray:
    height, width = array.shape[:2]
    pad_h = (multiple - height % multiple) % multiple
    pad_w = (multiple - width % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return array

    return cv2.copyMakeBorder(
        array,
        top=0,
        bottom=pad_h,
        left=0,
        right=pad_w,
        borderType=cv2.BORDER_REFLECT_101,
    )


def normalize_lama_output(output: np.ndarray) -> np.ndarray:
    data = output
    if data.ndim == 4:
        data = data[0]
    if data.ndim == 3 and data.shape[0] in {1, 3}:
        data = np.transpose(data, (1, 2, 0))
    if data.ndim == 2:
        data = data[:, :, None]

    if data.dtype != np.uint8:
        if float(np.nanmax(data)) <= 1.5:
            data = data * 255.0
        data = np.clip(data, 0, 255).astype(np.uint8)

    if data.shape[2] == 1:
        data = cv2.cvtColor(data, cv2.COLOR_GRAY2BGR)
    else:
        data = cv2.cvtColor(data[:, :, :3], cv2.COLOR_RGB2BGR)

    return data


def composite_inpaint_result(original: np.ndarray, cleaned: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if not np.any(mask > 0):
        return original.copy()

    alpha = (mask > 0).astype(np.float32)
    alpha = cv2.GaussianBlur(alpha, (0, 0), sigmaX=1.2, sigmaY=1.2)
    alpha = np.clip(alpha[:, :, None], 0.0, 1.0)

    merged = original.astype(np.float32) * (1.0 - alpha) + cleaned.astype(np.float32) * alpha
    return np.clip(merged, 0, 255).astype(np.uint8)


def debug_output_path(output_dir: Path, image_path: Path, input_dir: Path) -> Path:
    relative = image_path.relative_to(input_dir)
    return output_dir / relative.parent / f"{relative.stem}.debug.jpg"


def write_debug_panel(path: Path, original: np.ndarray, mask: np.ndarray, cleaned: np.ndarray) -> None:
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    overlay = original.copy()
    overlay[mask > 0] = (0, 0, 255)
    overlay = cv2.addWeighted(original, 0.65, overlay, 0.35, 0)
    panel = np.concatenate([original, overlay, cleaned, mask_bgr], axis=1)
    write_image(path, panel)


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

    # Logos and stickers are often bright, saturated marks on a comparatively plain background.
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


def inpaint_flag(method: str) -> int:
    return cv2.INPAINT_NS if method == "ns" else cv2.INPAINT_TELEA


def relative_output_path(input_dir: Path, output_dir: Path, image_path: Path) -> Path:
    return output_dir / image_path.relative_to(input_dir)


def mask_output_path(output_dir: Path, image_path: Path, input_dir: Path) -> Path:
    relative = image_path.relative_to(input_dir)
    return output_dir / relative.parent / f"{relative.stem}.mask.png"


def process_image(
    image_path: Path,
    input_dir: Path,
    output_dir: Path,
    detector: PaddleTextDetector,
    body_detector: BodyOverlapDetector,
    inpainter: OpenCVInpainter | LamaOnnxInpainter,
    dilate: int,
    mode: str,
    save_mask: bool,
    save_debug: bool,
    watermark_corners: bool,
    sticker_watermarks: bool,
    watermark_corner_ratio: float,
    watermark_min_area: int,
) -> ProcessResult:
    image = read_image(image_path)
    detections = detector.detect(image_path)

    text_mask = build_text_mask(image.shape, detections, dilate=dilate)
    watermark_mask, watermark_count = build_corner_watermark_mask(
        image=image,
        text_mask=text_mask,
        enabled=watermark_corners,
        corner_ratio=watermark_corner_ratio,
        min_area=watermark_min_area,
        dilate=dilate,
        text_near_radius=max(24, dilate * 4),
    )
    sticker_mask, sticker_count = build_colored_sticker_mask(
        image=image,
        text_mask=text_mask,
        enabled=sticker_watermarks,
        min_area=watermark_min_area,
        dilate=dilate,
        text_near_radius=max(28, dilate * 4),
    )
    watermark_count += sticker_count
    mask = merge_masks(text_mask, watermark_mask, sticker_mask)
    if not np.any(mask):
        return ProcessResult(path=image_path, status="skipped", text_count=0, message="no text or watermark")

    mask_path = mask_output_path(output_dir, image_path, input_dir) if save_mask or mode == "mask" else None
    if mask_path is not None:
        write_image(mask_path, mask)

    covers_body = body_detector.overlaps(image, mask)
    if covers_body:
        return ProcessResult(
            path=image_path,
            status="needs_aigc",
            mask=mask_path,
            text_count=len(detections),
            watermark_count=watermark_count,
            route="aigc",
            message="text overlaps body; AIGC route is not implemented yet",
        )

    output_path = relative_output_path(input_dir, output_dir, image_path)
    if mode == "mask":
        return ProcessResult(
            path=image_path,
            status="masked",
            mask=mask_path,
            text_count=len(detections),
            watermark_count=watermark_count,
            route="mask",
        )

    inpainted = inpainter.inpaint(image, mask)
    write_image(output_path, inpainted.image)
    if save_debug:
        write_debug_panel(debug_output_path(output_dir, image_path, input_dir), image, mask, inpainted.image)

    return ProcessResult(
        path=image_path,
        status="cleaned",
        output=output_path,
        mask=mask_path,
        text_count=len(detections),
        watermark_count=watermark_count,
        route=inpainted.route,
    )


def write_log(log_path: Path | None, result: ProcessResult) -> None:
    if log_path is None:
        return

    log_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "path": str(result.path),
        "status": result.status,
        "output": str(result.output) if result.output else None,
        "mask": str(result.mask) if result.mask else None,
        "text_count": result.text_count,
        "watermark_count": result.watermark_count,
        "route": result.route,
        "message": result.message,
    }
    with log_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch clean text from images.")
    parser.add_argument("--input", required=True, type=Path, help="Input image directory.")
    parser.add_argument("--output", required=True, type=Path, help="Output directory.")
    parser.add_argument("--recursive", action="store_true", help="Scan subdirectories recursively.")
    parser.add_argument("--device", default="directml", choices=["directml", "cpu"], help="Target device.")
    parser.add_argument("--dilate", default=8, type=int, help="Mask dilation pixels.")
    parser.add_argument("--mode", default="auto", choices=["auto", "inpaint", "mask"], help="Processing mode.")
    parser.add_argument("--inpaint-backend", default="opencv", choices=["opencv", "lama-onnx"], help="Inpaint backend.")
    parser.add_argument("--inpaint-radius", default=2.0, type=float, help="OpenCV inpaint radius.")
    parser.add_argument("--inpaint-method", default="telea", choices=["telea", "ns"], help="OpenCV inpaint method.")
    parser.add_argument("--lama-model", type=Path, help="Path to LaMa ONNX model.")
    parser.add_argument("--lang", default="ch", help="PaddleOCR language, for example ch or en.")
    parser.add_argument("--ocr-version", default="PP-OCRv4", help="PaddleOCR model version.")
    parser.add_argument("--ocr-upscale-small", default=640, type=int, help="Upscale images whose short side is smaller than this before OCR.")
    parser.add_argument("--ocr-min-score", default=0.55, type=float, help="Minimum OCR confidence score.")
    parser.add_argument("--no-angle-cls", action="store_true", help="Disable PaddleOCR angle classification.")
    parser.add_argument("--no-watermark-corners", action="store_true", help="Disable corner watermark detection.")
    parser.add_argument("--sticker-watermarks", action="store_true", help="Enable experimental full-image sticker detection.")
    parser.add_argument("--watermark-corner-ratio", default=0.18, type=float, help="Corner scan size ratio.")
    parser.add_argument("--watermark-min-area", default=40, type=int, help="Minimum colored watermark area.")
    parser.add_argument("--save-mask", action="store_true", help="Save generated masks next to cleaned outputs.")
    parser.add_argument("--save-debug", action="store_true", help="Save original/mask/result comparison panels.")
    parser.add_argument("--dry-run", action="store_true", help="Only list matched images; do not load PaddleOCR.")
    parser.add_argument("--log", type=Path, help="Optional JSONL processing log path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = args.input.resolve()
    output_dir = args.output.resolve()

    if not input_dir.exists() or not input_dir.is_dir():
        print(f"Input directory does not exist: {input_dir}", file=sys.stderr)
        return 2

    images = sorted(iter_images(input_dir, recursive=args.recursive))
    if args.dry_run:
        print(f"Found {len(images)} image(s).")
        for image_path in images:
            print(image_path)
        return 0

    try:
        if args.inpaint_backend == "lama-onnx":
            if args.lama_model is None:
                raise RuntimeError("`--lama-model` is required when `--inpaint-backend lama-onnx` is used.")
            inpainter: OpenCVInpainter | LamaOnnxInpainter = LamaOnnxInpainter(
                model_path=args.lama_model.resolve(),
                device=args.device,
            )
        else:
            inpainter = OpenCVInpainter(radius=args.inpaint_radius, method=args.inpaint_method)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if args.device == "directml":
        print("Device target: DirectML. PaddleOCR may still use its own available backend.")
    else:
        print("Device target: CPU.")

    try:
        detector = PaddleTextDetector(
            lang=args.lang,
            use_angle_cls=not args.no_angle_cls,
            ocr_version=args.ocr_version,
            upscale_small=args.ocr_upscale_small,
            min_score=args.ocr_min_score,
        )
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    body_detector = BodyOverlapDetector()
    stats: dict[str, int] = {}

    for image_path in images:
        try:
            result = process_image(
                image_path=image_path,
                input_dir=input_dir,
                output_dir=output_dir,
                detector=detector,
                body_detector=body_detector,
                inpainter=inpainter,
                dilate=max(args.dilate, 0),
                mode=args.mode,
                save_mask=args.save_mask,
                save_debug=args.save_debug,
                watermark_corners=not args.no_watermark_corners,
                sticker_watermarks=args.sticker_watermarks,
                watermark_corner_ratio=min(max(args.watermark_corner_ratio, 0.05), 0.35),
                watermark_min_area=max(args.watermark_min_area, 1),
            )
        except Exception as exc:
            result = ProcessResult(path=image_path, status="failed", message=str(exc))

        stats[result.status] = stats.get(result.status, 0) + 1
        write_log(args.log, result)
        detail = f", text={result.text_count}" if result.text_count else ""
        watermark = f", watermark={result.watermark_count}" if result.watermark_count else ""
        route = f", route={result.route}" if result.route else ""
        message = f", {result.message}" if result.message else ""
        print(f"[{result.status}] {image_path}{detail}{watermark}{route}{message}")

    print("Summary:", ", ".join(f"{key}={value}" for key, value in sorted(stats.items())) or "none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
