from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .models import InpaintResult


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


def inpaint_flag(method: str) -> int:
    return cv2.INPAINT_NS if method == "ns" else cv2.INPAINT_TELEA


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

