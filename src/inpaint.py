from __future__ import annotations

import base64
import json
import sys
import time
import uuid
from pathlib import Path
from typing import Any
from urllib import parse, request

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


class WebUIInpaintInpainter:
    def __init__(
        self,
        url: str,
        prompt: str,
        negative_prompt: str,
        steps: int,
        denoising_strength: float,
        cfg_scale: float,
        sampler_name: str,
        mask_blur: int,
        timeout: int,
    ) -> None:
        self._url = url.rstrip("/")
        self._prompt = prompt
        self._negative_prompt = negative_prompt
        self._steps = max(int(steps), 1)
        self._denoising_strength = min(max(float(denoising_strength), 0.0), 1.0)
        self._cfg_scale = max(float(cfg_scale), 1.0)
        self._sampler_name = sampler_name
        self._mask_blur = max(int(mask_blur), 0)
        self._timeout = max(int(timeout), 1)

    def inpaint(self, image: np.ndarray, mask: np.ndarray) -> InpaintResult:
        if not np.any(mask > 0):
            return InpaintResult(image=image.copy(), route="webui-api:empty-mask")

        payload = {
            "init_images": [_encode_image(image, ".png")],
            "mask": _encode_image(mask, ".png"),
            "prompt": self._prompt,
            "negative_prompt": self._negative_prompt,
            "steps": self._steps,
            "denoising_strength": self._denoising_strength,
            "cfg_scale": self._cfg_scale,
            "sampler_name": self._sampler_name,
            "mask_blur": self._mask_blur,
            "inpainting_fill": 1,
            "inpaint_full_res": True,
            "inpaint_full_res_padding": 32,
            "resize_mode": 0,
            "width": int(image.shape[1]),
            "height": int(image.shape[0]),
            "batch_size": 1,
            "n_iter": 1,
        }
        response = _post_json(f"{self._url}/sdapi/v1/img2img", payload=payload, timeout=self._timeout)
        images = response.get("images") if isinstance(response, dict) else None
        if not images:
            raise RuntimeError("WebUI img2img response did not contain images.")

        cleaned = _decode_image(images[0])
        if cleaned.shape[:2] != image.shape[:2]:
            cleaned = cv2.resize(cleaned, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
        return InpaintResult(
            image=cleaned,
            route=f"webui-api:{self._url}:steps{self._steps}:denoise{self._denoising_strength:g}",
        )


class ComfyUIApiInpainter:
    def __init__(
        self,
        url: str,
        workflow_path: Path,
        prompt: str | None,
        negative_prompt: str | None,
        steps: int | None,
        denoise: float | None,
        cfg: float | None,
        sampler: str | None,
        scheduler: str | None,
        timeout: int,
        poll_interval: float,
    ) -> None:
        self._url = url.rstrip("/")
        self._workflow_path = workflow_path
        self._prompt = prompt
        self._negative_prompt = negative_prompt
        self._steps = steps
        self._denoise = denoise
        self._cfg = cfg
        self._sampler = sampler
        self._scheduler = scheduler
        self._timeout = max(int(timeout), 1)
        self._poll_interval = max(float(poll_interval), 0.25)

        if not self._workflow_path.exists():
            raise RuntimeError(f"ComfyUI workflow does not exist: {self._workflow_path}")

    def inpaint(self, image: np.ndarray, mask: np.ndarray) -> InpaintResult:
        if not np.any(mask > 0):
            return InpaintResult(image=image.copy(), route="comfyui-api:empty-mask")

        workflow = self._load_workflow()
        upload_name = f"image_clean_{uuid.uuid4().hex}.png"
        uploaded = self._upload_rgba(image=image, mask=mask, filename=upload_name)
        remote_name = str(uploaded.get("name") or upload_name)
        self._configure_workflow(workflow, image_name=remote_name)

        prompt_id = self._queue_prompt(workflow)
        history = self._wait_for_history(prompt_id)
        output_info = _find_comfyui_output_image(history)
        cleaned = self._download_image(output_info)
        if cleaned.shape[:2] != image.shape[:2]:
            cleaned = cv2.resize(cleaned, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
        cleaned = composite_inpaint_result(original=image, cleaned=cleaned, mask=mask)

        return InpaintResult(
            image=cleaned,
            route=f"comfyui-api:{self._url}:prompt{prompt_id[:8]}",
        )

    def _load_workflow(self) -> dict[str, Any]:
        try:
            with self._workflow_path.open("r", encoding="utf-8") as handle:
                workflow = json.load(handle)
        except Exception as exc:
            raise RuntimeError(f"Failed to read ComfyUI workflow: {self._workflow_path}. Reason: {exc}") from exc

        if not isinstance(workflow, dict):
            raise RuntimeError(f"ComfyUI workflow must be a JSON object: {self._workflow_path}")
        return workflow

    def _upload_rgba(self, image: np.ndarray, mask: np.ndarray, filename: str) -> dict[str, Any]:
        rgba = _compose_rgba_for_comfyui(image=image, mask=mask)
        ok, encoded = cv2.imencode(".png", rgba)
        if not ok:
            raise RuntimeError("Failed to encode RGBA image for ComfyUI API.")

        return _post_multipart(
            f"{self._url}/upload/image",
            fields={"type": "input", "overwrite": "true"},
            files={"image": (filename, encoded.tobytes(), "image/png")},
            timeout=self._timeout,
        )

    def _configure_workflow(self, workflow: dict[str, Any], image_name: str) -> None:
        load_image = _find_comfyui_node(workflow, class_type="LoadImage")
        if load_image is None:
            raise RuntimeError("ComfyUI workflow must contain a LoadImage node.")
        load_image.setdefault("inputs", {})["image"] = image_name

        positive = _find_comfyui_clip_node(workflow, prefer_positive=True)
        if positive is not None and self._prompt is not None:
            positive.setdefault("inputs", {})["text"] = self._prompt

        negative = _find_comfyui_clip_node(workflow, prefer_positive=False)
        if negative is not None and self._negative_prompt is not None:
            negative.setdefault("inputs", {})["text"] = self._negative_prompt

        sampler = _find_comfyui_node(workflow, class_type="KSampler")
        if sampler is not None:
            inputs = sampler.setdefault("inputs", {})
            if self._steps is not None:
                inputs["steps"] = max(int(self._steps), 1)
            if self._denoise is not None:
                inputs["denoise"] = min(max(float(self._denoise), 0.0), 1.0)
            if self._cfg is not None:
                inputs["cfg"] = max(float(self._cfg), 1.0)
            if self._sampler is not None:
                inputs["sampler_name"] = self._sampler
            if self._scheduler is not None:
                inputs["scheduler"] = self._scheduler

        save_image = _find_comfyui_node(workflow, class_type="SaveImage")
        if save_image is not None:
            save_image.setdefault("inputs", {})["filename_prefix"] = f"image_clean_inpaint_{uuid.uuid4().hex[:8]}"

    def _queue_prompt(self, workflow: dict[str, Any]) -> str:
        response = _post_json(f"{self._url}/prompt", payload={"prompt": workflow}, timeout=self._timeout)
        prompt_id = response.get("prompt_id") if isinstance(response, dict) else None
        if not prompt_id:
            raise RuntimeError(f"ComfyUI prompt response did not contain prompt_id: {response}")
        return str(prompt_id)

    def _wait_for_history(self, prompt_id: str) -> dict[str, Any]:
        deadline = time.monotonic() + self._timeout
        history_url = f"{self._url}/history/{parse.quote(prompt_id)}"
        while time.monotonic() < deadline:
            history = _get_json(history_url, timeout=min(30, self._timeout))
            if isinstance(history, dict) and prompt_id in history:
                item = history[prompt_id]
                status = item.get("status") if isinstance(item, dict) else None
                if isinstance(status, dict) and status.get("status_str") == "error":
                    raise RuntimeError(f"ComfyUI prompt failed: {status.get('messages')}")
                return item
            time.sleep(self._poll_interval)

        raise RuntimeError(f"Timed out waiting for ComfyUI prompt: {prompt_id}")

    def _download_image(self, output_info: dict[str, Any]) -> np.ndarray:
        params = {
            "filename": output_info["filename"],
            "subfolder": output_info.get("subfolder", ""),
            "type": output_info.get("type", "output"),
        }
        raw = _get_bytes(f"{self._url}/view?{parse.urlencode(params)}", timeout=self._timeout)
        array = np.frombuffer(raw, dtype=np.uint8)
        image = cv2.imdecode(array, cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError("Failed to decode image downloaded from ComfyUI.")
        return image


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


def _compose_rgba_for_comfyui(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    inpaint_mask = np.clip(mask, 0, 255).astype(np.uint8)
    if inpaint_mask.shape[:2] != image.shape[:2]:
        inpaint_mask = cv2.resize(inpaint_mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    alpha = 255 - inpaint_mask
    return np.dstack([rgb, alpha])


def _encode_image(image: np.ndarray, ext: str) -> str:
    ok, encoded = cv2.imencode(ext, image)
    if not ok:
        raise RuntimeError("Failed to encode image for WebUI API.")
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def _decode_image(payload: str) -> np.ndarray:
    data = payload.split(",", 1)[1] if "," in payload else payload
    raw = base64.b64decode(data)
    array = np.frombuffer(raw, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError("Failed to decode image from WebUI API response.")
    return image


def _find_comfyui_node(workflow: dict[str, Any], class_type: str) -> dict[str, Any] | None:
    for node in workflow.values():
        if isinstance(node, dict) and node.get("class_type") == class_type:
            return node
    return None


def _find_comfyui_clip_node(workflow: dict[str, Any], prefer_positive: bool) -> dict[str, Any] | None:
    clip_nodes = [
        node
        for node in workflow.values()
        if isinstance(node, dict) and node.get("class_type") == "CLIPTextEncode"
    ]
    if not clip_nodes:
        return None
    if len(clip_nodes) == 1:
        return clip_nodes[0]

    negative_markers = ("watermark", "text", "logo", "caption", "subtitle", "signature")
    for node in clip_nodes:
        text = str(node.get("inputs", {}).get("text", "")).lower()
        looks_negative = any(marker in text for marker in negative_markers)
        if looks_negative != prefer_positive:
            return node

    return clip_nodes[0 if prefer_positive else -1]


def _find_comfyui_output_image(history: dict[str, Any]) -> dict[str, Any]:
    outputs = history.get("outputs") if isinstance(history, dict) else None
    if not isinstance(outputs, dict):
        raise RuntimeError(f"ComfyUI history did not contain outputs: {history}")

    for output in outputs.values():
        images = output.get("images") if isinstance(output, dict) else None
        if isinstance(images, list) and images:
            image = images[0]
            if isinstance(image, dict) and "filename" in image:
                return image

    raise RuntimeError(f"ComfyUI history did not contain output images: {history}")


def _post_json(url: str, payload: dict[str, Any], timeout: int) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    http_request = request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(http_request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        raise RuntimeError(f"WebUI API request failed: {url}. Reason: {exc}") from exc


def _get_json(url: str, timeout: int) -> dict[str, Any]:
    raw = _get_bytes(url, timeout=timeout)
    try:
        return json.loads(raw.decode("utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Failed to parse JSON response: {url}. Reason: {exc}") from exc


def _get_bytes(url: str, timeout: int) -> bytes:
    try:
        with request.urlopen(url, timeout=timeout) as response:
            return response.read()
    except Exception as exc:
        raise RuntimeError(f"HTTP GET request failed: {url}. Reason: {exc}") from exc


def _post_multipart(
    url: str,
    fields: dict[str, str],
    files: dict[str, tuple[str, bytes, str]],
    timeout: int,
) -> dict[str, Any]:
    boundary = f"----image-clean-{uuid.uuid4().hex}"
    body = bytearray()
    for name, value in fields.items():
        body.extend(f"--{boundary}\r\n".encode("ascii"))
        body.extend(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("ascii"))
        body.extend(str(value).encode("utf-8"))
        body.extend(b"\r\n")

    for name, (filename, content, content_type) in files.items():
        body.extend(f"--{boundary}\r\n".encode("ascii"))
        body.extend(
            (
                f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'
                f"Content-Type: {content_type}\r\n\r\n"
            ).encode("ascii")
        )
        body.extend(content)
        body.extend(b"\r\n")

    body.extend(f"--{boundary}--\r\n".encode("ascii"))
    http_request = request.Request(
        url,
        data=bytes(body),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    try:
        with request.urlopen(http_request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Multipart request failed: {url}. Reason: {exc}") from exc
