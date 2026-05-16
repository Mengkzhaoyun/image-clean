from __future__ import annotations

from pathlib import Path

import numpy as np

from .debug import write_debug_panel
from .io_utils import debug_output_path, mask_output_path, read_image, relative_output_path, write_image
from .inpaint import LamaOnnxInpainter, OpenCVInpainter
from .masks import build_colored_sticker_mask, build_corner_watermark_mask, build_text_mask, merge_masks
from .models import ProcessResult
from .ocr import PaddleTextDetector
from .protection import BodyOverlapDetector


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
    diagnostics = {
        "mask_pixels": int(np.count_nonzero(mask)),
        "text_mask_pixels": int(np.count_nonzero(text_mask)),
        "watermark_mask_pixels": int(np.count_nonzero(watermark_mask)),
        "sticker_mask_pixels": int(np.count_nonzero(sticker_mask)),
    }

    if not np.any(mask):
        if save_debug:
            write_debug_panel(debug_output_path(output_dir, image_path, input_dir), image, mask, image)
        return ProcessResult(
            path=image_path,
            status="skipped",
            text_count=len(detections),
            message="no text or watermark",
            diagnostics=diagnostics,
        )

    mask_path = mask_output_path(output_dir, image_path, input_dir) if save_mask or mode == "mask" else None
    if mask_path is not None:
        write_image(mask_path, mask)

    protection = body_detector.check(image, mask)
    diagnostics["face_overlap_pixels"] = protection.overlap_pixels
    diagnostics["face_protected_pixels"] = protection.protected_pixels
    if protection.overlaps:
        if save_debug:
            write_debug_panel(debug_output_path(output_dir, image_path, input_dir), image, mask, image)
        return ProcessResult(
            path=image_path,
            status="needs_aigc",
            mask=mask_path,
            text_count=len(detections),
            watermark_count=watermark_count,
            route="aigc",
            message="text overlaps body; AIGC route is not implemented yet",
            diagnostics=diagnostics,
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
            diagnostics=diagnostics,
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
        diagnostics=diagnostics,
    )
