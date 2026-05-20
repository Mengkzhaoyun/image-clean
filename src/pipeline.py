from __future__ import annotations

from pathlib import Path
import tempfile

import cv2
import numpy as np

from .debug import write_debug_panel, write_mask_review_panel
from .edge_text import detect_edge_text
from .io_utils import (
    debug_output_path,
    mask_output_path,
    mask_review_output_path,
    read_image,
    relative_output_path,
    restricted_mask_output_path,
    safe_mask_output_path,
    write_image,
)
from .inpaint import ComfyUIApiInpainter, LamaOnnxInpainter, OpenCVInpainter, WebUIInpaintInpainter
from .layout_text import build_vertical_dark_text_mask, build_vertical_text_column_mask
from .mask_analysis import MaskSource, analyze_mask_components, split_mask_by_protection
from .masks import (
    build_bright_text_refinement_mask,
    build_colored_sticker_mask,
    build_corner_watermark_mask,
    build_dark_stroke_refinement_mask,
    build_empty_edge_dark_text_mask,
    build_edge_column_refinement_mask,
    build_post_dark_residual_mask,
    build_text_block_expansion_mask,
    build_text_mask,
    build_text_stroke_completion_mask,
    build_vertical_bright_text_mask,
    merge_masks,
)
from .models import ProcessResult
from .ocr import PaddleTextDetector
from .protection import BodyOverlapDetector
from .vision import NoopVisionDetector


def process_image(
    image_path: Path,
    input_dir: Path,
    output_dir: Path,
    detector: PaddleTextDetector,
    vision_detector: NoopVisionDetector,
    body_detector: BodyOverlapDetector,
    inpainter: OpenCVInpainter | LamaOnnxInpainter | WebUIInpaintInpainter | ComfyUIApiInpainter,
    dilate: int,
    mode: str,
    save_mask: bool,
    save_debug: bool,
    watermark_corners: bool,
    sticker_watermarks: bool,
    watermark_corner_ratio: float,
    watermark_min_area: int,
    edge_text: bool,
    edge_text_ratio: float,
    edge_text_upscale: float,
    edge_text_min_score: float,
    vision_trigger: str,
    vision_low_count: int,
    vision_max_area_ratio: float,
    vision_shrink_ratio: float,
    vision_edge_crops: bool,
    vision_edge_crop_trigger: str,
    vision_edge_crop_ratio: float,
    dark_stroke_refine: bool,
    dark_stroke_edge_ratio: float,
    dark_stroke_anchor_radius: int,
    dark_stroke_min_area: int,
    dark_stroke_max_area_ratio: float,
    edge_column_refine: bool,
    edge_column_edge_ratio: float,
    edge_column_anchor_radius: int,
    edge_column_min_area: int,
    edge_column_max_area_ratio: float,
    vertical_text: bool,
    vertical_text_min_area: int,
    vertical_text_max_area_ratio: float,
    vertical_text_edge_ratio: float,
    vertical_columns: bool,
    vertical_column_min_height_ratio: float,
    vertical_column_max_width_ratio: float,
    protected_action: str,
    post_ocr_check: bool,
) -> ProcessResult:
    image = read_image(image_path)
    detections = detector.detect(image_path)
    edge_detections = detect_edge_text(
        image_path=image_path,
        image=image,
        detector=detector,
        enabled=edge_text,
        edge_ratio=edge_text_ratio,
        upscale=edge_text_upscale,
        min_score=edge_text_min_score,
    )
    pre_vision_count = len(detections) + len(edge_detections)
    use_vision = should_run_vision(
        trigger=vision_trigger,
        pre_vision_count=pre_vision_count,
        low_count=vision_low_count,
    )
    raw_vision_detections = vision_detector.detect(image_path, image) if use_vision else []
    vision_detections = filter_vision_detections(
        detections=raw_vision_detections,
        shape=image.shape,
        max_area_ratio=vision_max_area_ratio,
        shrink_ratio=vision_shrink_ratio,
    )
    use_vision_edge_crops = should_run_vision(
        trigger=vision_edge_crop_trigger,
        pre_vision_count=pre_vision_count,
        low_count=vision_low_count,
    )
    raw_vision_edge_detections = detect_vision_edge_crops(
        image_path=image_path,
        image=image,
        vision_detector=vision_detector,
        enabled=vision_edge_crops and use_vision_edge_crops,
        crop_ratio=vision_edge_crop_ratio,
    )
    vision_edge_detections = filter_vision_detections(
        detections=raw_vision_edge_detections,
        shape=image.shape,
        max_area_ratio=vision_max_area_ratio,
        shrink_ratio=vision_shrink_ratio,
    )
    all_detections = detections + edge_detections + vision_detections + vision_edge_detections

    text_mask = build_text_mask(image.shape, all_detections, dilate=dilate)
    dark_stroke_mask, dark_stroke_count = build_dark_stroke_refinement_mask(
        image=image,
        anchor_mask=text_mask,
        enabled=dark_stroke_refine,
        edge_ratio=dark_stroke_edge_ratio,
        anchor_radius=dark_stroke_anchor_radius,
        min_area=dark_stroke_min_area,
        max_area_ratio=dark_stroke_max_area_ratio,
        dilate=max(1, dilate // 2),
    )
    vertical_mask, vertical_count = build_vertical_dark_text_mask(
        image=image,
        anchor_mask=text_mask,
        enabled=vertical_text,
        dilate=dilate,
        min_area=vertical_text_min_area,
        max_area_ratio=vertical_text_max_area_ratio,
        edge_ratio=vertical_text_edge_ratio,
        anchor_radius=max(24, dilate * 8),
    )
    vertical_column_mask, vertical_column_count = build_vertical_text_column_mask(
        image=image,
        enabled=vertical_columns,
        dilate=dilate,
        min_height_ratio=vertical_column_min_height_ratio,
        max_width_ratio=vertical_column_max_width_ratio,
        edge_ratio=vertical_text_edge_ratio,
    )
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
    pre_fallback_mask = merge_masks(text_mask, vertical_mask, vertical_column_mask, watermark_mask, sticker_mask)
    empty_edge_mask, empty_edge_count = build_empty_edge_dark_text_mask(
        image=image,
        enabled=edge_column_refine and not np.any(pre_fallback_mask),
        edge_ratio=edge_column_edge_ratio,
        min_area=edge_column_min_area,
        dilate=max(1, dilate // 2),
    )
    base_mask = merge_masks(pre_fallback_mask, empty_edge_mask)
    edge_column_mask, edge_column_count = build_edge_column_refinement_mask(
        image=image,
        anchor_mask=text_mask,
        enabled=edge_column_refine,
        edge_ratio=edge_column_edge_ratio,
        anchor_radius=edge_column_anchor_radius,
        min_area=edge_column_min_area,
        max_area_ratio=edge_column_max_area_ratio,
        dilate=max(1, dilate // 2),
    )
    bright_text_mask, bright_text_count = build_bright_text_refinement_mask(
        image=image,
        anchor_mask=text_mask,
        enabled=edge_column_refine,
        edge_ratio=edge_column_edge_ratio,
        anchor_radius=edge_column_anchor_radius,
        min_area=edge_column_min_area,
        max_area_ratio=edge_column_max_area_ratio,
        dilate=max(1, dilate // 2),
    )
    refined_mask = merge_masks(base_mask, edge_column_mask, bright_text_mask)
    mask = merge_masks(refined_mask, dark_stroke_mask)
    base_protection = body_detector.check(image, base_mask)
    candidate_protection = body_detector.check(image, mask)
    edge_column_discarded = False
    if edge_column_refine and np.any(edge_column_mask):
        edge_protection = body_detector.check(image, refined_mask)
        overlap_delta = edge_protection.overlap_pixels - base_protection.overlap_pixels
        protected_limit = max(256, int(base_protection.protected_pixels * 0.02))
        if overlap_delta > protected_limit:
            edge_column_mask = np.zeros_like(edge_column_mask)
            edge_column_count = 0
            bright_text_mask = np.zeros_like(bright_text_mask)
            bright_text_count = 0
            refined_mask = base_mask
            mask = merge_masks(refined_mask, dark_stroke_mask)
            candidate_protection = body_detector.check(image, mask)
            edge_column_discarded = True
    text_block_mask, text_block_count = build_text_block_expansion_mask(
        image=image,
        anchor_mask=text_mask,
        enabled=edge_column_refine,
        dilate=dilate,
        min_area=max(64, edge_column_min_area * 12),
        max_area_ratio=0.18,
    )
    vertical_bright_mask, vertical_bright_count = build_vertical_bright_text_mask(
        image=image,
        enabled=False,
        dilate=max(1, dilate // 2),
        min_components=3,
        min_span_ratio=0.06,
        max_width_ratio=0.04,
    )
    if np.any(vertical_bright_mask):
        mask = merge_masks(mask, vertical_bright_mask)
        refined_mask = merge_masks(refined_mask, vertical_bright_mask)
        candidate_protection = body_detector.check(image, mask)
    text_block_discarded = False
    if edge_column_refine and np.any(text_block_mask):
        mask = merge_masks(mask, text_block_mask)
        refined_mask = merge_masks(refined_mask, text_block_mask)
        candidate_protection = body_detector.check(image, mask)
    stroke_completion_mask, stroke_completion_count = build_text_stroke_completion_mask(
        image=image,
        anchor_mask=text_mask,
        enabled=edge_column_refine,
        anchor_radius=max(edge_column_anchor_radius, 44),
        min_area=max(2, edge_column_min_area),
        max_area_ratio=max(edge_column_max_area_ratio, 0.006),
        dilate=max(1, dilate // 2),
    )
    stroke_completion_discarded = False
    if edge_column_refine and np.any(stroke_completion_mask):
        stroke_mask = merge_masks(mask, stroke_completion_mask)
        stroke_protection = body_detector.check(image, stroke_mask)
        overlap_delta = stroke_protection.overlap_pixels - candidate_protection.overlap_pixels
        protected_limit = max(256, int(base_protection.protected_pixels * 0.02))
        if overlap_delta <= protected_limit:
            mask = stroke_mask
            refined_mask = merge_masks(refined_mask, stroke_completion_mask)
            candidate_protection = stroke_protection
        else:
            stroke_completion_mask = np.zeros_like(stroke_completion_mask)
            stroke_completion_count = 0
            stroke_completion_discarded = True
    dark_stroke_discarded = False
    if dark_stroke_refine and np.any(dark_stroke_mask):
        refined_protection = body_detector.check(image, refined_mask)
        overlap_delta = candidate_protection.overlap_pixels - refined_protection.overlap_pixels
        protected_limit = max(256, int(base_protection.protected_pixels * 0.02))
        if overlap_delta > protected_limit:
            mask = refined_mask
            dark_stroke_mask = np.zeros_like(dark_stroke_mask)
            dark_stroke_count = 0
            candidate_protection = refined_protection
            dark_stroke_discarded = True
    diagnostics = {
        "mask_pixels": int(np.count_nonzero(mask)),
        "text_mask_pixels": int(np.count_nonzero(text_mask)),
        "dark_stroke_mask_pixels": int(np.count_nonzero(dark_stroke_mask)),
        "dark_stroke_discarded": dark_stroke_discarded,
        "edge_column_mask_pixels": int(np.count_nonzero(edge_column_mask)),
        "edge_column_count": edge_column_count,
        "edge_column_discarded": edge_column_discarded,
        "bright_text_mask_pixels": int(np.count_nonzero(bright_text_mask)),
        "bright_text_count": bright_text_count,
        "text_block_mask_pixels": int(np.count_nonzero(text_block_mask)),
        "text_block_count": text_block_count,
        "text_block_discarded": text_block_discarded,
        "vertical_bright_mask_pixels": int(np.count_nonzero(vertical_bright_mask)),
        "vertical_bright_count": vertical_bright_count,
        "stroke_completion_mask_pixels": int(np.count_nonzero(stroke_completion_mask)),
        "stroke_completion_count": stroke_completion_count,
        "stroke_completion_discarded": stroke_completion_discarded,
        "empty_edge_mask_pixels": int(np.count_nonzero(empty_edge_mask)),
        "empty_edge_count": empty_edge_count,
        "vertical_text_mask_pixels": int(np.count_nonzero(vertical_mask)),
        "vertical_column_mask_pixels": int(np.count_nonzero(vertical_column_mask)),
        "watermark_mask_pixels": int(np.count_nonzero(watermark_mask)),
        "sticker_mask_pixels": int(np.count_nonzero(sticker_mask)),
        "ocr_text_count": len(detections),
        "edge_text_count": len(edge_detections),
        "vision_text_count": len(vision_detections),
        "vision_raw_text_count": len(raw_vision_detections),
        "vision_edge_text_count": len(vision_edge_detections),
        "vision_edge_raw_text_count": len(raw_vision_edge_detections),
        "vision_edge_triggered": vision_edge_crops and use_vision_edge_crops,
        "dark_stroke_count": dark_stroke_count,
        "vertical_text_count": vertical_count,
        "vertical_column_count": vertical_column_count,
        "vision_triggered": use_vision,
        "pre_vision_text_count": pre_vision_count,
    }
    protected_mask = body_detector.build_protection_mask(image)
    safe_mask, restricted_mask = split_mask_by_protection(mask, protected_mask)
    diagnostics["safe_mask_pixels"] = int(np.count_nonzero(safe_mask))
    diagnostics["restricted_mask_pixels"] = int(np.count_nonzero(restricted_mask))
    diagnostics["safe_mask_components"] = _count_mask_components(safe_mask)
    diagnostics["restricted_mask_components"] = _count_mask_components(restricted_mask)
    diagnostics.update(
        analyze_mask_components(
            mask=mask,
            sources=[
                MaskSource("text", text_mask),
                MaskSource("edge_column", edge_column_mask),
                MaskSource("bright_text", bright_text_mask),
                MaskSource("text_block", text_block_mask),
                MaskSource("vertical_bright", vertical_bright_mask),
                MaskSource("stroke_completion", stroke_completion_mask),
                MaskSource("dark_stroke", dark_stroke_mask),
                MaskSource("vertical_text", vertical_mask),
                MaskSource("vertical_column", vertical_column_mask),
                MaskSource("watermark", watermark_mask),
                MaskSource("sticker", sticker_mask),
                MaskSource("empty_edge", empty_edge_mask),
            ],
            protected_mask=protected_mask,
        )
    )

    if not np.any(mask):
        if save_debug:
            write_debug_panel(
                debug_output_path(output_dir, image_path, input_dir),
                image,
                mask,
                image,
                result_label="SKIPPED: no mask",
            )
        return ProcessResult(
            path=image_path,
            status="skipped",
            text_count=len(all_detections),
            message="no text or watermark",
            diagnostics=diagnostics,
        )

    mask_path = mask_output_path(output_dir, image_path, input_dir) if save_mask or mode == "mask" else None
    if mask_path is not None:
        write_image(mask_path, mask)
        write_image(safe_mask_output_path(output_dir, image_path, input_dir), safe_mask)
        write_image(restricted_mask_output_path(output_dir, image_path, input_dir), restricted_mask)
    if save_debug:
        _write_mask_review_panels(
            output_dir=output_dir,
            image_path=image_path,
            input_dir=input_dir,
            image=image,
            mask=mask,
            safe_mask=safe_mask,
            restricted_mask=restricted_mask,
        )

    protection = candidate_protection
    diagnostics["face_overlap_pixels"] = protection.overlap_pixels
    diagnostics["face_protected_pixels"] = protection.protected_pixels
    if protection.overlaps and protected_action == "route":
        if save_debug:
            write_debug_panel(
                debug_output_path(output_dir, image_path, input_dir),
                image,
                mask,
                image,
                result_label="NOT REPAIRED: routed",
            )
        return ProcessResult(
            path=image_path,
            status="needs_aigc",
            mask=mask_path,
            text_count=len(all_detections),
            watermark_count=watermark_count,
            route="aigc",
            message="protected area overlap; automatic advanced repair required",
            diagnostics=diagnostics,
        )

    output_path = relative_output_path(input_dir, output_dir, image_path)
    if mode == "mask":
        return ProcessResult(
            path=image_path,
            status="masked",
            mask=mask_path,
            text_count=len(all_detections),
            watermark_count=watermark_count,
            route="mask",
            diagnostics=diagnostics,
        )

    inpainted = inpainter.inpaint(image, mask)
    post_residual_mask, post_residual_count = build_post_dark_residual_mask(
        image=inpainted.image,
        anchor_mask=text_mask,
        enabled=edge_column_refine,
        edge_ratio=edge_column_edge_ratio,
        anchor_radius=edge_column_anchor_radius,
        min_area=edge_column_min_area,
        max_area_ratio=edge_column_max_area_ratio,
        dilate=max(1, dilate // 2),
    )
    diagnostics["post_dark_residual_count"] = post_residual_count
    diagnostics["post_dark_residual_mask_pixels"] = int(np.count_nonzero(post_residual_mask))
    if np.any(post_residual_mask):
        post_mask = merge_masks(mask, post_residual_mask)
        post_protection = body_detector.check(image, post_mask)
        overlap_delta = post_protection.overlap_pixels - protection.overlap_pixels
        protected_limit = max(256, int(protection.protected_pixels * 0.02))
        if overlap_delta <= protected_limit:
            mask = post_mask
            protection = post_protection
            safe_mask, restricted_mask = split_mask_by_protection(mask, protected_mask)
            diagnostics["mask_pixels"] = int(np.count_nonzero(mask))
            diagnostics["safe_mask_pixels"] = int(np.count_nonzero(safe_mask))
            diagnostics["restricted_mask_pixels"] = int(np.count_nonzero(restricted_mask))
            diagnostics["safe_mask_components"] = _count_mask_components(safe_mask)
            diagnostics["restricted_mask_components"] = _count_mask_components(restricted_mask)
            diagnostics["face_overlap_pixels"] = protection.overlap_pixels
            if mask_path is not None:
                write_image(mask_path, mask)
                write_image(safe_mask_output_path(output_dir, image_path, input_dir), safe_mask)
                write_image(restricted_mask_output_path(output_dir, image_path, input_dir), restricted_mask)
            if save_debug:
                _write_mask_review_panels(
                    output_dir=output_dir,
                    image_path=image_path,
                    input_dir=input_dir,
                    image=image,
                    mask=mask,
                    safe_mask=safe_mask,
                    restricted_mask=restricted_mask,
                )
            primary_route = inpainted.route
            cleaned_residual = _fill_small_residuals_from_context(inpainted.image, post_residual_mask)
            residual_inpainter = OpenCVInpainter(radius=8.0, method="telea")
            residual_result = residual_inpainter.inpaint(cleaned_residual, post_residual_mask)
            inpainted = type(inpainted)(
                image=residual_result.image,
                route=f"{primary_route}+post:context-fill+{residual_result.route}",
            )
            diagnostics["post_dark_residual_route"] = residual_result.route
            diagnostics["post_dark_residual_applied"] = True
        else:
            diagnostics["post_dark_residual_applied"] = False
            diagnostics["post_dark_residual_discarded"] = True
    else:
        diagnostics["post_dark_residual_applied"] = False
    residual_detections = _detect_residual_text(
        image=inpainted.image,
        detector=detector,
        enabled=post_ocr_check,
    )
    diagnostics["residual_text_count"] = len(residual_detections)
    write_image(output_path, inpainted.image)
    if save_debug:
        write_debug_panel(debug_output_path(output_dir, image_path, input_dir), image, mask, inpainted.image)

    status = "cleaned_protected" if protection.overlaps else "cleaned"
    message = "protected area overlap; repaired by selected backend" if protection.overlaps else None
    if residual_detections:
        status = "quality_failed"
        residual_message = f"post OCR detected {len(residual_detections)} residual text candidate(s)"
        message = f"{message}; {residual_message}" if message else residual_message

    return ProcessResult(
        path=image_path,
        status=status,
        output=output_path,
        mask=mask_path,
        text_count=len(all_detections),
        watermark_count=watermark_count,
        route=inpainted.route,
        message=message,
        diagnostics=diagnostics,
    )


def _fill_small_residuals_from_context(image: np.ndarray, residual_mask: np.ndarray) -> np.ndarray:
    if not np.any(residual_mask):
        return image

    result = image.copy()
    height, width = residual_mask.shape[:2]
    num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(residual_mask, connectivity=8)
    for label in range(1, num_labels):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area <= 0:
            continue

        pad = max(12, min(max(w, h), 32))
        x1 = max(x - pad, 0)
        y1 = max(y - pad, 0)
        x2 = min(x + w + pad, width)
        y2 = min(y + h + pad, height)
        component = (labels[y1:y2, x1:x2] == label)
        sample_mask = ~cv2.dilate(component.astype(np.uint8), np.ones((5, 5), dtype=np.uint8), iterations=1).astype(bool)
        sample_pixels = result[y1:y2, x1:x2][sample_mask]
        if sample_pixels.size == 0:
            continue

        fill_color = np.median(sample_pixels, axis=0).astype(np.uint8)
        roi = result[y1:y2, x1:x2].copy()
        roi[component] = fill_color
        smooth = cv2.GaussianBlur(roi, (0, 0), sigmaX=2.0, sigmaY=2.0)
        feather = cv2.GaussianBlur(component.astype(np.float32), (0, 0), sigmaX=2.5, sigmaY=2.5)
        feather = np.clip(feather[:, :, None], 0.0, 1.0)
        result[y1:y2, x1:x2] = (smooth.astype(np.float32) * feather + roi.astype(np.float32) * (1.0 - feather)).astype(np.uint8)

    return result


def _write_mask_review_panels(
    output_dir: Path,
    image_path: Path,
    input_dir: Path,
    image: np.ndarray,
    mask: np.ndarray,
    safe_mask: np.ndarray,
    restricted_mask: np.ndarray,
) -> None:
    write_mask_review_panel(
        mask_review_output_path(output_dir, image_path, input_dir, kind="all"),
        image,
        mask,
        label="ALL MASK",
        color=(0, 0, 255),
    )
    write_mask_review_panel(
        mask_review_output_path(output_dir, image_path, input_dir, kind="safe"),
        image,
        safe_mask,
        label="SAFE MASK",
        color=(0, 180, 0),
    )
    write_mask_review_panel(
        mask_review_output_path(output_dir, image_path, input_dir, kind="restricted"),
        image,
        restricted_mask,
        label="RESTRICTED MASK",
        color=(255, 0, 0),
    )


def _count_mask_components(mask: np.ndarray) -> int:
    if not np.any(mask):
        return 0
    num_labels, _labels, stats, _centroids = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
    count = 0
    for label in range(1, num_labels):
        if int(stats[label, cv2.CC_STAT_AREA]) >= 8:
            count += 1
    return count


def _detect_residual_text(
    image: np.ndarray,
    detector: PaddleTextDetector,
    enabled: bool,
) -> list:
    if not enabled:
        return []

    with tempfile.NamedTemporaryFile(prefix="image-clean-post-ocr-", suffix=".png", delete=False) as temp_file:
        temp_path = Path(temp_file.name)

    write_image(temp_path, image)
    try:
        return detector.detect(temp_path)
    finally:
        temp_path.unlink(missing_ok=True)


def should_run_vision(trigger: str, pre_vision_count: int, low_count: int) -> bool:
    if trigger == "always":
        return True
    if trigger == "empty":
        return pre_vision_count == 0
    if trigger == "low-count":
        return pre_vision_count <= max(low_count, 0)
    return False


def detect_vision_edge_crops(
    image_path: Path,
    image: np.ndarray,
    vision_detector: NoopVisionDetector,
    enabled: bool,
    crop_ratio: float,
) -> list:
    if not enabled:
        return []

    height, width = image.shape[:2]
    crop_width = int(round(width * crop_ratio))
    if crop_width <= 8 or crop_width >= width:
        return []

    crops = [
        (0, image[:, :crop_width]),
        (width - crop_width, image[:, width - crop_width :]),
    ]
    detections = []
    for x_offset, crop in crops:
        for detection in vision_detector.detect(image_path, crop):
            detections.append(_offset_detection(detection, x_offset=x_offset, y_offset=0, width=width, height=height))
    return detections


def filter_vision_detections(
    detections: list,
    shape: tuple[int, ...],
    max_area_ratio: float,
    shrink_ratio: float,
) -> list:
    height, width = shape[:2]
    image_area = height * width
    filtered = []
    for detection in detections:
        xs = [point[0] for point in detection.box]
        ys = [point[1] for point in detection.box]
        x1 = min(xs)
        x2 = max(xs)
        y1 = min(ys)
        y2 = max(ys)
        area = max(x2 - x1, 0) * max(y2 - y1, 0)
        if area <= 0 or area / image_area > max_area_ratio:
            continue

        if shrink_ratio > 0:
            pad_x = int(round((x2 - x1) * shrink_ratio))
            pad_y = int(round((y2 - y1) * shrink_ratio))
            x1 = min(max(x1 + pad_x, 0), width - 1)
            x2 = min(max(x2 - pad_x, x1 + 1), width - 1)
            y1 = min(max(y1 + pad_y, 0), height - 1)
            y2 = min(max(y2 - pad_y, y1 + 1), height - 1)
            filtered.append(type(detection)(box=[(x1, y1), (x2, y1), (x2, y2), (x1, y2)], text=detection.text, score=detection.score))
        else:
            filtered.append(detection)

    return filtered


def _offset_detection(detection, x_offset: int, y_offset: int, width: int, height: int):
    box = []
    for x, y in detection.box:
        box.append(
            (
                min(max(int(x) + x_offset, 0), width - 1),
                min(max(int(y) + y_offset, 0), height - 1),
            )
        )
    return type(detection)(box=box, text=detection.text, score=detection.score)
