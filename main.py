from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

from src.inpaint import ComfyUIApiInpainter, LamaOnnxInpainter, OpenCVInpainter, WebUIInpaintInpainter
from src.io_utils import iter_images
from src.log import write_log
from src.models import ProcessResult
from src.ocr import PaddleTextDetector
from src.pipeline import process_image
from src.protection import BodyOverlapDetector
from src.report import write_report
from src.vision import FlorenceTextDetector, NoopVisionDetector


def parse_args() -> argparse.Namespace:
    env = load_dotenv(Path(".env"))
    parser = argparse.ArgumentParser(description="Batch clean text from images.")
    parser.add_argument(
        "--preset",
        choices=["manual", "watermark-first"],
        default="manual",
        help="Pipeline preset. watermark-first focuses on complete text/watermark mask coverage before photo-quality repair.",
    )
    parser.add_argument("--input", required=True, type=Path, help="Input image directory.")
    parser.add_argument("--output", required=True, type=Path, help="Output directory.")
    parser.add_argument("--auto-run-dir", action="store_true", help="Create a timestamped child directory under --output.")
    parser.add_argument("--run-id", help="Run id used with --auto-run-dir, default YYYYMMDD_HHMM.")
    parser.add_argument("--recursive", action="store_true", help="Scan subdirectories recursively.")
    parser.add_argument("--device", default="directml", choices=["directml", "cpu"], help="Target device.")
    parser.add_argument("--dilate", default=8, type=int, help="Mask dilation pixels.")
    parser.add_argument("--mode", default="auto", choices=["auto", "inpaint", "mask"], help="Processing mode.")
    parser.add_argument(
        "--inpaint-backend",
        default="opencv",
        choices=["opencv", "lama-onnx", "webui-api", "comfyui-api"],
        help="Inpaint backend.",
    )
    parser.add_argument("--inpaint-radius", default=2.0, type=float, help="OpenCV inpaint radius.")
    parser.add_argument("--inpaint-method", default="telea", choices=["telea", "ns"], help="OpenCV inpaint method.")
    parser.add_argument("--lama-model", type=Path, help="Path to LaMa ONNX model.")
    parser.add_argument("--webui-url", default="http://127.0.0.1:7860", help="Automatic1111/Forge WebUI API base URL.")
    parser.add_argument("--webui-prompt", default="clean photo background, natural skin, realistic details", help="Prompt for WebUI inpaint.")
    parser.add_argument(
        "--webui-negative-prompt",
        default="text, watermark, logo, letters, caption, subtitles, signature, artifacts, blurry, deformed",
        help="Negative prompt for WebUI inpaint.",
    )
    parser.add_argument("--webui-steps", default=24, type=int, help="WebUI inpaint sampling steps.")
    parser.add_argument("--webui-denoise", default=0.55, type=float, help="WebUI inpaint denoising strength.")
    parser.add_argument("--webui-cfg-scale", default=7.0, type=float, help="WebUI inpaint CFG scale.")
    parser.add_argument("--webui-sampler", default="DPM++ 2M Karras", help="WebUI sampler name.")
    parser.add_argument("--webui-mask-blur", default=8, type=int, help="WebUI inpaint mask blur.")
    parser.add_argument("--webui-timeout", default=600, type=int, help="WebUI API timeout seconds.")
    parser.add_argument("--comfyui-url", default=env_value(env, "COMFYUI_URL", "http://127.0.0.1:8188"), help="ComfyUI API base URL.")
    parser.add_argument(
        "--comfyui-workflow",
        default=env_path(env, "COMFYUI_WORKFLOW"),
        type=Path,
        help="ComfyUI API-format workflow JSON.",
    )
    parser.add_argument("--comfyui-prompt", default=env_value(env, "COMFYUI_PROMPT"), help="Positive prompt override for ComfyUI inpaint.")
    parser.add_argument(
        "--comfyui-negative-prompt",
        default=env_value(env, "COMFYUI_NEGATIVE_PROMPT"),
        help="Negative prompt override for ComfyUI inpaint.",
    )
    parser.add_argument("--comfyui-steps", default=env_int(env, "COMFYUI_STEPS"), type=int, help="KSampler steps override for ComfyUI.")
    parser.add_argument("--comfyui-denoise", default=env_float(env, "COMFYUI_DENOISE"), type=float, help="KSampler denoise override for ComfyUI.")
    parser.add_argument("--comfyui-cfg", default=env_float(env, "COMFYUI_CFG"), type=float, help="KSampler CFG override for ComfyUI.")
    parser.add_argument("--comfyui-sampler", default=env_value(env, "COMFYUI_SAMPLER"), help="KSampler sampler override for ComfyUI.")
    parser.add_argument("--comfyui-scheduler", default=env_value(env, "COMFYUI_SCHEDULER"), help="KSampler scheduler override for ComfyUI.")
    parser.add_argument("--comfyui-timeout", default=env_int(env, "COMFYUI_TIMEOUT", 900), type=int, help="ComfyUI API timeout seconds.")
    parser.add_argument(
        "--comfyui-poll-interval",
        default=env_float(env, "COMFYUI_POLL_INTERVAL", 1.5),
        type=float,
        help="ComfyUI history polling interval seconds.",
    )
    parser.add_argument("--lang", default="ch", help="PaddleOCR language, for example ch or en.")
    parser.add_argument("--ocr-version", default="PP-OCRv4", help="PaddleOCR model version.")
    parser.add_argument(
        "--ocr-upscale-small",
        default=640,
        type=int,
        help="Upscale images whose short side is smaller than this before OCR.",
    )
    parser.add_argument("--ocr-min-score", default=0.55, type=float, help="Minimum OCR confidence score.")
    parser.add_argument("--no-angle-cls", action="store_true", help="Disable PaddleOCR angle classification.")
    parser.add_argument("--no-watermark-corners", action="store_true", help="Disable corner watermark detection.")
    parser.add_argument("--sticker-watermarks", action="store_true", help="Enable experimental full-image sticker detection.")
    parser.add_argument("--edge-text", action="store_true", help="Enable conservative second-pass OCR on image edge bands.")
    parser.add_argument("--edge-text-ratio", default=0.18, type=float, help="Edge band size ratio for second-pass OCR.")
    parser.add_argument("--edge-text-upscale", default=2.0, type=float, help="Upscale factor for edge OCR crops.")
    parser.add_argument("--edge-text-min-score", default=0.5, type=float, help="Minimum OCR score for edge text detections.")
    parser.add_argument("--vision-text", action="store_true", help="Enable optional Florence-2 vision text detection.")
    parser.add_argument("--vision-model", default="microsoft/Florence-2-base", help="Vision model id for optional text detection.")
    parser.add_argument("--vision-task", default="<OCR_WITH_REGION>", help="Florence task prompt for optional text detection.")
    parser.add_argument("--vision-max-new-tokens", default=256, type=int, help="Max new tokens for Florence generation.")
    parser.add_argument("--vision-max-side", default=768, type=int, help="Resize image longest side before Florence detection.")
    parser.add_argument(
        "--vision-trigger",
        default="always",
        choices=["always", "empty", "low-count"],
        help="When to run Florence: always, only when OCR found nothing, or when OCR count is low.",
    )
    parser.add_argument("--vision-low-count", default=1, type=int, help="Threshold used by --vision-trigger low-count.")
    parser.add_argument("--vision-max-area-ratio", default=0.18, type=float, help="Maximum area ratio for each Florence text box.")
    parser.add_argument("--vision-shrink-ratio", default=0.08, type=float, help="Shrink each Florence text box before masking.")
    parser.add_argument("--vision-edge-crops", action="store_true", help="Run Florence on left/right edge crops to catch vertical edge text.")
    parser.add_argument(
        "--vision-edge-crop-trigger",
        default="always",
        choices=["always", "empty", "low-count"],
        help="When to run Florence edge crops.",
    )
    parser.add_argument("--vision-edge-crop-ratio", default=0.42, type=float, help="Width ratio for left/right Florence edge crops.")
    parser.add_argument("--dark-stroke-refine", action="store_true", help="Refine masks with dark strokes near existing edge text masks.")
    parser.add_argument("--dark-stroke-edge-ratio", default=0.45, type=float, help="Horizontal edge band ratio for dark stroke refinement.")
    parser.add_argument("--dark-stroke-anchor-radius", default=36, type=int, help="Maximum distance from existing text mask for dark stroke refinement.")
    parser.add_argument("--dark-stroke-min-area", default=6, type=int, help="Minimum component area for dark stroke refinement.")
    parser.add_argument("--dark-stroke-max-area-ratio", default=0.01, type=float, help="Maximum component area ratio for dark stroke refinement.")
    parser.add_argument("--edge-column-refine", action="store_true", help="Refine missing dark strokes near anchored edge text columns.")
    parser.add_argument("--edge-column-edge-ratio", default=0.46, type=float, help="Horizontal edge band ratio for anchored edge column refinement.")
    parser.add_argument("--edge-column-anchor-radius", default=20, type=int, help="Maximum distance from existing masks for anchored edge column refinement.")
    parser.add_argument("--edge-column-min-area", default=4, type=int, help="Minimum component area for anchored edge column refinement.")
    parser.add_argument("--edge-column-max-area-ratio", default=0.003, type=float, help="Maximum component area ratio for anchored edge column refinement.")
    parser.add_argument("--vertical-text", action="store_true", help="Enable experimental vertical dark text layout detection.")
    parser.add_argument("--vertical-text-min-area", default=18, type=int, help="Minimum component area for vertical text detection.")
    parser.add_argument("--vertical-text-max-area-ratio", default=0.03, type=float, help="Maximum component area ratio for vertical text detection.")
    parser.add_argument("--vertical-text-edge-ratio", default=0.42, type=float, help="Horizontal band ratio for vertical text detection.")
    parser.add_argument("--vertical-columns", action="store_true", help="Enable experimental vertical text column detection.")
    parser.add_argument("--vertical-column-min-height-ratio", default=0.28, type=float, help="Minimum height ratio for vertical text columns.")
    parser.add_argument("--vertical-column-max-width-ratio", default=0.18, type=float, help="Maximum width ratio for vertical text columns.")
    parser.add_argument("--no-face-protect", action="store_true", help="Disable OpenCV face overlap protection.")
    parser.add_argument("--face-padding-ratio", default=0.25, type=float, help="Expand detected face boxes by this ratio.")
    parser.add_argument(
        "--face-overlap-min-pixels",
        default=32,
        type=int,
        help="Minimum mask pixels overlapping face protection to route to AIGC.",
    )
    parser.add_argument(
        "--protected-action",
        default="route",
        choices=["route", "repair"],
        help="Action when mask overlaps protected areas: route to advanced repair bucket or repair with selected backend.",
    )
    parser.add_argument("--watermark-corner-ratio", default=0.18, type=float, help="Corner scan size ratio.")
    parser.add_argument("--watermark-min-area", default=40, type=int, help="Minimum colored watermark area.")
    parser.add_argument("--save-mask", action="store_true", help="Save generated masks next to cleaned outputs.")
    parser.add_argument("--save-debug", action="store_true", help="Save original/mask/result comparison panels.")
    parser.add_argument("--dry-run", action="store_true", help="Only list matched images; do not load PaddleOCR.")
    parser.add_argument("--log", type=Path, help="Optional JSONL processing log path.")
    parser.add_argument("--report", type=Path, help="Optional Markdown batch report path.")
    parser.add_argument("--anonymous-report", action="store_true", help="Hide source filenames in Markdown reports.")
    parser.add_argument("--post-ocr-check", action="store_true", help="Run OCR after repair and mark residual text as quality_failed.")
    return parser.parse_args()


def load_dotenv(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            values[key] = os.environ.get(key, value)
    return values


def env_value(env: dict[str, str], key: str, default: str | None = None) -> str | None:
    return os.environ.get(key) or env.get(key) or default


def env_path(env: dict[str, str], key: str) -> Path | None:
    value = env_value(env, key)
    return Path(value) if value else None


def env_int(env: dict[str, str], key: str, default: int | None = None) -> int | None:
    value = env_value(env, key)
    return int(value) if value not in {None, ""} else default


def env_float(env: dict[str, str], key: str, default: float | None = None) -> float | None:
    value = env_value(env, key)
    return float(value) if value not in {None, ""} else default


def apply_preset(args: argparse.Namespace) -> argparse.Namespace:
    if args.preset != "watermark-first":
        return args

    args.edge_text = True
    args.sticker_watermarks = True
    args.vision_text = True
    args.vision_trigger = "empty"
    args.vision_edge_crops = True
    args.vision_edge_crop_trigger = "empty"
    args.vision_edge_crop_ratio = max(args.vision_edge_crop_ratio, 0.42)
    args.vision_max_area_ratio = max(args.vision_max_area_ratio, 0.24)
    args.vision_shrink_ratio = min(args.vision_shrink_ratio, 0.04)
    args.dark_stroke_refine = False
    args.edge_column_refine = True
    args.edge_column_edge_ratio = max(args.edge_column_edge_ratio, 0.5)
    args.edge_column_anchor_radius = max(args.edge_column_anchor_radius, 32)
    args.vertical_text = False
    args.vertical_columns = False
    args.protected_action = "repair"
    args.post_ocr_check = True

    default_lama = Path("models") / "Carve" / "LaMa-ONNX" / "lama.onnx"
    if args.inpaint_backend == "opencv" and default_lama.exists():
        args.inpaint_backend = "lama-onnx"
        args.lama_model = args.lama_model or default_lama

    return args


def main() -> int:
    args = apply_preset(parse_args())
    input_dir = args.input.resolve()
    output_root = args.output.resolve()
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M")
    output_dir = output_root / run_id if args.auto_run_dir else output_root
    log_path = args.log
    report_path = args.report
    if args.auto_run_dir:
        log_path = log_path or Path("logs") / f"{run_id}.jsonl"
        report_path = report_path or output_root / f"{run_id}.md"

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
            inpainter: OpenCVInpainter | LamaOnnxInpainter | WebUIInpaintInpainter | ComfyUIApiInpainter = LamaOnnxInpainter(
                model_path=args.lama_model.resolve(),
                device=args.device,
            )
        elif args.inpaint_backend == "webui-api":
            inpainter = WebUIInpaintInpainter(
                url=args.webui_url,
                prompt=args.webui_prompt,
                negative_prompt=args.webui_negative_prompt,
                steps=args.webui_steps,
                denoising_strength=args.webui_denoise,
                cfg_scale=args.webui_cfg_scale,
                sampler_name=args.webui_sampler,
                mask_blur=args.webui_mask_blur,
                timeout=args.webui_timeout,
            )
        elif args.inpaint_backend == "comfyui-api":
            if args.comfyui_workflow is None:
                raise RuntimeError("`--comfyui-workflow` is required when `--inpaint-backend comfyui-api` is used.")
            inpainter = ComfyUIApiInpainter(
                url=args.comfyui_url,
                workflow_path=args.comfyui_workflow.resolve(),
                prompt=args.comfyui_prompt,
                negative_prompt=args.comfyui_negative_prompt,
                steps=args.comfyui_steps,
                denoise=args.comfyui_denoise,
                cfg=args.comfyui_cfg,
                sampler=args.comfyui_sampler,
                scheduler=args.comfyui_scheduler,
                timeout=args.comfyui_timeout,
                poll_interval=args.comfyui_poll_interval,
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

    body_detector = BodyOverlapDetector(
        protect_faces=not args.no_face_protect,
        face_padding_ratio=args.face_padding_ratio,
        face_overlap_min_pixels=args.face_overlap_min_pixels,
    )
    try:
        vision_detector: NoopVisionDetector | FlorenceTextDetector
        if args.vision_text:
            vision_detector = FlorenceTextDetector(
                model_id=args.vision_model,
                task=args.vision_task,
                max_new_tokens=args.vision_max_new_tokens,
                max_side=args.vision_max_side,
            )
        else:
            vision_detector = NoopVisionDetector()
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    stats: dict[str, int] = {}
    results: list[ProcessResult] = []

    for image_path in images:
        try:
            result = process_image(
                image_path=image_path,
                input_dir=input_dir,
                output_dir=output_dir,
                detector=detector,
                vision_detector=vision_detector,
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
                edge_text=args.edge_text,
                edge_text_ratio=min(max(args.edge_text_ratio, 0.05), 0.35),
                edge_text_upscale=max(args.edge_text_upscale, 1.0),
                edge_text_min_score=max(min(args.edge_text_min_score, 1.0), 0.0),
                vision_trigger=args.vision_trigger,
                vision_low_count=max(args.vision_low_count, 0),
                vision_max_area_ratio=max(min(args.vision_max_area_ratio, 0.8), 0.01),
                vision_shrink_ratio=max(min(args.vision_shrink_ratio, 0.4), 0.0),
                vision_edge_crops=args.vision_edge_crops,
                vision_edge_crop_trigger=args.vision_edge_crop_trigger,
                vision_edge_crop_ratio=max(min(args.vision_edge_crop_ratio, 0.49), 0.08),
                dark_stroke_refine=args.dark_stroke_refine,
                dark_stroke_edge_ratio=max(min(args.dark_stroke_edge_ratio, 0.5), 0.05),
                dark_stroke_anchor_radius=max(args.dark_stroke_anchor_radius, 1),
                dark_stroke_min_area=max(args.dark_stroke_min_area, 1),
                dark_stroke_max_area_ratio=max(min(args.dark_stroke_max_area_ratio, 0.1), 0.0001),
                edge_column_refine=args.edge_column_refine,
                edge_column_edge_ratio=max(min(args.edge_column_edge_ratio, 0.5), 0.05),
                edge_column_anchor_radius=max(args.edge_column_anchor_radius, 1),
                edge_column_min_area=max(args.edge_column_min_area, 1),
                edge_column_max_area_ratio=max(min(args.edge_column_max_area_ratio, 0.1), 0.0001),
                vertical_text=args.vertical_text,
                vertical_text_min_area=max(args.vertical_text_min_area, 1),
                vertical_text_max_area_ratio=max(min(args.vertical_text_max_area_ratio, 0.2), 0.001),
                vertical_text_edge_ratio=max(min(args.vertical_text_edge_ratio, 0.5), 0.05),
                vertical_columns=args.vertical_columns,
                vertical_column_min_height_ratio=max(min(args.vertical_column_min_height_ratio, 0.9), 0.05),
                vertical_column_max_width_ratio=max(min(args.vertical_column_max_width_ratio, 0.5), 0.02),
                protected_action=args.protected_action,
                post_ocr_check=args.post_ocr_check,
            )
        except Exception as exc:
            result = ProcessResult(path=image_path, status="failed", message=str(exc))

        stats[result.status] = stats.get(result.status, 0) + 1
        results.append(result)
        write_log(log_path, result)
        detail = f", text={result.text_count}" if result.text_count else ""
        watermark = f", watermark={result.watermark_count}" if result.watermark_count else ""
        route = f", route={result.route}" if result.route else ""
        message = f", {result.message}" if result.message else ""
        print(f"[{result.status}] {image_path}{detail}{watermark}{route}{message}")

    write_report(report_path, results, anonymous=args.anonymous_report)
    if report_path is not None:
        print(f"Report: {report_path.resolve()}")
    print("Summary:", ", ".join(f"{key}={value}" for key, value in sorted(stats.items())) or "none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
