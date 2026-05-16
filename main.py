from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.inpaint import LamaOnnxInpainter, OpenCVInpainter
from src.io_utils import iter_images
from src.log import write_log
from src.models import ProcessResult
from src.ocr import PaddleTextDetector
from src.pipeline import process_image
from src.protection import BodyOverlapDetector


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
    parser.add_argument("--no-face-protect", action="store_true", help="Disable OpenCV face overlap protection.")
    parser.add_argument("--face-padding-ratio", default=0.25, type=float, help="Expand detected face boxes by this ratio.")
    parser.add_argument(
        "--face-overlap-min-pixels",
        default=32,
        type=int,
        help="Minimum mask pixels overlapping face protection to route to AIGC.",
    )
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

    body_detector = BodyOverlapDetector(
        protect_faces=not args.no_face_protect,
        face_padding_ratio=args.face_padding_ratio,
        face_overlap_min_pixels=args.face_overlap_min_pixels,
    )
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
