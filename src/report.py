from __future__ import annotations

from collections import Counter
from pathlib import Path

from .evaluate import evaluate_results
from .models import ProcessResult


def write_report(report_path: Path | None, results: list[ProcessResult], anonymous: bool = False) -> None:
    if report_path is None:
        return

    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    counts = Counter(result.status for result in results)

    lines.append("# Image Clean Report")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Total: {len(results)}")
    for status, count in sorted(counts.items()):
        lines.append(f"- {status}: {count}")

    lines.append("")
    lines.append("## Evaluation")
    lines.append("")
    for item in evaluate_results(results):
        lines.append(f"- {item}")

    lines.append("")
    lines.append("## Details")
    lines.append("")
    lines.append("| Status | Image | Text | Edge Text | Vision | Vision Text | Vision Edge | Edge Column | Text Block | Bright Vert | Stroke Fill | Dark Stroke | Vertical Text | Vertical Col | Residual Text | Watermark | Mask Pixels | Safe | Restricted | Components | Mask Classes | Collisions | Face Overlap | Route | Message |")
    lines.append("| :-- | :-- | --: | --: | :-- | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | :-- | --: | --: | :-- | :-- |")

    for index, result in enumerate(results, start=1):
        diagnostics = result.diagnostics or {}
        image = f"sample-{index:03d}" if anonymous else result.path.name.replace("|", "\\|")
        route = (result.route or "").replace("|", "\\|")
        message = (result.message or "").replace("|", "\\|")
        mask_pixels = diagnostics.get("mask_pixels", "")
        face_overlap = diagnostics.get("face_overlap_pixels", "")
        edge_text = diagnostics.get("edge_text_count", "")
        vision = "yes" if diagnostics.get("vision_triggered") else "no"
        vision_text = diagnostics.get("vision_text_count", "")
        vision_edge_text = diagnostics.get("vision_edge_text_count", "")
        edge_column = diagnostics.get("edge_column_count", "")
        text_block = diagnostics.get("text_block_count", "")
        vertical_bright = diagnostics.get("vertical_bright_count", "")
        stroke_completion = diagnostics.get("stroke_completion_count", "")
        post_dark = diagnostics.get("post_dark_residual_count", "")
        dark_stroke = diagnostics.get("dark_stroke_count", "")
        if post_dark not in ("", None):
            dark_stroke = f"{dark_stroke}+post:{post_dark}"
        vertical_text = diagnostics.get("vertical_text_count", "")
        vertical_col = diagnostics.get("vertical_column_count", "")
        residual_text = diagnostics.get("residual_text_count", "")
        component_count = diagnostics.get("mask_component_count", "")
        mask_classes = str(diagnostics.get("mask_top_classes", "")).replace("|", "\\|")
        collisions = diagnostics.get("mask_collision_count", "")
        safe_pixels = diagnostics.get("safe_mask_pixels", "")
        restricted_pixels = diagnostics.get("restricted_mask_pixels", "")
        lines.append(
            f"| {result.status} | {image} | {result.text_count} | {edge_text} | {vision} | {vision_text} | {vision_edge_text} | {edge_column} | {text_block} | {vertical_bright} | {stroke_completion} | {dark_stroke} | {vertical_text} | {vertical_col} | {residual_text} | {result.watermark_count} | "
            f"{mask_pixels} | {safe_pixels} | {restricted_pixels} | {component_count} | {mask_classes} | {collisions} | {face_overlap} | {route} | {message} |"
        )

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
