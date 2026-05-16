from __future__ import annotations

from collections import Counter
from pathlib import Path

from .models import ProcessResult


def write_report(report_path: Path | None, results: list[ProcessResult]) -> None:
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
    lines.append("## Details")
    lines.append("")
    lines.append("| Status | Image | Text | Watermark | Mask Pixels | Face Overlap | Route | Message |")
    lines.append("| :-- | :-- | --: | --: | --: | --: | :-- | :-- |")

    for result in results:
        diagnostics = result.diagnostics or {}
        image = result.path.name.replace("|", "\\|")
        route = (result.route or "").replace("|", "\\|")
        message = (result.message or "").replace("|", "\\|")
        mask_pixels = diagnostics.get("mask_pixels", "")
        face_overlap = diagnostics.get("face_overlap_pixels", "")
        lines.append(
            f"| {result.status} | {image} | {result.text_count} | {result.watermark_count} | "
            f"{mask_pixels} | {face_overlap} | {route} | {message} |"
        )

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
