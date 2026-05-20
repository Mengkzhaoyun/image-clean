from __future__ import annotations

from collections import Counter

from .models import ProcessResult


def evaluate_results(results: list[ProcessResult]) -> list[str]:
    if not results:
        return ["No images were processed."]

    counts = Counter(result.status for result in results)
    total = len(results)
    needs_aigc = counts.get("needs_aigc", 0)
    failed = counts.get("failed", 0)
    skipped = counts.get("skipped", 0)
    cleaned = counts.get("cleaned", 0)
    cleaned_protected = counts.get("cleaned_protected", 0)
    quality_failed = counts.get("quality_failed", 0)

    lines: list[str] = []
    lines.append(
        f"Processed {total} image(s): {cleaned} cleaned, {cleaned_protected} protected repairs, "
        f"{quality_failed} quality failed, {needs_aigc} need advanced repair, {skipped} skipped, {failed} failed."
    )

    if failed:
        lines.append("There are failed images; use the JSONL diagnostics to drive the next automatic fix.")
    if needs_aigc:
        ratio = needs_aigc / total
        if ratio >= 0.4:
            lines.append("Many images were routed to the advanced-repair bucket. The protection rule is conservative or text often overlaps people.")
        else:
            lines.append("Some images need advanced repair because masks overlapped protected regions.")
    if cleaned_protected:
        lines.append(f"{cleaned_protected} protected-overlap image(s) were automatically repaired by the selected backend.")
    if quality_failed:
        lines.append(f"{quality_failed} image(s) failed automatic post-repair OCR quality check and need another automatic strategy.")
    if skipped:
        lines.append("Skipped images may be truly clean or OCR misses; use the next iteration to improve automatic recall.")

    high_face_overlap = [
        result
        for result in results
        if (result.diagnostics or {}).get("face_overlap_pixels", 0) >= 1000
    ]
    if high_face_overlap:
        lines.append(f"{len(high_face_overlap)} image(s) have high protected-overlap risk; compare automatic repair quality before making this route default.")

    high_mask = [
        result
        for result in results
        if (result.diagnostics or {}).get("mask_pixels", 0) >= 50000
    ]
    if high_mask:
        lines.append(f"{len(high_mask)} image(s) have large masks; next rules should distinguish over-detection from large poster text.")

    unknown_components = sum((result.diagnostics or {}).get("mask_unknown_component_count", 0) for result in results)
    if unknown_components:
        lines.append(f"Mask analysis found {unknown_components} component(s) that were not attributed to a known source; inspect these for missing classifier coverage.")

    large_components = sum((result.diagnostics or {}).get("mask_large_component_count", 0) for result in results)
    if large_components:
        lines.append(f"Mask analysis found {large_components} large component(s); classify them as true poster text/sticker or over-detection.")

    collision_components = sum((result.diagnostics or {}).get("mask_collision_count", 0) for result in results)
    if collision_components:
        lines.append(f"Mask/protection collision analysis found {collision_components} component(s) touching protected regions.")

    safe_pixels = sum((result.diagnostics or {}).get("safe_mask_pixels", 0) for result in results)
    restricted_pixels = sum((result.diagnostics or {}).get("restricted_mask_pixels", 0) for result in results)
    if safe_pixels or restricted_pixels:
        lines.append(f"Three-mask split totals: safe={safe_pixels} px, restricted={restricted_pixels} px.")

    edge_text_count = sum((result.diagnostics or {}).get("edge_text_count", 0) for result in results)
    if edge_text_count:
        lines.append(f"Edge OCR added {edge_text_count} text candidate(s); evaluate false positives through generated masks and diagnostics.")

    vision_text_count = sum((result.diagnostics or {}).get("vision_text_count", 0) for result in results)
    if vision_text_count:
        lines.append(f"Vision text detection added {vision_text_count} text candidate(s); compare this recall against OCR-only runs.")
    vision_triggered_count = sum(1 for result in results if (result.diagnostics or {}).get("vision_triggered"))
    if vision_triggered_count:
        lines.append(f"Vision detection ran on {vision_triggered_count} image(s).")
    vision_edge_text_count = sum((result.diagnostics or {}).get("vision_edge_text_count", 0) for result in results)
    if vision_edge_text_count:
        lines.append(f"Vision edge crops added {vision_edge_text_count} text candidate(s); use this to judge edge vertical-text recall.")
    vision_edge_triggered_count = sum(1 for result in results if (result.diagnostics or {}).get("vision_edge_triggered"))
    if vision_edge_triggered_count:
        lines.append(f"Vision edge crops ran on {vision_edge_triggered_count} image(s).")
    edge_column_count = sum((result.diagnostics or {}).get("edge_column_count", 0) for result in results)
    if edge_column_count:
        lines.append(f"Anchored edge-column refinement added {edge_column_count} dark stroke component(s) near existing masks.")
    text_block_count = sum((result.diagnostics or {}).get("text_block_count", 0) for result in results)
    if text_block_count:
        lines.append(f"Text block expansion added {text_block_count} region-level component(s) around detected text.")
    stroke_completion_count = sum((result.diagnostics or {}).get("stroke_completion_count", 0) for result in results)
    if stroke_completion_count:
        lines.append(f"Text stroke completion added {stroke_completion_count} high-contrast component(s) near existing text masks.")
    edge_column_discarded_count = sum(1 for result in results if (result.diagnostics or {}).get("edge_column_discarded"))
    if edge_column_discarded_count:
        lines.append(f"Anchored edge-column refinement was discarded on {edge_column_discarded_count} image(s) because it increased protected-overlap risk.")
    dark_stroke_count = sum((result.diagnostics or {}).get("dark_stroke_count", 0) for result in results)
    if dark_stroke_count:
        lines.append(f"Dark stroke refinement added {dark_stroke_count} component(s) near existing edge text masks.")
    dark_stroke_discarded_count = sum(1 for result in results if (result.diagnostics or {}).get("dark_stroke_discarded"))
    if dark_stroke_discarded_count:
        lines.append(f"Dark stroke refinement was discarded on {dark_stroke_discarded_count} image(s) because it increased protected-overlap risk.")
    post_dark_residual_count = sum((result.diagnostics or {}).get("post_dark_residual_count", 0) for result in results)
    if post_dark_residual_count:
        lines.append(f"Post-repair dark residual cleanup found {post_dark_residual_count} small component(s) near text masks.")

    vertical_text_count = sum((result.diagnostics or {}).get("vertical_text_count", 0) for result in results)
    if vertical_text_count:
        lines.append(f"Vertical layout detection added {vertical_text_count} component candidate(s); check mask growth and protected overlap.")
    vertical_column_count = sum((result.diagnostics or {}).get("vertical_column_count", 0) for result in results)
    if vertical_column_count:
        lines.append(f"Vertical column detection added {vertical_column_count} column candidate(s); compare against component-based vertical detection.")

    residual_text_count = sum((result.diagnostics or {}).get("residual_text_count", 0) for result in results)
    if residual_text_count:
        lines.append(f"Post-repair OCR found {residual_text_count} residual text candidate(s).")

    if (cleaned or cleaned_protected or quality_failed) and not failed:
        lines.append("Cleaned outputs are available in the run directory for automatic comparison across iterations.")

    return lines
