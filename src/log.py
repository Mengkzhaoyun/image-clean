from __future__ import annotations

import json
from pathlib import Path

from .models import ProcessResult


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
        "diagnostics": result.diagnostics,
    }
    with log_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload, ensure_ascii=False) + "\n")
