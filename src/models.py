from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TextDetection:
    box: list[tuple[int, int]]
    text: str
    score: float


@dataclass(frozen=True)
class ProcessResult:
    path: Path
    status: str
    output: Path | None = None
    mask: Path | None = None
    text_count: int = 0
    watermark_count: int = 0
    route: str | None = None
    message: str | None = None
    diagnostics: dict[str, Any] | None = None


@dataclass(frozen=True)
class InpaintResult:
    image: np.ndarray
    route: str


@dataclass(frozen=True)
class ProtectionResult:
    overlaps: bool
    reason: str | None = None
    overlap_pixels: int = 0
    protected_pixels: int = 0

