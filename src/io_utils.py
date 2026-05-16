from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def iter_images(input_dir: Path, recursive: bool) -> Iterable[Path]:
    pattern = "**/*" if recursive else "*"
    for path in input_dir.glob(pattern):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def read_image(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to read image: {path}")
    return image


def write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix or ".png"
    ok, encoded = cv2.imencode(ext, image)
    if not ok:
        raise ValueError(f"Failed to encode image: {path}")
    encoded.tofile(str(path))


def relative_output_path(input_dir: Path, output_dir: Path, image_path: Path) -> Path:
    return output_dir / image_path.relative_to(input_dir)


def mask_output_path(output_dir: Path, image_path: Path, input_dir: Path) -> Path:
    relative = image_path.relative_to(input_dir)
    return output_dir / relative.parent / f"{relative.stem}.mask.png"


def debug_output_path(output_dir: Path, image_path: Path, input_dir: Path) -> Path:
    relative = image_path.relative_to(input_dir)
    return output_dir / relative.parent / f"{relative.stem}.debug.jpg"

