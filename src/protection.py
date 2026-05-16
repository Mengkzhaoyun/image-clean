from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .models import ProtectionResult


class BodyOverlapDetector:
    def __init__(self, protect_faces: bool, face_padding_ratio: float, face_overlap_min_pixels: int) -> None:
        self._protect_faces = protect_faces
        self._face_padding_ratio = max(float(face_padding_ratio), 0.0)
        self._face_overlap_min_pixels = max(int(face_overlap_min_pixels), 1)
        self._face_detector = self._create_face_detector() if protect_faces else None

    def check(self, image: np.ndarray, mask: np.ndarray) -> ProtectionResult:
        if not self._protect_faces or self._face_detector is None or not np.any(mask):
            return ProtectionResult(overlaps=False)

        protected = self._build_face_protection_mask(image)
        protected_pixels = int(np.count_nonzero(protected))
        if protected_pixels == 0:
            return ProtectionResult(overlaps=False)

        overlap = cv2.bitwise_and(mask, protected)
        overlap_pixels = int(np.count_nonzero(overlap))
        return ProtectionResult(
            overlaps=overlap_pixels >= self._face_overlap_min_pixels,
            reason="face_overlap" if overlap_pixels >= self._face_overlap_min_pixels else None,
            overlap_pixels=overlap_pixels,
            protected_pixels=protected_pixels,
        )

    def _create_face_detector(self) -> cv2.CascadeClassifier | None:
        cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        detector = cv2.CascadeClassifier(str(cascade_path))
        return detector if not detector.empty() else None

    def _build_face_protection_mask(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self._face_detector.detectMultiScale(
            gray,
            scaleFactor=1.08,
            minNeighbors=4,
            minSize=(24, 24),
        )

        protection = np.zeros(image.shape[:2], dtype=np.uint8)
        height, width = image.shape[:2]
        for x, y, w, h in faces:
            pad_x = int(round(w * self._face_padding_ratio))
            pad_y = int(round(h * self._face_padding_ratio))
            x1 = max(int(x) - pad_x, 0)
            y1 = max(int(y) - pad_y, 0)
            x2 = min(int(x + w) + pad_x, width)
            y2 = min(int(y + h) + pad_y, height)
            protection[y1:y2, x1:x2] = 255

        return protection
