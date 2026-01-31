from __future__ import annotations

from typing import Optional

import numpy as np
import pytesseract
from PIL import Image, ImageOps

from scr_autopilot.config import OcrOpenCvConfig, OcrTesseractConfig, PixelRoi
from scr_autopilot.vision import OcrDebug, recognize_digits


def clamp_roi(frame: np.ndarray, roi: PixelRoi) -> Optional[np.ndarray]:
    x0 = max(roi.x, 0)
    y0 = max(roi.y, 0)
    x1 = min(roi.x + roi.width, frame.shape[1])
    y1 = min(roi.y + roi.height, frame.shape[0])
    if x1 <= x0 or y1 <= y0:
        return None
    return frame[y0:y1, x0:x1]


def to_pil(frame: np.ndarray) -> Image.Image:
    return Image.fromarray(frame[:, :, ::-1])


def preprocess_for_tesseract(
    roi: np.ndarray,
    settings: OcrTesseractConfig,
) -> Image.Image:
    image = to_pil(roi)
    if settings.scale != 1.0:
        width = max(1, int(image.width * settings.scale))
        height = max(1, int(image.height * settings.scale))
        image = image.resize((width, height), Image.BILINEAR)
    gray = ImageOps.autocontrast(image.convert("L"))
    if settings.threshold is not None:
        gray = gray.point(lambda p: 255 if p > settings.threshold else 0)
    if settings.invert:
        gray = ImageOps.invert(gray)
    return gray


def tesseract_config(settings: OcrTesseractConfig) -> str:
    return (
        f"--psm {settings.psm} --oem {settings.oem} "
        f"-c tessedit_char_whitelist={settings.whitelist}"
    )


def read_roi_text(roi: np.ndarray, settings: OcrTesseractConfig) -> str:
    processed = preprocess_for_tesseract(roi, settings)
    text = pytesseract.image_to_string(processed, config=tesseract_config(settings), lang=settings.lang)
    cleaned = "".join(char for char in text.strip().lower() if char.isalnum() or char == ".")
    return cleaned


def read_speed_opencv(
    roi: np.ndarray,
    templates: dict[str, np.ndarray],
    settings: OcrOpenCvConfig,
) -> tuple[str, Optional[OcrDebug]]:
    if not templates:
        return "", None
    text, debug = recognize_digits(
        roi,
        templates,
        threshold=settings.threshold,
        invert=settings.invert,
        min_area=settings.min_area,
        min_height=settings.min_height,
    )
    cleaned = "".join(char for char in text.strip().lower() if char.isalnum())
    return cleaned, debug
