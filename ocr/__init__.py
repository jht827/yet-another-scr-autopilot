"""OCR package entry points."""

from .ocr_monitor import OcrReading, stream_ocr

__all__ = ["OcrReading", "stream_ocr"]
