"""Vision utilities for HUD capture and OCR-like parsing."""

from scr_autopilot.vision.hud_capture import (
    FrameGrabber,
    FrameSample,
    extract_roi,
    load_templates,
    match_templates,
    preprocess_for_ocr,
    vote_frames,
)
from scr_autopilot.vision.screen_capture import (
    NormalizedRoi,
    ScreenGrabber,
    WindowRegion,
    find_window_region,
    roi_from_normalized,
    roi_to_normalized,
)
from scr_autopilot.vision.template_ocr import (
    DigitMatch,
    OcrDebug,
    load_digit_templates,
    recognize_digits,
    summarize_matches,
)
from scr_autopilot.vision.roi_selector import RoiSelection, select_roi

__all__ = [
    "FrameGrabber",
    "FrameSample",
    "extract_roi",
    "load_templates",
    "match_templates",
    "preprocess_for_ocr",
    "vote_frames",
    "NormalizedRoi",
    "ScreenGrabber",
    "WindowRegion",
    "find_window_region",
    "roi_from_normalized",
    "roi_to_normalized",
    "RoiSelection",
    "select_roi",
    "DigitMatch",
    "OcrDebug",
    "load_digit_templates",
    "recognize_digits",
    "summarize_matches",
]
