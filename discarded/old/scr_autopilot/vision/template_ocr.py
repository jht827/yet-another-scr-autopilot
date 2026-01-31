from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

TemplateDict = Dict[str, np.ndarray]


@dataclass(frozen=True)
class DigitMatch:
    label: str
    score: float
    bounds: Tuple[int, int, int, int]


@dataclass(frozen=True)
class OcrDebug:
    thresholded: np.ndarray
    contour_boxes: List[Tuple[int, int, int, int]]
    matches: List[DigitMatch]


def load_digit_templates(folder: Path) -> TemplateDict:
    templates: TemplateDict = {}
    for path in folder.glob("*.png"):
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        templates[path.stem] = image
    return templates


def preprocess_roi(
    roi: np.ndarray,
    threshold: Optional[int],
    invert: bool,
) -> np.ndarray:
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    if threshold is None:
        thresh_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
        return cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            thresh_type,
            11,
            2,
        )
    thresh_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    _, thresholded = cv2.threshold(blurred, threshold, 255, thresh_type)
    return thresholded


def segment_digits(
    thresholded: np.ndarray,
    min_area: int,
    min_height: int,
) -> List[Tuple[int, int, int, int]]:
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes: List[Tuple[int, int, int, int]] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w * h < min_area or h < min_height:
            continue
        boxes.append((x, y, w, h))
    boxes.sort(key=lambda item: item[0])
    return boxes


def _match_digit(
    digit_roi: np.ndarray,
    templates: TemplateDict,
    method: int = cv2.TM_CCOEFF_NORMED,
) -> Optional[DigitMatch]:
    if not templates:
        return None
    best_label = None
    best_score = -1.0
    for label, template in templates.items():
        resized = cv2.resize(digit_roi, (template.shape[1], template.shape[0]))
        result = cv2.matchTemplate(resized, template, method)
        _, max_val, _, _ = cv2.minMaxLoc(result)
        if max_val > best_score:
            best_score = float(max_val)
            best_label = label
    if best_label is None:
        return None
    return DigitMatch(label=best_label, score=best_score, bounds=(0, 0, digit_roi.shape[1], digit_roi.shape[0]))


def recognize_digits(
    roi: np.ndarray,
    templates: TemplateDict,
    threshold: Optional[int],
    invert: bool,
    min_area: int,
    min_height: int,
) -> Tuple[str, Optional[OcrDebug]]:
    thresholded = preprocess_roi(roi, threshold, invert)
    boxes = segment_digits(thresholded, min_area=min_area, min_height=min_height)
    matches: List[DigitMatch] = []
    output: List[str] = []
    for x, y, w, h in boxes:
        digit_crop = thresholded[y : y + h, x : x + w]
        match = _match_digit(digit_crop, templates)
        if match is None:
            continue
        matches.append(DigitMatch(label=match.label, score=match.score, bounds=(x, y, w, h)))
        output.append(match.label)
    debug = OcrDebug(thresholded=thresholded, contour_boxes=boxes, matches=matches)
    return "".join(output), debug


def summarize_matches(matches: Iterable[DigitMatch]) -> str:
    parts = [f"{match.label}:{match.score:.3f}@{match.bounds}" for match in matches]
    return ", ".join(parts)
