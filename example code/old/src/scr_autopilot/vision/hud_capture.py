import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class FrameSample:
    frame: np.ndarray
    timestamp: float


class FrameGrabber:
    """Low-latency frame grabber that always keeps the most recent frame."""

    def __init__(self, source: int | str = 0) -> None:
        self._capture = cv2.VideoCapture(source)
        self._lock = threading.Lock()
        self._latest: Optional[FrameSample] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        self._capture.release()

    def read_latest(self) -> Optional[FrameSample]:
        with self._lock:
            return self._latest

    def _run(self) -> None:
        while self._running:
            ok, frame = self._capture.read()
            if not ok:
                time.sleep(0.01)
                continue
            timestamp = time.time()
            with self._lock:
                self._latest = FrameSample(frame=frame, timestamp=timestamp)


def extract_roi(frame: np.ndarray, roi: Tuple[int, int, int, int]) -> np.ndarray:
    """Extract a region of interest defined as (x, y, width, height)."""
    x, y, w, h = roi
    return frame[y : y + h, x : x + w]


def preprocess_for_ocr(roi: np.ndarray) -> np.ndarray:
    """Prepare a ROI for template-based OCR-like matching."""
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    thresholded = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2,
    )
    return thresholded


def load_templates(folder: Path) -> Dict[str, np.ndarray]:
    """Load template images from a folder keyed by filename stem."""
    templates: Dict[str, np.ndarray] = {}
    for path in folder.glob("*.png"):
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        templates[path.stem] = image
    return templates


def match_templates(
    processed_roi: np.ndarray,
    templates: Dict[str, np.ndarray],
    method: int = cv2.TM_CCOEFF_NORMED,
) -> Dict[str, float]:
    """Return similarity scores for each template in the ROI."""
    scores: Dict[str, float] = {}
    for label, template in templates.items():
        result = cv2.matchTemplate(processed_roi, template, method)
        _, max_val, _, _ = cv2.minMaxLoc(result)
        scores[label] = float(max_val)
    return scores


def vote_frames(score_history: Iterable[Dict[str, float]]) -> Optional[str]:
    """Pick the most frequent top label from recent frame scores."""
    winners: list[str] = []
    for scores in score_history:
        if not scores:
            continue
        winners.append(max(scores, key=scores.get))
    if not winners:
        return None
    return max(set(winners), key=winners.count)
