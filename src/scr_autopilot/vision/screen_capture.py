import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np

try:
    import mss
except ImportError:  # pragma: no cover - optional dependency
    mss = None


@dataclass(frozen=True)
class WindowRegion:
    left: int
    top: int
    width: int
    height: int


@dataclass(frozen=True)
class NormalizedRoi:
    x: float
    y: float
    width: float
    height: float


@dataclass(frozen=True)
class ScreenSample:
    frame: np.ndarray
    timestamp: float


def roi_from_normalized(region: WindowRegion, roi: NormalizedRoi) -> Tuple[int, int, int, int]:
    """Convert normalized ROI to absolute pixels inside a window region."""
    x = int(region.left + roi.x * region.width)
    y = int(region.top + roi.y * region.height)
    width = int(roi.width * region.width)
    height = int(roi.height * region.height)
    return x, y, width, height


def roi_to_normalized(region: WindowRegion, roi: Tuple[int, int, int, int]) -> NormalizedRoi:
    """Convert absolute ROI (x, y, w, h) into normalized coordinates for a window region."""
    x, y, width, height = roi
    return NormalizedRoi(
        x=(x - region.left) / region.width,
        y=(y - region.top) / region.height,
        width=width / region.width,
        height=height / region.height,
    )


class ScreenGrabber:
    """Low-latency screen capture for a window region using MSS."""

    def __init__(self, region_provider: Callable[[], WindowRegion]) -> None:
        if mss is None:
            raise ImportError("mss is required for ScreenGrabber. Install with `pip install mss`.")
        self._region_provider = region_provider
        self._lock = threading.Lock()
        self._latest: Optional[ScreenSample] = None
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

    def read_latest(self) -> Optional[ScreenSample]:
        with self._lock:
            return self._latest

    def _run(self) -> None:
        with mss.mss() as sct:
            while self._running:
                region = self._region_provider()
                monitor = {
                    "left": region.left,
                    "top": region.top,
                    "width": region.width,
                    "height": region.height,
                }
                screenshot = sct.grab(monitor)
                frame = np.array(screenshot)[:, :, :3]
                frame = frame[:, :, ::-1]  # BGRA -> BGR
                timestamp = time.time()
                with self._lock:
                    self._latest = ScreenSample(frame=frame, timestamp=timestamp)
