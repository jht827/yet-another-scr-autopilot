from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from .screen_capture import NormalizedRoi, WindowRegion, roi_to_normalized


@dataclass(frozen=True)
class RoiSelection:
    roi: Tuple[int, int, int, int]
    normalized: NormalizedRoi


def select_roi(frame: np.ndarray, window_region: Optional[WindowRegion] = None) -> RoiSelection:
    """Display a GUI selector to choose a ROI on a captured frame."""
    roi = cv2.selectROI("Select ROI", frame, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow("Select ROI")
    if window_region is None:
        height, width = frame.shape[:2]
        window_region = WindowRegion(left=0, top=0, width=width, height=height)
    normalized = roi_to_normalized(window_region, roi)
    return RoiSelection(roi=roi, normalized=normalized)
