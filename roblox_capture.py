import argparse
import os
import time
from typing import Optional

import cv2

from scr_autopilot.hud_config import PixelRoi, WINDOW_TITLE, format_roi
from scr_autopilot.vision import ScreenGrabber, WindowRegion, select_roi

try:
    import Quartz
except ImportError as exc:  # pragma: no cover - platform dependency
    raise ImportError(
        "Quartz is required on macOS. Install pyobjc with `pip install pyobjc`."
    ) from exc


def find_window_region(title_hint: str) -> WindowRegion:
    current_pid = os.getpid()
    options = Quartz.kCGWindowListOptionOnScreenOnly | Quartz.kCGWindowListExcludeDesktopElements
    window_list = Quartz.CGWindowListCopyWindowInfo(options, Quartz.kCGNullWindowID)
    for window in window_list:
        owner_pid = window.get("kCGWindowOwnerPID")
        if owner_pid == current_pid:
            continue
        window_title = window.get("kCGWindowName", "") or ""
        owner_name = window.get("kCGWindowOwnerName", "") or ""
        if title_hint.lower() in window_title.lower() or title_hint.lower() in owner_name.lower():
            bounds = window.get("kCGWindowBounds", {})
            return WindowRegion(
                left=int(bounds.get("X", 0)),
                top=int(bounds.get("Y", 0)),
                width=int(bounds.get("Width", 0)),
                height=int(bounds.get("Height", 0)),
            )
    raise RuntimeError(f"Window matching '{title_hint}' not found.")


def region_provider(title_hint: str) -> WindowRegion:
    return find_window_region(title_hint)


def capture_loop(title_hint: str, show_preview: bool) -> None:
    grabber = ScreenGrabber(lambda: region_provider(title_hint))
    grabber.start()
    if show_preview:
        cv2.namedWindow("Roblox Capture", cv2.WINDOW_NORMAL)
    print("Press 'b' to box ROI, 'q' to quit.")
    while True:
        sample = grabber.read_latest()
        if sample is None:
            time.sleep(0.01)
            continue
        frame = sample.frame
        if show_preview:
            cv2.imshow("Roblox Capture", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("b"):
            selection = select_roi(frame, region_provider(title_hint))
            x, y, w, h = selection.roi
            print("Absolute ROI:", selection.roi)
            print("Normalized ROI:", selection.normalized)
            print("HUD config snippet:", format_roi(PixelRoi(x=x, y=y, width=w, height=h)))
        elif key == ord("q"):
            break
        time.sleep(0.005)
    grabber.stop()
    cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture Roblox window and select HUD ROI.")
    parser.add_argument(
        "--title",
        default=WINDOW_TITLE,
        help="Window title/owner hint to match.",
    )
    parser.add_argument("--no-preview", action="store_true", help="Disable live preview window.")
    args = parser.parse_args()
    capture_loop(args.title, show_preview=not args.no_preview)


if __name__ == "__main__":
    main()
