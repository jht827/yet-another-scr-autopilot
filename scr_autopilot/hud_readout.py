import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional, Tuple

import cv2
import mss
import numpy as np

from scr_autopilot.vision import (
    NormalizedRoi,
    ScreenGrabber,
    WindowRegion,
    find_window_region,
    load_templates,
    preprocess_for_ocr,
    roi_from_normalized,
)


@dataclass(frozen=True)
class HudRois:
    speed: NormalizedRoi
    limit: NormalizedRoi
    distance: NormalizedRoi


DEFAULT_ROIS = HudRois(
    speed=NormalizedRoi(
        x=0.5218560860793544,
        y=0.5764462809917356,
        width=0.0484196368527236,
        height=0.045454545454545456,
    ),
    limit=NormalizedRoi(
        x=0.5292535305985205,
        y=0.6539256198347108,
        width=0.03227975790181574,
        height=0.03409090909090909,
    ),
    distance=NormalizedRoi(
        x=-0.006724949562878279,
        y=0.6590909090909091,
        width=0.02824478816408877,
        height=0.01962809917355372,
    ),
)


def resolve_region(args: argparse.Namespace) -> WindowRegion:
    if args.region:
        left, top, width, height = (int(part) for part in args.region.split(","))
        return WindowRegion(left=left, top=top, width=width, height=height)
    with mss.mss() as sct:
        monitor = sct.monitors[args.monitor]
        return WindowRegion(
            left=int(monitor["left"]),
            top=int(monitor["top"]),
            width=int(monitor["width"]),
            height=int(monitor["height"]),
        )


def debug_log(enabled: bool, message: str) -> None:
    if enabled:
        print(f"[debug] {message}")


def build_region_provider(
    args: argparse.Namespace,
    log: Callable[[str], None],
) -> Callable[[], WindowRegion]:
    fallback = resolve_region(args)
    if args.no_window or not args.window_title:
        log(f"Using fixed region: {fallback}")
        return lambda: fallback

    last_region = fallback
    last_error: Optional[str] = None
    last_refresh = 0.0

    def provider() -> WindowRegion:
        nonlocal last_region, last_error, last_refresh
        now = time.time()
        if now - last_refresh < args.window_refresh:
            return last_region
        last_refresh = now
        try:
            last_region = find_window_region(args.window_title)
            if last_error:
                log("Window capture recovered.")
            last_error = None
        except RuntimeError as exc:
            if str(exc) != last_error:
                log(f"Window capture failed: {exc}. Falling back to {last_region}.")
                last_error = str(exc)
        return last_region

    log(f"Attempting to capture window matching '{args.window_title}'.")
    return provider


def clamp_roi(frame: np.ndarray, roi: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
    x, y, w, h = roi
    x0 = max(x, 0)
    y0 = max(y, 0)
    x1 = min(x + w, frame.shape[1])
    y1 = min(y + h, frame.shape[0])
    if x1 <= x0 or y1 <= y0:
        return None
    return frame[y0:y1, x0:x1]


def segment_digits(binary: np.ndarray) -> Iterable[np.ndarray]:
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    height, width = binary.shape[:2]
    min_height = max(6, int(height * 0.4))
    min_area = int(height * width * 0.01)
    boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if h < min_height or w < 3:
            continue
        if w * h < min_area:
            continue
        boxes.append((x, y, w, h))
    boxes.sort(key=lambda box: box[0])
    return [binary[y : y + h, x : x + w] for x, y, w, h in boxes]


def match_digit(digit: np.ndarray, templates: Dict[str, np.ndarray]) -> Tuple[str, float]:
    best_label = "?"
    best_score = -1.0
    if not templates:
        return best_label, best_score
    template_shape = next(iter(templates.values())).shape[::-1]
    resized = cv2.resize(digit, template_shape)
    for label, template in templates.items():
        result = cv2.matchTemplate(resized, template, cv2.TM_CCOEFF_NORMED)
        _, score, _, _ = cv2.minMaxLoc(result)
        if score > best_score:
            best_label = label
            best_score = float(score)
    return best_label, best_score


def decode_roi(roi: np.ndarray, templates: Dict[str, np.ndarray]) -> Tuple[str, float]:
    processed = preprocess_for_ocr(roi)
    digits = segment_digits(processed)
    if not digits:
        return "", 0.0
    labels: list[str] = []
    scores: list[float] = []
    for digit in digits:
        label, score = match_digit(digit, templates)
        labels.append(label)
        scores.append(score)
    return "".join(labels), float(min(scores)) if scores else 0.0


def format_value(label: str, score: float, threshold: float) -> str:
    if not label or score < threshold:
        return "--"
    return label


def main() -> None:
    parser = argparse.ArgumentParser(description="Read HUD values from normalized ROIs.")
    parser.add_argument(
        "--templates",
        type=Path,
        required=True,
        help="Folder with digit templates (0.png-9.png) for OCR matching.",
    )
    parser.add_argument(
        "--monitor",
        type=int,
        default=1,
        help="Monitor index for capture (used when --region is omitted).",
    )
    parser.add_argument(
        "--region",
        help="Optional capture region as 'left,top,width,height'.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.05,
        help="Seconds between terminal updates.",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.5,
        help="Minimum template score to accept a digit.",
    )
    parser.add_argument(
        "--window-title",
        default="Roblox",
        help="Window title/owner hint to capture (default: Roblox).",
    )
    parser.add_argument(
        "--no-window",
        action="store_true",
        help="Disable window capture and use monitor/region instead.",
    )
    parser.add_argument(
        "--window-refresh",
        type=float,
        default=1.0,
        help="Seconds between window region refreshes when using --window-title.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging about capture status and frame timing.",
    )
    parser.add_argument(
        "--debug-dir",
        type=Path,
        help="Optional folder to dump a full frame + ROI crops for debugging.",
    )
    args = parser.parse_args()

    templates = load_templates(args.templates)
    if not templates:
        raise RuntimeError(f"No templates found in {args.templates} (expected *.png files).")

    log = lambda message: debug_log(args.debug, message)
    region_provider = build_region_provider(args, log)
    grabber = ScreenGrabber(region_provider)
    grabber.start()
    debug_dumped = False
    last_debug = 0.0
    last_sample_time: Optional[float] = None
    try:
        while True:
            sample = grabber.read_latest()
            if sample is None:
                if args.debug and time.time() - last_debug > 1.0:
                    log("Waiting for first frame...")
                    last_debug = time.time()
                time.sleep(0.01)
                continue
            frame = sample.frame
            region = region_provider()
            speed_roi = clamp_roi(frame, roi_from_normalized(region, DEFAULT_ROIS.speed))
            limit_roi = clamp_roi(frame, roi_from_normalized(region, DEFAULT_ROIS.limit))
            distance_roi = clamp_roi(frame, roi_from_normalized(region, DEFAULT_ROIS.distance))
            if args.debug_dir and not debug_dumped:
                args.debug_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(args.debug_dir / "frame.png"), frame)
                if speed_roi is not None:
                    cv2.imwrite(str(args.debug_dir / "speed_roi.png"), speed_roi)
                if limit_roi is not None:
                    cv2.imwrite(str(args.debug_dir / "limit_roi.png"), limit_roi)
                if distance_roi is not None:
                    cv2.imwrite(str(args.debug_dir / "distance_roi.png"), distance_roi)
                debug_dumped = True

            speed_label, speed_score = ("", 0.0)
            limit_label, limit_score = ("", 0.0)
            distance_label, distance_score = ("", 0.0)
            if speed_roi is not None:
                speed_label, speed_score = decode_roi(speed_roi, templates)
            if limit_roi is not None:
                limit_label, limit_score = decode_roi(limit_roi, templates)
            if distance_roi is not None:
                distance_label, distance_score = decode_roi(distance_roi, templates)

            speed_value = format_value(speed_label, speed_score, args.min_score)
            limit_value = format_value(limit_label, limit_score, args.min_score)
            distance_value = format_value(distance_label, distance_score, args.min_score)

            line = (
                f"Speed: {speed_value} | Limit: {limit_value} | "
                f"Next Station: {distance_value}"
            )
            print(f"\r{line:<80}", end="", flush=True)
            if args.debug and time.time() - last_debug > 1.0:
                now = time.time()
                frame_age = now - sample.timestamp
                fps = 0.0
                if last_sample_time is not None:
                    fps = 1.0 / max(now - last_sample_time, 1e-6)
                last_sample_time = now
                mean_luma = float(np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)))
                log(
                    "Frame ok | "
                    f"region={region.width}x{region.height}+{region.left},{region.top} | "
                    f"age={frame_age:.3f}s | fps≈{fps:.1f} | mean_luma={mean_luma:.1f}"
                )
                last_debug = now
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nStopping HUD reader.")
    finally:
        grabber.stop()


if __name__ == "__main__":
    main()
