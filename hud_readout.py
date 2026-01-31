import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import cv2
import mss
import numpy as np

from scr_autopilot.vision import (
    NormalizedRoi,
    ScreenGrabber,
    WindowRegion,
    load_templates,
    preprocess_for_ocr,
    roi_from_normalized,
)

DEFAULT_TEMPLATE_DIRS = {
    "speed": Path("templates/speed"),
    "limit": Path("templates/limit"),
    "distance": Path("templates/distance"),
}


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
    min_height = max(4, int(height * 0.25))
    min_area = int(height * width * 0.003)
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
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    processed = preprocess_for_ocr(roi)
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, otsu_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    variants = (processed, cv2.bitwise_not(processed), otsu, otsu_inv)
    best_label = ""
    best_score = 0.0
    for candidate in variants:
        digits = segment_digits(candidate)
        if not digits:
            label, score = match_digit(candidate, templates)
            if score > best_score:
                best_label = label
                best_score = score
            continue
        labels: list[str] = []
        scores: list[float] = []
        for digit in digits:
            label, score = match_digit(digit, templates)
            labels.append(label)
            scores.append(score)
        if scores and min(scores) > best_score:
            best_label = "".join(labels)
            best_score = float(min(scores))
    return best_label, best_score


def dump_roi_variants(
    dump_dir: Path,
    label: str,
    roi: np.ndarray,
) -> None:
    dump_dir.mkdir(parents=True, exist_ok=True)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    processed = preprocess_for_ocr(roi)
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, otsu_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cv2.imwrite(str(dump_dir / f"{label}_raw.png"), roi)
    cv2.imwrite(str(dump_dir / f"{label}_processed.png"), processed)
    cv2.imwrite(str(dump_dir / f"{label}_processed_inv.png"), cv2.bitwise_not(processed))
    cv2.imwrite(str(dump_dir / f"{label}_otsu.png"), otsu)
    cv2.imwrite(str(dump_dir / f"{label}_otsu_inv.png"), otsu_inv)


def format_value(label: str, score: float, threshold: float) -> str:
    if not label or score < threshold:
        return "--"
    return label


def load_template_dir(path: Path, label: str) -> Dict[str, np.ndarray]:
    templates = load_templates(path)
    if not templates:
        print(
            f"Warning: no templates found for {label} in {path}. "
            "Add digit PNGs (0.png-9.png) or pass a custom path."
        )
    return templates


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read HUD values from normalized ROIs with per-field templates.",
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
        "--speed-templates",
        type=Path,
        default=DEFAULT_TEMPLATE_DIRS["speed"],
        help=f"Template folder for speed digits (default: {DEFAULT_TEMPLATE_DIRS['speed']}).",
    )
    parser.add_argument(
        "--limit-templates",
        type=Path,
        default=DEFAULT_TEMPLATE_DIRS["limit"],
        help=f"Template folder for limit digits (default: {DEFAULT_TEMPLATE_DIRS['limit']}).",
    )
    parser.add_argument(
        "--distance-templates",
        type=Path,
        default=DEFAULT_TEMPLATE_DIRS["distance"],
        help=f"Template folder for distance digits (default: {DEFAULT_TEMPLATE_DIRS['distance']}).",
    )
    parser.add_argument(
        "--dump-rois",
        type=Path,
        help="Optional folder to dump ROI images (raw + processed variants) on first frame.",
    )
    args = parser.parse_args()

    speed_templates = load_template_dir(args.speed_templates, "speed")
    limit_templates = load_template_dir(args.limit_templates, "limit")
    distance_templates = load_template_dir(args.distance_templates, "distance")
    if not speed_templates and not limit_templates and not distance_templates:
        print(
            "No templates loaded. Use --dump-rois to capture HUD digits and build templates, "
            "or provide template folders with digit PNGs."
        )

    region = resolve_region(args)
    grabber = ScreenGrabber(lambda: region)
    grabber.start()
    dumped = False
    try:
        while True:
            sample = grabber.read_latest()
            if sample is None:
                time.sleep(0.01)
                continue
            frame = sample.frame
            speed_roi = clamp_roi(frame, roi_from_normalized(region, DEFAULT_ROIS.speed))
            limit_roi = clamp_roi(frame, roi_from_normalized(region, DEFAULT_ROIS.limit))
            distance_roi = clamp_roi(frame, roi_from_normalized(region, DEFAULT_ROIS.distance))
            if args.dump_rois and not dumped:
                if speed_roi is not None:
                    dump_roi_variants(args.dump_rois, "speed", speed_roi)
                if limit_roi is not None:
                    dump_roi_variants(args.dump_rois, "limit", limit_roi)
                if distance_roi is not None:
                    dump_roi_variants(args.dump_rois, "distance", distance_roi)
                dumped = True

            speed_label, speed_score = ("", 0.0)
            limit_label, limit_score = ("", 0.0)
            distance_label, distance_score = ("", 0.0)
            if speed_roi is not None:
                speed_label, speed_score = decode_roi(speed_roi, speed_templates)
            if limit_roi is not None:
                limit_label, limit_score = decode_roi(limit_roi, limit_templates)
            if distance_roi is not None:
                distance_label, distance_score = decode_roi(distance_roi, distance_templates)

            speed_value = format_value(speed_label, speed_score, args.min_score)
            limit_value = format_value(limit_label, limit_score, args.min_score)
            distance_value = format_value(distance_label, distance_score, args.min_score)

            line = (
                f"Speed: {speed_value} | Limit: {limit_value} | "
                f"Next Station: {distance_value}"
            )
            print(f"\r{line:<80}", end="", flush=True)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nStopping HUD reader.")
    finally:
        grabber.stop()


if __name__ == "__main__":
    main()
