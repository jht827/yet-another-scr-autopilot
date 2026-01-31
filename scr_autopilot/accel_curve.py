from __future__ import annotations

import argparse
import csv
import re
import time
from pathlib import Path
from typing import Callable, Optional

from scr_autopilot.config import HudRois, OcrOpenCvConfig, OcrTesseractConfig, PixelRoi, load_config
from scr_autopilot.ocr_tools import clamp_roi, read_roi_text, read_speed_opencv
from scr_autopilot.vision import ScreenGrabber, WindowRegion, find_window_region, load_digit_templates


def parse_speed(text: str) -> Optional[float]:
    match = re.search(r"\d+(?:\.\d+)?", text)
    if not match:
        return None
    try:
        return float(match.group())
    except ValueError:
        return None


def parse_roi(value: Optional[str], default: PixelRoi) -> PixelRoi:
    if not value:
        return default
    x, y, w, h = (int(part) for part in value.split(","))
    return PixelRoi(x=x, y=y, width=w, height=h)


def build_region_provider(
    refresh_seconds: float,
    window_title: str,
    log: Callable[[str], None],
) -> Callable[[], WindowRegion]:
    last_region = find_window_region(window_title)
    last_error: Optional[str] = None
    last_refresh = 0.0

    def provider() -> WindowRegion:
        nonlocal last_region, last_error, last_refresh
        now = time.time()
        if now - last_refresh < refresh_seconds:
            return last_region
        last_refresh = now
        try:
            last_region = find_window_region(window_title)
            if last_error:
                log("Window capture recovered.")
            last_error = None
        except RuntimeError as exc:
            if str(exc) != last_error:
                log(f"Window capture failed: {exc}. Falling back to {last_region}.")
                last_error = str(exc)
        return last_region

    log(f"Attempting to capture window matching '{window_title}'.")
    return provider


def compute_slope(samples: list[tuple[float, float]]) -> Optional[float]:
    if len(samples) < 2:
        return None
    times = [t for t, _ in samples]
    speeds = [v for _, v in samples]
    t0 = times[0]
    times = [t - t0 for t in times]
    mean_t = sum(times) / len(times)
    mean_v = sum(speeds) / len(speeds)
    denom = sum((t - mean_t) ** 2 for t in times)
    if denom == 0:
        return None
    numer = sum((t - mean_t) * (v - mean_v) for t, v in zip(times, speeds))
    return numer / denom


def build_parser(config) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure acceleration curves from HUD speed OCR.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to the central scr_autopilot.toml config file.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=config.accel_curve.interval,
        help="Seconds between OCR samples.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Optional duration (seconds) to record before exiting.",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=config.accel_curve.smooth_window,
        help="Sample window size for smoothing acceleration.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=config.accel_curve.output_csv,
        help="CSV file to write timestamp, speed, and acceleration.",
    )
    parser.add_argument(
        "--speed-roi",
        help="Absolute ROI for speed as 'x,y,width,height' relative to capture region.",
    )
    parser.add_argument(
        "--speed-ocr",
        choices=("tesseract", "opencv"),
        default=config.ocr.speed_backend,
        help="OCR backend for the speed readout.",
    )
    parser.add_argument(
        "--speed-template-dir",
        type=Path,
        default=config.ocr.opencv.speed_template_dir,
        help="Folder containing digit templates for OpenCV OCR (png per digit).",
    )
    parser.add_argument(
        "--window-title",
        default=config.window.title,
        help="Window title/owner hint to match.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Enable debug logging.",
    )
    return parser


def main() -> None:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=Path, default=None)
    pre_args, _ = pre_parser.parse_known_args()
    config = load_config(pre_args.config)
    parser = build_parser(config)
    args = parser.parse_args()

    rois = HudRois(
        speed=parse_roi(args.speed_roi, config.rois.speed),
        limit=config.rois.limit,
        distance=config.rois.distance,
    )

    tesseract_settings = OcrTesseractConfig(
        threshold=config.ocr.tesseract.threshold,
        invert=config.ocr.tesseract.invert,
        scale=config.ocr.tesseract.scale,
        whitelist=config.ocr.tesseract.whitelist,
        lang=config.ocr.tesseract.lang,
        psm=config.ocr.tesseract.psm,
        oem=config.ocr.tesseract.oem,
    )
    opencv_settings = OcrOpenCvConfig(
        threshold=config.ocr.opencv.threshold,
        invert=config.ocr.opencv.invert,
        min_area=config.ocr.opencv.min_area,
        min_height=config.ocr.opencv.min_height,
        speed_template_dir=args.speed_template_dir,
    )

    log = lambda message: print(f"[debug] {message}") if args.debug else None
    speed_templates: dict[str, np.ndarray] = {}
    if args.speed_ocr == "opencv":
        speed_templates = load_digit_templates(args.speed_template_dir)
        if not speed_templates:
            log("No OpenCV templates found. Falling back to Tesseract.")
            args.speed_ocr = "tesseract"

    region_provider = build_region_provider(config.window.refresh_seconds, args.window_title, log)
    grabber = ScreenGrabber(region_provider)
    grabber.start()

    samples: list[tuple[float, float]] = []
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    start_time = time.time()
    with args.output_csv.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp", "speed", "acceleration"])
        try:
            while True:
                sample = grabber.read_latest()
                if sample is None:
                    time.sleep(0.01)
                    continue
                frame = sample.frame
                speed_roi = clamp_roi(frame, rois.speed)
                if speed_roi is None:
                    time.sleep(args.interval)
                    continue
                if args.speed_ocr == "opencv":
                    speed_text, _ = read_speed_opencv(speed_roi, speed_templates, opencv_settings)
                else:
                    speed_text = read_roi_text(speed_roi, tesseract_settings)
                speed_value = parse_speed(speed_text)
                if speed_value is None:
                    time.sleep(args.interval)
                    continue
                timestamp = time.time()
                samples.append((timestamp, speed_value))
                window = samples[-args.smooth_window :]
                acceleration = compute_slope(window)
                writer.writerow([timestamp, speed_value, acceleration if acceleration is not None else ""])
                accel_display = f"{acceleration:.3f}" if acceleration is not None else "--"
                print(
                    f"\rSpeed: {speed_value:.1f} | Accel: {accel_display} units/s^2",
                    end="",
                    flush=True,
                )
                if args.duration and timestamp - start_time >= args.duration:
                    print("\nDuration reached. Stopping capture.")
                    break
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\nStopping acceleration capture.")
        finally:
            grabber.stop()


if __name__ == "__main__":
    main()
