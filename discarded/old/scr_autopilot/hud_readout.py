import argparse
import time
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

from scr_autopilot.config import HudRois, OcrOpenCvConfig, OcrTesseractConfig, PixelRoi, load_config
from scr_autopilot.ocr_tools import (
    clamp_roi,
    preprocess_for_tesseract,
    read_roi_text,
    read_speed_opencv,
    to_pil,
)
from scr_autopilot.vision import (
    OcrDebug,
    ScreenGrabber,
    WindowRegion,
    find_window_region,
    load_digit_templates,
    summarize_matches,
)


def resolve_region(title: str) -> WindowRegion:
    return find_window_region(title)


def debug_log(enabled: bool, message: str) -> None:
    if enabled:
        print(f"[debug] {message}")


def build_region_provider(
    args: argparse.Namespace,
    log: Callable[[str], None],
) -> Callable[[], WindowRegion]:
    last_region = resolve_region(args.window_title)
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


def parse_roi(value: Optional[str], default: PixelRoi) -> PixelRoi:
    if not value:
        return default
    x, y, w, h = (int(part) for part in value.split(","))
    return PixelRoi(x=x, y=y, width=w, height=h)


def build_parser(config) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read HUD values from fixed pixel ROIs using Tesseract OCR.",
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
        default=config.readout.interval,
        help="Seconds between terminal updates.",
    )
    parser.add_argument(
        "--speed-roi",
        help=(
            "Absolute ROI for speed as 'x,y,width,height' relative to capture region. "
            f"Default is tuned for {config.window.reference_size[0]}x{config.window.reference_size[1]}."
        ),
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
        "--limit-roi",
        help=(
            "Absolute ROI for speed limit as 'x,y,width,height' relative to capture region. "
            f"Default is tuned for {config.window.reference_size[0]}x{config.window.reference_size[1]}."
        ),
    )
    parser.add_argument(
        "--distance-roi",
        help=(
            "Absolute ROI for next-station distance as 'x,y,width,height' relative to capture region. "
            f"Default is tuned for {config.window.reference_size[0]}x{config.window.reference_size[1]}."
        ),
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=config.ocr.tesseract.threshold if config.ocr.tesseract.threshold is not None else -1,
        help="Binarization threshold (0-255). Set to -1 to disable thresholding.",
    )
    parser.add_argument(
        "--invert",
        action="store_true",
        default=config.ocr.tesseract.invert,
        help="Invert ROI colors before OCR.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=config.ocr.tesseract.scale,
        help="Scale factor before OCR (higher can improve accuracy).",
    )
    parser.add_argument(
        "--opencv-threshold",
        type=int,
        default=config.ocr.opencv.threshold if config.ocr.opencv.threshold is not None else -1,
        help="Threshold for OpenCV template OCR (0-255). Set to -1 for adaptive thresholding.",
    )
    parser.add_argument(
        "--opencv-invert",
        action="store_true",
        default=config.ocr.opencv.invert,
        help="Invert OpenCV OCR thresholding (useful for bright text).",
    )
    parser.add_argument(
        "--opencv-min-area",
        type=int,
        default=config.ocr.opencv.min_area,
        help="Minimum contour area for digit segmentation in OpenCV OCR.",
    )
    parser.add_argument(
        "--opencv-min-height",
        type=int,
        default=config.ocr.opencv.min_height,
        help="Minimum contour height for digit segmentation in OpenCV OCR.",
    )
    parser.add_argument(
        "--opencv-debug",
        action="store_true",
        default=config.readout.opencv_debug,
        help="Dump OpenCV OCR intermediate images and match summaries.",
    )
    parser.add_argument(
        "--whitelist",
        default=config.ocr.tesseract.whitelist,
        help="Tesseract whitelist for allowed characters.",
    )
    parser.add_argument(
        "--lang",
        default=config.ocr.tesseract.lang,
        help="Tesseract language to use (default: eng).",
    )
    parser.add_argument(
        "--tesseract-psm",
        type=int,
        default=config.ocr.tesseract.psm,
        help="Tesseract page segmentation mode (PSM).",
    )
    parser.add_argument(
        "--tesseract-oem",
        type=int,
        default=config.ocr.tesseract.oem,
        help="Tesseract OCR engine mode (OEM).",
    )
    parser.add_argument(
        "--window-title",
        default=config.window.title,
        help="Window title/owner hint to match.",
    )
    parser.add_argument(
        "--window-refresh",
        type=float,
        default=config.window.refresh_seconds,
        help="Seconds between window region refreshes when tracking the Roblox window.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=config.readout.debug,
        help="Enable debug logging about capture status and frame timing.",
    )
    parser.add_argument(
        "--debug-dir",
        type=Path,
        default=config.readout.debug_dir,
        help="Folder to dump a full frame + ROI crops for debugging.",
    )
    parser.add_argument(
        "--debug-interval",
        type=float,
        default=config.readout.debug_interval,
        help="Seconds between debug dumps (frame stats + OpenCV debug output).",
    )
    return parser


def maybe_dump_debug(
    debug_dir: Optional[Path],
    frame: np.ndarray,
    rois: dict[str, Optional[np.ndarray]],
    tesseract_settings: OcrTesseractConfig,
    dumped: bool,
) -> bool:
    if not debug_dir or dumped:
        return dumped
    debug_dir.mkdir(parents=True, exist_ok=True)
    to_pil(frame).save(debug_dir / "frame.png")
    for name, roi in rois.items():
        if roi is None:
            continue
        to_pil(roi).save(debug_dir / f"{name}.png")
        processed = preprocess_for_tesseract(roi, tesseract_settings)
        processed.save(debug_dir / f"{name}_ocr.png")
    return True


def build_opencv_debug_frame(roi: np.ndarray, debug: OcrDebug) -> np.ndarray:
    overlay = cv2.cvtColor(debug.thresholded, cv2.COLOR_GRAY2BGR)
    for x, y, w, h in debug.contour_boxes:
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 1)
    return overlay


def dump_opencv_debug(
    debug_dir: Path,
    roi: np.ndarray,
    debug: OcrDebug,
    text: str,
) -> None:
    debug_dir.mkdir(parents=True, exist_ok=True)
    to_pil(roi).save(debug_dir / "speed_roi.png")
    overlay = build_opencv_debug_frame(roi, debug)
    cv2.imwrite(str(debug_dir / "speed_thresholded.png"), debug.thresholded)
    cv2.imwrite(str(debug_dir / "speed_contours.png"), overlay)
    for index, match in enumerate(debug.matches):
        x, y, w, h = match.bounds
        digit = debug.thresholded[y : y + h, x : x + w]
        cv2.imwrite(str(debug_dir / f"digit_{index}_{match.label}.png"), digit)
    (debug_dir / "speed_matches.txt").write_text(
        f"text={text}\nmatches={summarize_matches(debug.matches)}\n"
    )


def format_value(value: str) -> str:
    return value if value else "--"


def main() -> None:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=Path, default=None)
    pre_args, _ = pre_parser.parse_known_args()
    config = load_config(pre_args.config)
    parser = build_parser(config)
    args = parser.parse_args()

    threshold = None if args.threshold < 0 else args.threshold
    opencv_threshold = None if args.opencv_threshold < 0 else args.opencv_threshold

    rois = HudRois(
        speed=parse_roi(args.speed_roi, config.rois.speed),
        limit=parse_roi(args.limit_roi, config.rois.limit),
        distance=parse_roi(args.distance_roi, config.rois.distance),
    )

    tesseract_settings = OcrTesseractConfig(
        threshold=threshold,
        invert=args.invert,
        scale=args.scale,
        whitelist=args.whitelist,
        lang=args.lang,
        psm=args.tesseract_psm,
        oem=args.tesseract_oem,
    )
    opencv_settings = OcrOpenCvConfig(
        threshold=opencv_threshold,
        invert=args.opencv_invert,
        min_area=args.opencv_min_area,
        min_height=args.opencv_min_height,
        speed_template_dir=args.speed_template_dir,
    )

    log = lambda message: debug_log(args.debug, message)
    speed_templates: dict[str, np.ndarray] = {}
    if args.speed_ocr == "opencv":
        speed_templates = load_digit_templates(args.speed_template_dir)
        if not speed_templates:
            log(
                f"No OpenCV templates found in {args.speed_template_dir}. Falling back to Tesseract."
            )
            args.speed_ocr = "tesseract"
    region_provider = build_region_provider(args, log)
    grabber = ScreenGrabber(region_provider)
    grabber.start()
    debug_dumped = False
    last_debug = 0.0
    last_opencv_debug = 0.0
    last_sample_time: Optional[float] = None
    try:
        while True:
            sample = grabber.read_latest()
            if sample is None:
                if args.debug and time.time() - last_debug > args.debug_interval:
                    log("Waiting for first frame...")
                    last_debug = time.time()
                time.sleep(0.01)
                continue
            frame = sample.frame
            region = region_provider()
            speed_roi = clamp_roi(frame, rois.speed)
            limit_roi = clamp_roi(frame, rois.limit)
            distance_roi = clamp_roi(frame, rois.distance)
            debug_dumped = maybe_dump_debug(
                args.debug_dir,
                frame,
                {"speed_roi": speed_roi, "limit_roi": limit_roi, "distance_roi": distance_roi},
                tesseract_settings,
                debug_dumped,
            )

            speed_debug: Optional[OcrDebug] = None
            if speed_roi is not None and args.speed_ocr == "opencv":
                speed_text, speed_debug = read_speed_opencv(
                    speed_roi, speed_templates, opencv_settings
                )
                speed_value = format_value(speed_text)
            else:
                speed_value = format_value(
                    read_roi_text(speed_roi, tesseract_settings) if speed_roi is not None else ""
                )
            limit_value = format_value(
                read_roi_text(limit_roi, tesseract_settings) if limit_roi is not None else ""
            )
            distance_value = format_value(
                read_roi_text(distance_roi, tesseract_settings) if distance_roi is not None else ""
            )

            line = (
                f"Speed: {speed_value} | Limit: {limit_value} | "
                f"Next Station: {distance_value}"
            )
            print(f"\r{line:<80}", end="", flush=True)
            now = time.time()
            if args.opencv_debug and speed_debug is not None:
                if now - last_opencv_debug > args.debug_interval:
                    opencv_dir = args.debug_dir / f"opencv_speed_{int(now)}"
                    dump_opencv_debug(opencv_dir, speed_roi, speed_debug, speed_value)
                    log(f"Saved OpenCV debug output to {opencv_dir}.")
                    last_opencv_debug = now
            if args.debug and now - last_debug > args.debug_interval:
                frame_age = now - sample.timestamp
                fps = 0.0
                if last_sample_time is not None:
                    fps = 1.0 / max(now - last_sample_time, 1e-6)
                last_sample_time = now
                mean_luma = float(np.mean(frame[:, :, 0]))
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
