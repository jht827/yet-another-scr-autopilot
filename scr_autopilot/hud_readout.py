import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pytesseract
from PIL import Image, ImageOps

from scr_autopilot.hud_config import HUD_ROIS, REFERENCE_SIZE, WINDOW_TITLE, HudRois, PixelRoi
from scr_autopilot.vision import ScreenGrabber, WindowRegion, find_window_region


def resolve_region() -> WindowRegion:
    return find_window_region(WINDOW_TITLE)


def debug_log(enabled: bool, message: str) -> None:
    if enabled:
        print(f"[debug] {message}")


def build_region_provider(
    args: argparse.Namespace,
    log: Callable[[str], None],
) -> Callable[[], WindowRegion]:
    last_region = resolve_region()
    last_error: Optional[str] = None
    last_refresh = 0.0

    def provider() -> WindowRegion:
        nonlocal last_region, last_error, last_refresh
        now = time.time()
        if now - last_refresh < args.window_refresh:
            return last_region
        last_refresh = now
        try:
            last_region = find_window_region(WINDOW_TITLE)
            if last_error:
                log("Window capture recovered.")
            last_error = None
        except RuntimeError as exc:
            if str(exc) != last_error:
                log(f"Window capture failed: {exc}. Falling back to {last_region}.")
                last_error = str(exc)
        return last_region

    log(f"Attempting to capture window matching '{WINDOW_TITLE}'.")
    return provider


def clamp_roi(frame: np.ndarray, roi: PixelRoi) -> Optional[np.ndarray]:
    x0 = max(roi.x, 0)
    y0 = max(roi.y, 0)
    x1 = min(roi.x + roi.width, frame.shape[1])
    y1 = min(roi.y + roi.height, frame.shape[0])
    if x1 <= x0 or y1 <= y0:
        return None
    return frame[y0:y1, x0:x1]


def parse_roi(value: Optional[str], default: PixelRoi) -> PixelRoi:
    if not value:
        return default
    x, y, w, h = (int(part) for part in value.split(","))
    return PixelRoi(x=x, y=y, width=w, height=h)


def to_pil(frame: np.ndarray) -> Image.Image:
    return Image.fromarray(frame[:, :, ::-1])


def preprocess_for_tesseract(
    roi: np.ndarray,
    threshold: Optional[int],
    invert: bool,
    scale: float,
) -> Image.Image:
    image = to_pil(roi)
    if scale != 1.0:
        width = max(1, int(image.width * scale))
        height = max(1, int(image.height * scale))
        image = image.resize((width, height), Image.BILINEAR)
    gray = ImageOps.autocontrast(image.convert("L"))
    if threshold is not None:
        gray = gray.point(lambda p: 255 if p > threshold else 0)
    if invert:
        gray = ImageOps.invert(gray)
    return gray


def tesseract_config(args: argparse.Namespace) -> str:
    return (
        f"--psm {args.tesseract_psm} --oem {args.tesseract_oem} "
        f"-c tessedit_char_whitelist={args.whitelist}"
    )


def read_roi_text(roi: np.ndarray, args: argparse.Namespace) -> str:
    processed = preprocess_for_tesseract(roi, args.threshold, args.invert, args.scale)
    text = pytesseract.image_to_string(processed, config=tesseract_config(args), lang=args.lang)
    cleaned = "".join(char for char in text.strip().lower() if char.isalnum())
    return cleaned


def maybe_dump_debug(
    debug_dir: Optional[Path],
    frame: np.ndarray,
    rois: dict[str, Optional[np.ndarray]],
    args: argparse.Namespace,
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
        processed = preprocess_for_tesseract(roi, args.threshold, args.invert, args.scale)
        processed.save(debug_dir / f"{name}_ocr.png")
    return True


def format_value(value: str) -> str:
    return value if value else "--"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read HUD values from fixed pixel ROIs using Tesseract OCR.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.05,
        help="Seconds between terminal updates.",
    )
    parser.add_argument(
        "--speed-roi",
        help=(
            "Absolute ROI for speed as 'x,y,width,height' relative to capture region. "
            f"Default is tuned for {REFERENCE_SIZE[0]}x{REFERENCE_SIZE[1]}."
        ),
    )
    parser.add_argument(
        "--limit-roi",
        help=(
            "Absolute ROI for speed limit as 'x,y,width,height' relative to capture region. "
            f"Default is tuned for {REFERENCE_SIZE[0]}x{REFERENCE_SIZE[1]}."
        ),
    )
    parser.add_argument(
        "--distance-roi",
        help=(
            "Absolute ROI for next-station distance as 'x,y,width,height' relative to capture region. "
            f"Default is tuned for {REFERENCE_SIZE[0]}x{REFERENCE_SIZE[1]}."
        ),
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=160,
        help="Binarization threshold (0-255). Set to -1 to disable thresholding.",
    )
    parser.add_argument(
        "--invert",
        action="store_true",
        help="Invert ROI colors before OCR.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=2.0,
        help="Scale factor before OCR (higher can improve accuracy).",
    )
    parser.add_argument(
        "--whitelist",
        default="0123456789abcdefghijklmnopqrstuvwxyz",
        help="Tesseract whitelist for allowed characters.",
    )
    parser.add_argument(
        "--lang",
        default="eng",
        help="Tesseract language to use (default: eng).",
    )
    parser.add_argument(
        "--tesseract-psm",
        type=int,
        default=7,
        help="Tesseract page segmentation mode (PSM).",
    )
    parser.add_argument(
        "--tesseract-oem",
        type=int,
        default=1,
        help="Tesseract OCR engine mode (OEM).",
    )
    parser.add_argument(
        "--window-refresh",
        type=float,
        default=1.0,
        help="Seconds between window region refreshes when tracking the Roblox window.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging about capture status and frame timing.",
    )
    parser.add_argument(
        "--debug-dir",
        type=Path,
        default=Path("hud_debug"),
        help="Folder to dump a full frame + ROI crops for debugging.",
    )
    args = parser.parse_args()

    threshold = None if args.threshold < 0 else args.threshold
    args.threshold = threshold

    rois = HudRois(
        speed=parse_roi(args.speed_roi, HUD_ROIS.speed),
        limit=parse_roi(args.limit_roi, HUD_ROIS.limit),
        distance=parse_roi(args.distance_roi, HUD_ROIS.distance),
    )

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
            speed_roi = clamp_roi(frame, rois.speed)
            limit_roi = clamp_roi(frame, rois.limit)
            distance_roi = clamp_roi(frame, rois.distance)
            debug_dumped = maybe_dump_debug(
                args.debug_dir,
                frame,
                {"speed_roi": speed_roi, "limit_roi": limit_roi, "distance_roi": distance_roi},
                args,
                debug_dumped,
            )

            speed_value = format_value(read_roi_text(speed_roi, args) if speed_roi is not None else "")
            limit_value = format_value(read_roi_text(limit_roi, args) if limit_roi is not None else "")
            distance_value = format_value(
                read_roi_text(distance_roi, args) if distance_roi is not None else ""
            )

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
