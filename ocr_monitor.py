"""Continuously monitor speed and distance using full-screen OCR."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Tuple

import mss
import pytesseract
from PIL import Image, ImageOps

import ocr_config


@dataclass(frozen=True)
class Regions:
    speed: Tuple[int, int, int, int]
    miles: Tuple[int, int, int, int]


def _apply_ocr_settings() -> None:
    if ocr_config.TESSERACT_CMD:
        pytesseract.pytesseract.tesseract_cmd = ocr_config.TESSERACT_CMD


def _crop(image: Image.Image, region: Tuple[int, int, int, int]) -> Image.Image:
    left, top, right, bottom = region
    return image.crop((left, top, right, bottom))


def _preprocess(image: Image.Image) -> Image.Image:
    grayscale = ImageOps.grayscale(image)
    return grayscale.point(lambda p: 255 if p > ocr_config.THRESHOLD else 0)


def _read_digits(image: Image.Image) -> str:
    config = (
        f"--psm {ocr_config.TESSERACT_PSM} "
        f"-c tessedit_char_whitelist={ocr_config.TESSERACT_WHITELIST}"
    )
    text = pytesseract.image_to_string(image, config=config)
    return "".join(ch for ch in text if ch.isdigit())


def _grab_full_screen(sct: mss.mss) -> Image.Image:
    monitor = sct.monitors[1]
    shot = sct.grab(monitor)
    return Image.frombytes("RGB", (shot.width, shot.height), shot.rgb)


def _print_status(speed: str, miles: str, fps: float) -> None:
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] speed={speed or '-'} miles={miles or '-'} fps={fps:.1f}", flush=True)


def main() -> None:
    _apply_ocr_settings()
    regions = Regions(speed=ocr_config.REGION_SPEED, miles=ocr_config.REGION_MILES)

    target_frame_time = 1.0 / max(1, ocr_config.TARGET_FPS)
    last_speed = None
    last_miles = None
    last_status = 0.0
    frame_count = 0
    fps_timer = time.perf_counter()
    fps = 0.0

    with mss.mss() as sct:
        while True:
            loop_start = time.perf_counter()
            full = _grab_full_screen(sct)

            speed_img = _preprocess(_crop(full, regions.speed))
            miles_img = _preprocess(_crop(full, regions.miles))

            speed = _read_digits(speed_img)
            miles = _read_digits(miles_img)

            now = time.perf_counter()
            frame_count += 1
            elapsed = now - fps_timer
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                fps_timer = now
                frame_count = 0

            changed = speed != last_speed or miles != last_miles
            time_since_status = now - last_status
            if (not ocr_config.PRINT_ON_CHANGE_ONLY) or changed or time_since_status >= ocr_config.STATUS_EVERY_SECONDS:
                _print_status(speed, miles, fps)
                last_status = now
                last_speed = speed
                last_miles = miles

            loop_end = time.perf_counter()
            sleep_for = target_frame_time - (loop_end - loop_start)
            if sleep_for > 0:
                time.sleep(sleep_for)


if __name__ == "__main__":
    main()
