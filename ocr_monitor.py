"""Continuously monitor speed and distance using full-screen OCR."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Tuple

import mss
import pytesseract
from PIL import Image, ImageChops, ImageOps

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


def _red_mask(image: Image.Image) -> Image.Image:
    red, green, blue = image.split()
    non_red = ImageChops.lighter(green, blue)
    red_only = ImageChops.subtract(red, non_red)
    return red_only.point(lambda p: 255 if p > ocr_config.RED_THRESHOLD else 0)


def _preprocess(image: Image.Image) -> Image.Image:
    grayscale = ImageOps.grayscale(image)
    base = grayscale.point(lambda p: 255 if p > ocr_config.THRESHOLD else 0)
    if ocr_config.USE_RED_DETECTION:
        red = _red_mask(image)
        return ImageChops.lighter(base, red)
    return base


def _read_digits(image: Image.Image) -> str:
    config = (
        f"--psm {ocr_config.TESSERACT_PSM} "
        f"-c tessedit_char_whitelist={ocr_config.TESSERACT_WHITELIST}"
    )
    text = pytesseract.image_to_string(image, config=config)
    return "".join(ch for ch in text if ch.isdigit())


def _parse_int(text: str) -> int | None:
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def _digits_only(text: str) -> str:
    return "".join(ch for ch in text if ch.isdigit())


def _clamp_distance_digits(text: str) -> str:
    digits = _digits_only(text)
    if len(digits) <= ocr_config.MAX_DISTANCE_DIGITS:
        return digits
    return digits[: ocr_config.MAX_DISTANCE_DIGITS]


def _is_stopped(speed_value: int | None) -> bool:
    return speed_value is not None and speed_value <= ocr_config.STOP_SPEED_THRESHOLD


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
    last_speed_text = None
    last_miles_text = None
    last_speed_value: int | None = None
    last_miles_value: int | None = None
    last_status = 0.0
    last_update = time.perf_counter()
    stop_start: float | None = None
    single_digit_streak = 0
    frame_count = 0
    fps_timer = time.perf_counter()
    fps = 0.0

    with mss.mss() as sct:
        while True:
            loop_start = time.perf_counter()
            full = _grab_full_screen(sct)

            speed_img = _preprocess(_crop(full, regions.speed))
            miles_img = _preprocess(_crop(full, regions.miles))

            raw_speed = _read_digits(speed_img)
            raw_miles = _read_digits(miles_img)
            miles_digits = _clamp_distance_digits(raw_miles)

            speed_value = _parse_int(raw_speed)
            miles_value = _parse_int(miles_digits)

            now = time.perf_counter()
            delta_t = max(0.001, now - last_update)
            last_update = now

            if _is_stopped(speed_value):
                if stop_start is None:
                    stop_start = now
            else:
                stop_start = None

            if speed_value is not None:
                if last_speed_value is not None:
                    max_delta = ocr_config.MAX_SPEED_DELTA_PER_SEC * delta_t
                    if abs(speed_value - last_speed_value) > max_delta:
                        speed_value = last_speed_value
                if last_speed_value is not None and last_speed_value >= ocr_config.SINGLE_DIGIT_SPEED_IF_PREV_HIGH:
                    if speed_value < 10:
                        single_digit_streak += 1
                        if single_digit_streak < ocr_config.SINGLE_DIGIT_CONFIRM_FRAMES:
                            speed_value = last_speed_value
                    else:
                        single_digit_streak = 0
                else:
                    single_digit_streak = 0

            if miles_value is not None:
                if last_miles_value is not None:
                    max_drop = ocr_config.MAX_DISTANCE_DROP_PER_SEC * delta_t
                    max_rise = ocr_config.MAX_DISTANCE_RISE_PER_SEC * delta_t
                    delta = miles_value - last_miles_value
                    allow_reset = (
                        stop_start is not None
                        and (now - stop_start) >= ocr_config.MIN_SPEED_STABLE_FOR_RESET_SEC
                    )
                    if delta > max_rise and not allow_reset:
                        miles_value = last_miles_value
                    if delta < -max_drop and not allow_reset:
                        miles_value = last_miles_value

            speed = f"{speed_value}" if speed_value is not None else ""
            miles = f"{miles_value:0{ocr_config.MAX_DISTANCE_DIGITS}d}" if miles_value is not None else ""
            if speed_value is not None:
                last_speed_value = speed_value
            if miles_value is not None:
                last_miles_value = miles_value
            frame_count += 1
            elapsed = now - fps_timer
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                fps_timer = now
                frame_count = 0

            changed = speed != last_speed_text or miles != last_miles_text
            time_since_status = now - last_status
            if (not ocr_config.PRINT_ON_CHANGE_ONLY) or changed or time_since_status >= ocr_config.STATUS_EVERY_SECONDS:
                _print_status(speed, miles, fps)
                last_status = now
                last_speed_text = speed
                last_miles_text = miles

            loop_end = time.perf_counter()
            sleep_for = target_frame_time - (loop_end - loop_start)
            if sleep_for > 0:
                time.sleep(sleep_for)


if __name__ == "__main__":
    main()
