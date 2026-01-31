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


@dataclass
class CorrectionState:
    last_speed_value: int | None = None
    last_miles_value: int | None = None
    stop_start: float | None = None
    single_digit_streak: int = 0
    speed_reject_streak: int = 0
    miles_reject_streak: int = 0


@dataclass
class StatusState:
    last_speed_text: str | None = None
    last_miles_text: str | None = None
    last_status: float = 0.0
    last_update: float = 0.0
    fps_timer: float = 0.0
    frame_count: int = 0
    fps: float = 0.0

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


def _read_regions(full: Image.Image, regions: Regions) -> tuple[str, str]:
    speed_img = _preprocess(_crop(full, regions.speed))
    miles_img = _preprocess(_crop(full, regions.miles))
    raw_speed = _read_digits(speed_img)
    raw_miles = _read_digits(miles_img)
    return raw_speed, raw_miles


def _update_stop_timer(state: CorrectionState, speed_value: int | None, now: float) -> None:
    if _is_stopped(speed_value):
        if state.stop_start is None:
            state.stop_start = now
    else:
        state.stop_start = None


def _apply_speed_correction(
    raw_speed: str,
    state: CorrectionState,
    delta_t: float,
) -> int | None:
    speed_value = _parse_int(raw_speed)
    if speed_value is None:
        return None

    if state.last_speed_value is not None:
        max_delta = ocr_config.MAX_SPEED_DELTA_PER_SEC * delta_t
        if abs(speed_value - state.last_speed_value) > max_delta:
            speed_value = state.last_speed_value
            state.speed_reject_streak += 1
        else:
            state.speed_reject_streak = 0

    if (
        state.last_speed_value is not None
        and state.last_speed_value >= ocr_config.SINGLE_DIGIT_SPEED_IF_PREV_HIGH
    ):
        if speed_value < 10:
            state.single_digit_streak += 1
            if state.single_digit_streak < ocr_config.SINGLE_DIGIT_CONFIRM_FRAMES:
                speed_value = state.last_speed_value
                state.speed_reject_streak += 1
        else:
            state.single_digit_streak = 0
            state.speed_reject_streak = 0
    else:
        state.single_digit_streak = 0

    if state.speed_reject_streak >= ocr_config.MAX_SPEED_REJECT_FRAMES:
        state.speed_reject_streak = 0
        return _parse_int(raw_speed)

    return speed_value


def _apply_miles_correction(
    raw_miles: str,
    state: CorrectionState,
    delta_t: float,
    now: float,
) -> int | None:
    miles_digits = _clamp_distance_digits(raw_miles)
    miles_value = _parse_int(miles_digits)
    if miles_value is None:
        return None

    if state.last_miles_value is not None:
        max_drop = ocr_config.MAX_DISTANCE_DROP_PER_SEC * delta_t
        max_rise = ocr_config.MAX_DISTANCE_RISE_PER_SEC * delta_t
        delta = miles_value - state.last_miles_value
        allow_reset = (
            state.stop_start is not None
            and (now - state.stop_start) >= ocr_config.MIN_SPEED_STABLE_FOR_RESET_SEC
        )
        if delta > max_rise and not allow_reset:
            miles_value = state.last_miles_value
            state.miles_reject_streak += 1
        elif delta < -max_drop and not allow_reset:
            miles_value = state.last_miles_value
            state.miles_reject_streak += 1
        else:
            state.miles_reject_streak = 0

    if state.miles_reject_streak >= ocr_config.MAX_DISTANCE_REJECT_FRAMES:
        state.miles_reject_streak = 0
        return _parse_int(miles_digits)

    return miles_value


def _format_output(speed_value: int | None, miles_value: int | None) -> tuple[str, str]:
    speed = f"{speed_value}" if speed_value is not None else ""
    miles = f"{miles_value:0{ocr_config.MAX_DISTANCE_DIGITS}d}" if miles_value is not None else ""
    return speed, miles


def _update_fps(state: StatusState, now: float) -> None:
    state.frame_count += 1
    elapsed = now - state.fps_timer
    if elapsed >= 1.0:
        state.fps = state.frame_count / elapsed
        state.fps_timer = now
        state.frame_count = 0


def _should_print(
    state: StatusState,
    speed: str,
    miles: str,
    now: float,
) -> bool:
    changed = speed != state.last_speed_text or miles != state.last_miles_text
    time_since_status = now - state.last_status
    return (
        (not ocr_config.PRINT_ON_CHANGE_ONLY)
        or changed
        or time_since_status >= ocr_config.STATUS_EVERY_SECONDS
    )

def main() -> None:
    _apply_ocr_settings()
    regions = Regions(speed=ocr_config.REGION_SPEED, miles=ocr_config.REGION_MILES)

    target_frame_time = 1.0 / max(1, ocr_config.TARGET_FPS)
    correction_state = CorrectionState()
    status_state = StatusState(last_update=time.perf_counter(), fps_timer=time.perf_counter())

    with mss.mss() as sct:
        while True:
            loop_start = time.perf_counter()
            full = _grab_full_screen(sct)

            raw_speed, raw_miles = _read_regions(full, regions)

            now = time.perf_counter()
            delta_t = max(0.001, now - status_state.last_update)
            status_state.last_update = now

            speed_value = _apply_speed_correction(raw_speed, correction_state, delta_t)
            _update_stop_timer(correction_state, speed_value, now)
            miles_value = _apply_miles_correction(raw_miles, correction_state, delta_t, now)

            if speed_value is not None:
                correction_state.last_speed_value = speed_value
            if miles_value is not None:
                correction_state.last_miles_value = miles_value

            speed, miles = _format_output(speed_value, miles_value)
            _update_fps(status_state, now)

            if _should_print(status_state, speed, miles, now):
                _print_status(speed, miles, status_state.fps)
                status_state.last_status = now
                status_state.last_speed_text = speed
                status_state.last_miles_text = miles

            loop_end = time.perf_counter()
            sleep_for = target_frame_time - (loop_end - loop_start)
            if sleep_for > 0:
                time.sleep(sleep_for)


if __name__ == "__main__":
    main()
