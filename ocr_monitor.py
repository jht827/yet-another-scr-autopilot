"""Continuously monitor speed and distance using full-screen OCR."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Tuple

import mss
import pytesseract
from PIL import Image, ImageChops, ImageOps

import ocr_config
from ocr_corrections import DistanceState, SpeedState, apply_distance_correction, apply_speed_correction


@dataclass(frozen=True)
class Regions:
    speed: Tuple[int, int, int, int]
    miles: Tuple[int, int, int, int]


@dataclass
class CorrectionState:
    speed: SpeedState
    distance: DistanceState
    stop_start: float | None = None


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
    # Allow overriding the Tesseract binary path for environments without a default install.
    if ocr_config.TESSERACT_CMD:
        pytesseract.pytesseract.tesseract_cmd = ocr_config.TESSERACT_CMD


def _crop(image: Image.Image, region: Tuple[int, int, int, int]) -> Image.Image:
    # Region tuple is (left, top, right, bottom) in screen coordinates.
    left, top, right, bottom = region
    return image.crop((left, top, right, bottom))


def _red_mask(image: Image.Image) -> Image.Image:
    # Isolate red pixels to help OCR for red HUD digits.
    red, green, blue = image.split()
    non_red = ImageChops.lighter(green, blue)
    red_only = ImageChops.subtract(red, non_red)
    return red_only.point(lambda p: 255 if p > ocr_config.RED_THRESHOLD else 0)


def _preprocess(image: Image.Image) -> Image.Image:
    # Threshold to a high-contrast image and optionally blend in the red channel mask.
    grayscale = ImageOps.grayscale(image)
    base = grayscale.point(lambda p: 255 if p > ocr_config.THRESHOLD else 0)
    if ocr_config.USE_RED_DETECTION:
        red = _red_mask(image)
        return ImageChops.lighter(base, red)
    return base


def _read_digits(image: Image.Image) -> str:
    # Restrict OCR to numeric characters for faster, cleaner results.
    config = (
        f"--psm {ocr_config.TESSERACT_PSM} "
        f"-c tessedit_char_whitelist={ocr_config.TESSERACT_WHITELIST}"
    )
    text = pytesseract.image_to_string(image, config=config)
    return "".join(ch for ch in text if ch.isdigit())


def _is_stopped(speed_value: int | None) -> bool:
    # Treat very low speeds as a stop for distance reset logic.
    return speed_value is not None and speed_value <= ocr_config.STOP_SPEED_THRESHOLD


def _grab_full_screen(sct: mss.mss) -> Image.Image:
    # Capture the primary monitor as an RGB image for OCR processing.
    monitor = sct.monitors[1]
    shot = sct.grab(monitor)
    return Image.frombytes("RGB", (shot.width, shot.height), shot.rgb)


def _print_status(speed: str, miles: str, fps: float) -> None:
    # Emit a compact status line with OCR values and current FPS.
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] speed={speed or '-'} miles={miles or '-'} fps={fps:.1f}", flush=True)


def _read_regions(full: Image.Image, regions: Regions) -> tuple[str, str]:
    # Crop and preprocess the speed/mileage regions before OCR.
    speed_img = _preprocess(_crop(full, regions.speed))
    miles_img = _preprocess(_crop(full, regions.miles))
    raw_speed = _read_digits(speed_img)
    raw_miles = _read_digits(miles_img)
    return raw_speed, raw_miles


def _update_stop_timer(state: CorrectionState, speed_value: int | None, now: float) -> None:
    # Track how long we've been stopped to allow a distance reset.
    if _is_stopped(speed_value):
        if state.stop_start is None:
            state.stop_start = now
    else:
        state.stop_start = None


def _format_output(speed_value: int | None, miles_value: int | None) -> tuple[str, str]:
    # Format output with fixed-width mileage and blank placeholders.
    speed = f"{speed_value}" if speed_value is not None else ""
    miles = f"{miles_value:0{ocr_config.MAX_DISTANCE_DIGITS}d}" if miles_value is not None else ""
    return speed, miles


def _update_fps(state: StatusState, now: float) -> None:
    # Update FPS once per second using a rolling frame counter.
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
    # Decide whether to print based on change detection or time interval.
    changed = speed != state.last_speed_text or miles != state.last_miles_text
    time_since_status = now - state.last_status
    return (
        (not ocr_config.PRINT_ON_CHANGE_ONLY)
        or changed
        or time_since_status >= ocr_config.STATUS_EVERY_SECONDS
    )


def _allow_distance_reset(state: CorrectionState, now: float) -> bool:
    # Permit distance reset only after a stable stopped interval.
    return state.stop_start is not None and (
        now - state.stop_start
    ) >= ocr_config.MIN_SPEED_STABLE_FOR_RESET_SEC


def main() -> None:
    _apply_ocr_settings()
    regions = Regions(speed=ocr_config.REGION_SPEED, miles=ocr_config.REGION_MILES)

    target_frame_time = 1.0 / max(1, ocr_config.TARGET_FPS)
    correction_state = CorrectionState(speed=SpeedState(), distance=DistanceState())
    status_state = StatusState(last_update=time.perf_counter(), fps_timer=time.perf_counter())

    with mss.mss() as sct:
        while True:
            loop_start = time.perf_counter()
            full = _grab_full_screen(sct)

            raw_speed, raw_miles = _read_regions(full, regions)

            now = time.perf_counter()
            delta_t = max(0.001, now - status_state.last_update)
            status_state.last_update = now

            speed_value = apply_speed_correction(raw_speed, correction_state.speed, delta_t)
            _update_stop_timer(correction_state, speed_value, now)
            allow_reset = _allow_distance_reset(correction_state, now)
            miles_value = apply_distance_correction(
                raw_miles,
                correction_state.distance,
                speed_value,
                delta_t,
                allow_reset,
            )

            if speed_value is not None:
                correction_state.speed.last_speed_value = speed_value
            if miles_value is not None:
                correction_state.distance.last_miles_value = miles_value

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
