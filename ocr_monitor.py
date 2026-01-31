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
    next_signal_number: Tuple[int, int, int, int]
    next_signal_distance: Tuple[int, int, int, int]
    platform_number: Tuple[int, int, int, int]
    route_number: Tuple[int, int, int, int]
    next_station: Tuple[int, int, int, int]


@dataclass
class CorrectionState:
    speed: SpeedState
    distance: DistanceState
    next_signal_distance: DistanceState
    stop_start: float | None = None


@dataclass
class StatusState:
    last_speed_text: str | None = None
    last_miles_text: str | None = None
    last_next_signal_number_text: str | None = None
    last_next_signal_distance_text: str | None = None
    last_platform_number_text: str | None = None
    last_route_number_text: str | None = None
    last_next_station_text: str | None = None
    last_status: float = 0.0
    last_update: float = 0.0
    fps_timer: float = 0.0
    frame_count: int = 0
    fps: float = 0.0
    last_slow_read: float = 0.0


def _apply_ocr_settings() -> None:
    # Allow overriding the Tesseract binary path for environments without a default install.
    if ocr_config.TESSERACT_CMD:
        pytesseract.pytesseract.tesseract_cmd = ocr_config.TESSERACT_CMD


def _crop(image: Image.Image, region: Tuple[int, int, int, int]) -> Image.Image:
    # Region tuple is (left, top, right, bottom) in screen coordinates.
    # Keeping this tiny helper avoids repeating tuple unpacking in the main loop.
    left, top, right, bottom = region
    return image.crop((left, top, right, bottom))


def _red_mask(image: Image.Image) -> Image.Image:
    # Isolate red pixels to help OCR for red HUD digits.
    # The subtraction removes non-red content so only strong red survives.
    red, green, blue = image.split()
    non_red = ImageChops.lighter(green, blue)
    red_only = ImageChops.subtract(red, non_red)
    return red_only.point(lambda p: 255 if p > ocr_config.RED_THRESHOLD else 0)


def _preprocess(image: Image.Image) -> Image.Image:
    # Threshold to a high-contrast image and optionally blend in the red channel mask.
    # This keeps background noise low while preserving digits for OCR.
    grayscale = ImageOps.grayscale(image)
    base = grayscale.point(lambda p: 255 if p > ocr_config.THRESHOLD else 0)
    if ocr_config.USE_RED_DETECTION:
        red = _red_mask(image)
        return ImageChops.lighter(base, red)
    return base


def _read_digits(image: Image.Image) -> str:
    # Restrict OCR to numeric characters for faster, cleaner results.
    # The whitelist prevents stray punctuation or letters from being returned.
    config = (
        f"--psm {ocr_config.TESSERACT_PSM} "
        f"-c tessedit_char_whitelist={ocr_config.TESSERACT_WHITELIST}"
    )
    text = pytesseract.image_to_string(image, config=config)
    return "".join(ch for ch in text if ch.isdigit())


def _read_text(
    image: Image.Image,
    whitelist: str | None,
    allow_spaces: bool,
    force_upper: bool = True,
) -> str:
    config = f"--psm {ocr_config.TESSERACT_PSM}"
    if whitelist:
        config = f"{config} -c tessedit_char_whitelist={whitelist}"
    text = pytesseract.image_to_string(image, config=config)
    if force_upper:
        text = text.upper()
    cleaned = " ".join(text.split())
    if not whitelist and not allow_spaces:
        return cleaned
    allowed = set(whitelist or "")
    if allow_spaces:
        allowed.add(" ")
    return "".join(ch for ch in cleaned if ch in allowed)


def _normalize_platform_number(raw_platform_number: str) -> str:
    if raw_platform_number == "41":
        return "1"
    return raw_platform_number


def _is_stopped(speed_value: int | None) -> bool:
    # Treat very low speeds as a stop for distance reset logic.
    # This threshold smooths small OCR jitters when the vehicle is idle.
    return speed_value is not None and speed_value <= ocr_config.STOP_SPEED_THRESHOLD


def _grab_full_screen(sct: mss.mss) -> Image.Image:
    # Capture the primary monitor as an RGB image for OCR processing.
    # The OCR regions are then cropped from this full-frame capture.
    monitor = sct.monitors[1]
    shot = sct.grab(monitor)
    return Image.frombytes("RGB", (shot.width, shot.height), shot.rgb)


def _print_status(
    speed: str,
    miles: str,
    next_signal_number: str,
    next_signal_distance: str,
    platform_number: str,
    route_number: str,
    next_station: str,
    fps: float,
) -> None:
    # Emit a compact status line with OCR values and current FPS.
    # Use flush=True so log consumers see updates immediately.
    timestamp = time.strftime("%H:%M:%S")
    print(
        f"[{timestamp}] speed={speed or '-'} miles={miles or '-'} "
        f"signal={next_signal_number or '-'} signal_dist={next_signal_distance or '-'} "
        f"platform={platform_number or '-'} route={route_number or '-'} "
        f"next_station={next_station or '-'} fps={fps:.1f}",
        flush=True,
    )


def _read_regions(full: Image.Image, regions: Regions) -> tuple[str, str]:
    # Crop and preprocess the speed/mileage regions before OCR.
    # Returning raw digits keeps corrections centralized in ocr_corrections.
    speed_img = _preprocess(_crop(full, regions.speed))
    miles_img = _preprocess(_crop(full, regions.miles))
    raw_speed = _read_digits(speed_img)
    raw_miles = _read_digits(miles_img)
    return raw_speed, raw_miles


def _update_stop_timer(state: CorrectionState, speed_value: int | None, now: float) -> None:
    # Track how long we've been stopped to allow a distance reset.
    # The timer is cleared as soon as we detect movement.
    if _is_stopped(speed_value):
        if state.stop_start is None:
            state.stop_start = now
    else:
        state.stop_start = None


def _format_output(speed_value: int | None, miles_value: int | None) -> tuple[str, str]:
    # Format output with fixed-width mileage and blank placeholders.
    # Keep formatting isolated to simplify log output changes.
    speed = f"{speed_value}" if speed_value is not None else ""
    miles = f"{miles_value:0{ocr_config.MAX_DISTANCE_DIGITS}d}" if miles_value is not None else ""
    return speed, miles


def _format_distance(value: int | None) -> str:
    return f"{value:0{ocr_config.MAX_DISTANCE_DIGITS}d}" if value is not None else ""


def _update_fps(state: StatusState, now: float) -> None:
    # Update FPS once per second using a rolling frame counter.
    # This avoids per-frame prints while still exposing throughput.
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
    next_signal_number: str,
    next_signal_distance: str,
    platform_number: str,
    route_number: str,
    next_station: str,
    now: float,
) -> bool:
    # Decide whether to print based on change detection or time interval.
    # This keeps the log quieter while still emitting periodic status lines.
    changed = (
        speed != state.last_speed_text
        or miles != state.last_miles_text
        or next_signal_number != state.last_next_signal_number_text
        or next_signal_distance != state.last_next_signal_distance_text
        or platform_number != state.last_platform_number_text
        or route_number != state.last_route_number_text
        or next_station != state.last_next_station_text
    )
    time_since_status = now - state.last_status
    return (
        (not ocr_config.PRINT_ON_CHANGE_ONLY)
        or changed
        or time_since_status >= ocr_config.STATUS_EVERY_SECONDS
    )


def _allow_distance_reset(state: CorrectionState, now: float) -> bool:
    # Permit distance reset only after a stable stopped interval.
    # This prevents false resets from brief OCR dropouts.
    return state.stop_start is not None and (
        now - state.stop_start
    ) >= ocr_config.MIN_SPEED_STABLE_FOR_RESET_SEC


def main() -> None:
    _apply_ocr_settings()
    regions = Regions(
        speed=ocr_config.REGION_SPEED,
        miles=ocr_config.REGION_MILES,
        next_signal_number=ocr_config.REGION_NEXT_SIGNAL_NUMBER,
        next_signal_distance=ocr_config.REGION_NEXT_SIGNAL_DISTANCE,
        platform_number=ocr_config.REGION_PLATFORM_NUMBER,
        route_number=ocr_config.REGION_ROUTE_NUMBER,
        next_station=ocr_config.REGION_NEXT_STATION,
    )

    target_frame_time = 1.0 / max(1, ocr_config.TARGET_FPS)
    correction_state = CorrectionState(
        speed=SpeedState(),
        distance=DistanceState(),
        next_signal_distance=DistanceState(),
    )
    status_state = StatusState(last_update=time.perf_counter(), fps_timer=time.perf_counter())

    with mss.mss() as sct:
        while True:
            loop_start = time.perf_counter()
            # Capture the frame as early as possible to minimize latency.
            full = _grab_full_screen(sct)

            raw_speed, raw_miles = _read_regions(full, regions)
            raw_next_signal_number = _read_text(
                _preprocess(_crop(full, regions.next_signal_number)),
                whitelist=ocr_config.ALPHANUM_WHITELIST,
                allow_spaces=False,
            )
            raw_next_signal_distance = _read_digits(
                _preprocess(_crop(full, regions.next_signal_distance))
            )

            now = time.perf_counter()
            # Clamp delta_t so corrections don't explode on long pauses.
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
            next_signal_distance_value = apply_distance_correction(
                raw_next_signal_distance,
                correction_state.next_signal_distance,
                speed_value,
                delta_t,
                allow_reset,
            )

            if speed_value is not None:
                correction_state.speed.last_speed_value = speed_value
            if miles_value is not None:
                correction_state.distance.last_miles_value = miles_value
            if next_signal_distance_value is not None:
                correction_state.next_signal_distance.last_miles_value = next_signal_distance_value

            speed, miles = _format_output(speed_value, miles_value)
            next_signal_number = raw_next_signal_number
            next_signal_distance = _format_distance(next_signal_distance_value)

            if now - status_state.last_slow_read >= ocr_config.SLOW_READ_INTERVAL:
                status_state.last_slow_read = now
                raw_platform_number = _read_digits(
                    _preprocess(_crop(full, regions.platform_number))
                )
                raw_platform_number = _normalize_platform_number(raw_platform_number)
                raw_route_number = _read_digits(_preprocess(_crop(full, regions.route_number)))
                raw_next_station = _read_text(
                    _preprocess(_crop(full, regions.next_station)),
                    whitelist=f"{ocr_config.STATION_NAME_WHITELIST} ",
                    allow_spaces=True,
                    force_upper=False,
                )
                status_state.last_platform_number_text = raw_platform_number
                status_state.last_route_number_text = raw_route_number
                status_state.last_next_station_text = raw_next_station

            platform_number = status_state.last_platform_number_text or ""
            route_number = status_state.last_route_number_text or ""
            next_station = status_state.last_next_station_text or ""
            _update_fps(status_state, now)

            if _should_print(
                status_state,
                speed,
                miles,
                next_signal_number,
                next_signal_distance,
                platform_number,
                route_number,
                next_station,
                now,
            ):
                # Cache last printed values for change detection.
                _print_status(
                    speed,
                    miles,
                    next_signal_number,
                    next_signal_distance,
                    platform_number,
                    route_number,
                    next_station,
                    status_state.fps,
                )
                status_state.last_status = now
                status_state.last_speed_text = speed
                status_state.last_miles_text = miles
                status_state.last_next_signal_number_text = next_signal_number
                status_state.last_next_signal_distance_text = next_signal_distance
                status_state.last_platform_number_text = platform_number
                status_state.last_route_number_text = route_number
                status_state.last_next_station_text = next_station

            loop_end = time.perf_counter()
            # Sleep only if we're faster than the target FPS.
            sleep_for = target_frame_time - (loop_end - loop_start)
            if sleep_for > 0:
                time.sleep(sleep_for)


if __name__ == "__main__":
    main()
