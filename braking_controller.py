"""Main braking automation loop."""
from __future__ import annotations

import ocr_config
import time

from braking_curve import SpeedJitterFilter, load_braking_curve
from key_controls import KeyController
from ocr import stream_ocr


def run_braking_loop(show_status: bool = True) -> None:
    curve = load_braking_curve(
        ocr_config.BRAKING_CURVE_CSV,
        jitter_window=ocr_config.JITTER_WINDOW,
    )
    speed_filter = SpeedJitterFilter(ocr_config.JITTER_WINDOW)
    controller = KeyController()
    braking_issued = False
    doors_opened = False

    for reading in stream_ocr(print_status=False):
        if reading.speed_value is None or reading.miles_value is None:
            continue

        filtered_speed = speed_filter.update(float(reading.speed_value))
        effective_distance = float(reading.miles_value) + ocr_config.STATION_DISTANCE_ADDITION
        required_distance = curve.required_distance(filtered_speed)

        if show_status:
            timestamp = time.strftime("%H:%M:%S")
            print(
                f"[{timestamp}] speed={filtered_speed:.1f} "
                f"distance={effective_distance:.1f} required_stop={required_distance:.1f}",
                flush=True,
            )

        if not braking_issued and effective_distance <= required_distance:
            timestamp = time.strftime("%H:%M:%S")
            print(f"[{timestamp}] braking: holding {ocr_config.BRAKE_KEY}", flush=True)
            controller.hold_key(ocr_config.BRAKE_KEY, ocr_config.BRAKE_HOLD_SECONDS)
            braking_issued = True

        if filtered_speed <= 0 and not doors_opened:
            timestamp = time.strftime("%H:%M:%S")
            print(f"[{timestamp}] doors: opening", flush=True)
            controller.press_key(
                ocr_config.DOOR_KEY,
                times=ocr_config.DOOR_PRESS_COUNT,
                interval=ocr_config.DOOR_PRESS_INTERVAL,
            )
            doors_opened = True

        if filtered_speed > ocr_config.STOP_SPEED_THRESHOLD:
            if doors_opened:
                doors_opened = False
            if braking_issued and effective_distance > required_distance:
                braking_issued = False


if __name__ == "__main__":
    run_braking_loop(show_status=False)
