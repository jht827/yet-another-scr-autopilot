"""OCR correction helpers for speed and distance."""
from __future__ import annotations

from dataclasses import dataclass

import ocr_config


@dataclass
class DistanceState:
    last_miles_value: int | None = None
    last_prediction: float | None = None
    miles_reject_streak: int = 0


@dataclass
class SpeedState:
    last_speed_value: int | None = None
    single_digit_streak: int = 0
    speed_reject_streak: int = 0


def _parse_int(text: str) -> int | None:
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def _digits_only(text: str) -> str:
    return "".join(ch for ch in text if ch.isdigit())


def clamp_distance_digits(text: str) -> str:
    digits = _digits_only(text)
    if len(digits) <= ocr_config.MAX_DISTANCE_DIGITS:
        return digits
    return digits[: ocr_config.MAX_DISTANCE_DIGITS]


def apply_speed_correction(raw_speed: str, state: SpeedState, delta_t: float) -> int | None:
    """Clamp speed changes and reject single-digit dropouts."""
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


def _distance_tolerance(speed_value: int | None) -> float:
    speed = speed_value or 0
    return ocr_config.DISTANCE_BASE_TOLERANCE + (speed * ocr_config.DISTANCE_TOLERANCE_PER_MPH)


def _predict_distance(
    last_value: int,
    speed_value: int | None,
    delta_t: float,
) -> float:
    speed = speed_value or 0
    # Offset for OCR speed lag so prediction doesn't overrun.
    adjusted_delta = max(0.0, delta_t - ocr_config.SPEED_LAG_SEC)
    return last_value + (speed / 3600.0) * adjusted_delta


def apply_distance_correction(
    raw_miles: str,
    state: DistanceState,
    speed_value: int | None,
    delta_t: float,
    allow_reset: bool,
) -> int | None:
    """Use speed integration to validate OCR distance at high speed."""
    miles_digits = clamp_distance_digits(raw_miles)
    miles_value = _parse_int(miles_digits)
    if miles_value is None:
        return None

    if state.last_miles_value is None:
        state.last_prediction = float(miles_value)
        return miles_value

    prediction = _predict_distance(state.last_miles_value, speed_value, delta_t)
    tolerance = _distance_tolerance(speed_value)
    delta = miles_value - prediction
    max_distance = (10**ocr_config.MAX_DISTANCE_DIGITS) - 1
    allow_jump = allow_reset and miles_value <= max_distance

    if abs(delta) > tolerance and not allow_jump:
        miles_value = state.last_miles_value
        state.miles_reject_streak += 1
    else:
        state.miles_reject_streak = 0

    if state.miles_reject_streak >= ocr_config.MAX_DISTANCE_REJECT_FRAMES:
        state.miles_reject_streak = 0
        return _parse_int(miles_digits)

    state.last_prediction = prediction
    return miles_value
