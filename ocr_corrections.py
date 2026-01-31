"""OCR correction helpers for speed and distance."""
from __future__ import annotations

from dataclasses import dataclass, field

import ocr_config

_DIGIT_CONFUSIONS: dict[str, tuple[str, ...]] = {}


@dataclass
class DistanceState:
    last_miles_value: int | None = None
    last_prediction: float | None = None
    miles_reject_streak: int = 0
    recent_raw_miles: list[str] = field(default_factory=list)


@dataclass
class SpeedState:
    last_speed_value: int | None = None
    single_digit_streak: int = 0
    speed_reject_streak: int = 0


def _parse_int(text: str) -> int | None:
    # Convert numeric strings to ints, returning None on empty/invalid input.
    # OCR sometimes returns whitespace or garbage; those should be ignored.
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def _digits_only(text: str) -> str:
    # Strip non-numeric characters from OCR output.
    # This preserves just the digits for downstream numeric parsing.
    return "".join(ch for ch in text if ch.isdigit())


def clamp_distance_digits(text: str) -> str:
    # Enforce the expected max digit width for distance values.
    # Extra digits are most likely OCR noise, so truncate them.
    digits = _digits_only(text)
    if len(digits) <= ocr_config.MAX_DISTANCE_DIGITS:
        return digits
    return digits[: ocr_config.MAX_DISTANCE_DIGITS]


def apply_speed_correction(raw_speed: str, state: SpeedState, delta_t: float) -> int | None:
    """Clamp speed changes and reject single-digit dropouts."""
    # Keep speed stable by rejecting implausible jumps between frames.
    speed_value = _parse_int(raw_speed)
    if speed_value is None:
        return None

    if state.last_speed_value is not None:
        # Limit acceleration/deceleration per second to avoid OCR spikes.
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
        # Guard against dropping from a multi-digit speed to a single digit.
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
        # After too many rejections, accept the raw value to recover.
        state.speed_reject_streak = 0
        return _parse_int(raw_speed)

    return speed_value


def _distance_tolerance(speed_value: int | None) -> float:
    # Allow larger distance deltas at higher speeds.
    # This protects against false positives at speed.
    speed = speed_value or 0
    return ocr_config.DISTANCE_BASE_TOLERANCE + (speed * ocr_config.DISTANCE_TOLERANCE_PER_MPH)


def _predict_distance(
    last_value: int,
    speed_value: int | None,
    delta_t: float,
) -> float:
    # Estimate distance from last value using current speed.
    speed = speed_value or 0
    # Offset for OCR speed lag so prediction doesn't overrun.
    adjusted_delta = max(0.0, delta_t - ocr_config.SPEED_LAG_SEC)
    return last_value + (speed / 3600.0) * adjusted_delta


def _best_distance_candidate(digits: str, prediction: float, tolerance: float) -> int | None:
    # Search for a single-digit substitution that stays within tolerance.
    # This handles common OCR digit confusions (like 5 vs 6).
    if not digits:
        return None
    best_value = None
    best_delta = None
    for index, original in enumerate(digits):
        for replacement in _DIGIT_CONFUSIONS.get(original, ()):
            candidate_digits = f"{digits[:index]}{replacement}{digits[index + 1:]}"
            candidate = _parse_int(candidate_digits)
            if candidate is None:
                continue
            delta = abs(candidate - prediction)
            if delta > tolerance:
                continue
            if best_delta is None or delta < best_delta:
                best_delta = delta
                best_value = candidate
    return best_value


def _should_fix_07x_to_02x(miles_digits: str, state: DistanceState) -> bool:
    # Fix common OCR confusion where "02x" appears as "07x" after "03x" readings.
    # The lookback window ensures this only triggers when the pattern repeats.
    if len(miles_digits) != ocr_config.MAX_DISTANCE_DIGITS:
        return False
    if not miles_digits.startswith("07"):
        return False
    lookback = ocr_config.DISTANCE_07X_FIX_LOOKBACK
    required = ocr_config.DISTANCE_03X_REQUIRED
    recent = state.recent_raw_miles[-lookback:] if lookback > 0 else []
    count_03x = sum(1 for digits in recent if len(digits) == 3 and digits.startswith("03"))
    return count_03x >= required


def apply_distance_correction(
    raw_miles: str,
    state: DistanceState,
    speed_value: int | None,
    delta_t: float,
    allow_reset: bool,
) -> int | None:
    """Use speed integration to validate OCR distance at high speed."""
    # Keep the raw digits around for pattern-based corrections.
    raw_digits = clamp_distance_digits(raw_miles)
    miles_digits = raw_digits
    if _should_fix_07x_to_02x(miles_digits, state):
        # Apply the correction before parsing to keep history consistent.
        miles_digits = f"02{miles_digits[2]}"
    state.recent_raw_miles.append(raw_digits)
    if len(state.recent_raw_miles) > ocr_config.DISTANCE_07X_FIX_LOOKBACK:
        state.recent_raw_miles = state.recent_raw_miles[-ocr_config.DISTANCE_07X_FIX_LOOKBACK :]
    miles_value = _parse_int(miles_digits)
    if miles_value is None:
        return None

    if state.last_miles_value is None:
        # First reading establishes the baseline for predictions.
        state.last_prediction = float(miles_value)
        return miles_value

    prediction = _predict_distance(state.last_miles_value, speed_value, delta_t)
    tolerance = _distance_tolerance(speed_value)
    delta = miles_value - prediction
    max_distance = (10**ocr_config.MAX_DISTANCE_DIGITS) - 1
    allow_jump = allow_reset and miles_value <= max_distance

    if abs(delta) > tolerance and not allow_jump:
        # If OCR is out of bounds, attempt a digit-level correction first.
        candidate = _best_distance_candidate(miles_digits, prediction, tolerance)
        if candidate is not None:
            miles_value = candidate
            state.miles_reject_streak = 0
        else:
            # Otherwise, stick to the last good value and count the reject.
            miles_value = state.last_miles_value
            state.miles_reject_streak += 1
    else:
        state.miles_reject_streak = 0

    if state.miles_reject_streak >= ocr_config.MAX_DISTANCE_REJECT_FRAMES:
        # After too many rejects, trust raw OCR to re-sync.
        state.miles_reject_streak = 0
        return _parse_int(miles_digits)

    state.last_prediction = prediction
    return miles_value
