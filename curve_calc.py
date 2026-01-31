import csv
from functools import lru_cache
from statistics import mean

from config import (
    DECEL_CURVE_PATH,
    CURVE_SMOOTHING_WINDOW,
    DISTANCE_ADJUST_MI,
    MPH_TO_MILES_PER_SEC,
)


def _smooth_series(values, window):
    if window <= 1:
        return values
    half = window // 2
    smoothed = []
    for idx in range(len(values)):
        start = max(0, idx - half)
        end = min(len(values), idx + half + 1)
        smoothed.append(mean(values[start:end]))
    return smoothed


@lru_cache(maxsize=1)
def _load_decel_curve():
    timestamps = []
    speeds_mph = []
    with open(DECEL_CURVE_PATH, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            try:
                timestamps.append(float(row["timestamp"]))
                speeds_mph.append(float(row["speed"]))
            except (KeyError, ValueError):
                continue

    if len(timestamps) < 2:
        raise ValueError("Deceleration curve is too short or invalid.")

    speeds_mph = _smooth_series(speeds_mph, CURVE_SMOOTHING_WINDOW)

    distances = [0.0]
    for idx in range(1, len(timestamps)):
        dt = timestamps[idx] - timestamps[idx - 1]
        if dt <= 0:
            distances.append(distances[-1])
            continue
        avg_speed_miles_per_sec = (
            (speeds_mph[idx] + speeds_mph[idx - 1]) * 0.5 * MPH_TO_MILES_PER_SEC
        )
        distances.append(distances[-1] + avg_speed_miles_per_sec * dt)

    total_distance = distances[-1]
    speed_distance_pairs = []
    for speed, distance_travelled in zip(speeds_mph, distances):
        distance_remaining = max(total_distance - distance_travelled, 0.0)
        speed_distance_pairs.append((speed, distance_remaining))

    speed_distance_pairs.sort(key=lambda pair: pair[0])

    aggregated = []
    current_speed = None
    bucket = []
    for speed, distance in speed_distance_pairs:
        if current_speed is None or abs(speed - current_speed) < 1e-6:
            current_speed = speed
            bucket.append(distance)
        else:
            aggregated.append((current_speed, mean(bucket)))
            current_speed = speed
            bucket = [distance]
    if bucket:
        aggregated.append((current_speed, mean(bucket)))

    speeds_sorted = [pair[0] for pair in aggregated]
    distances_sorted = [pair[1] for pair in aggregated]

    return speeds_sorted, distances_sorted


def braking_distance(speed_mph, distance_adjust_mi=DISTANCE_ADJUST_MI):
    speeds, distances = _load_decel_curve()
    if speed_mph <= speeds[0]:
        return max(distances[0] + distance_adjust_mi, 0.0)
    if speed_mph >= speeds[-1]:
        return max(distances[-1] + distance_adjust_mi, 0.0)

    lo = 0
    hi = len(speeds) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if speeds[mid] < speed_mph:
            lo = mid + 1
        elif speeds[mid] > speed_mph:
            hi = mid - 1
        else:
            return max(distances[mid] + distance_adjust_mi, 0.0)

    upper = lo
    lower = lo - 1
    speed_low = speeds[lower]
    speed_high = speeds[upper]
    dist_low = distances[lower]
    dist_high = distances[upper]

    if speed_high == speed_low:
        interpolated = dist_low
    else:
        ratio = (speed_mph - speed_low) / (speed_high - speed_low)
        interpolated = dist_low + ratio * (dist_high - dist_low)

    return max(interpolated + distance_adjust_mi, 0.0)
