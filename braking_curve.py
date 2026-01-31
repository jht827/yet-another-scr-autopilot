"""Braking curve processing and jitter handling."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List
import csv
from collections import deque


@dataclass
class BrakingSample:
    timestamp: float
    speed: float


@dataclass
class BrakingCurve:
    speeds: List[float]
    distances: List[float]

    def required_distance(self, speed: float) -> float:
        """Return the remaining distance needed to stop at the given speed."""
        if not self.speeds:
            return 0.0
        if speed >= self.speeds[0]:
            return self.distances[0]
        if speed <= self.speeds[-1]:
            return self.distances[-1]

        for index in range(1, len(self.speeds)):
            high = self.speeds[index - 1]
            low = self.speeds[index]
            if low <= speed <= high:
                span = high - low
                if span == 0:
                    return self.distances[index]
                ratio = (speed - low) / span
                return self.distances[index] + ratio * (
                    self.distances[index - 1] - self.distances[index]
                )
        return self.distances[-1]


class SpeedJitterFilter:
    """Smooth incoming speed values to reduce jitter."""

    def __init__(self, window: int) -> None:
        self._window = max(1, window)
        self._values: deque[float] = deque(maxlen=self._window)

    def update(self, speed: float) -> float:
        self._values.append(speed)
        if not self._values:
            return speed
        ordered = sorted(self._values)
        mid = len(ordered) // 2
        if len(ordered) % 2 == 1:
            return ordered[mid]
        return (ordered[mid - 1] + ordered[mid]) / 2.0


def _smooth_speeds(samples: Iterable[BrakingSample], window: int) -> List[float]:
    filter_window = SpeedJitterFilter(window)
    smoothed: List[float] = []
    last_speed = None
    for sample in samples:
        filtered = filter_window.update(sample.speed)
        if last_speed is not None:
            filtered = min(filtered, last_speed)
        smoothed.append(filtered)
        last_speed = filtered
    return smoothed


def _integrate_distances(samples: List[BrakingSample], speeds: List[float]) -> List[float]:
    distances = [0.0 for _ in samples]
    remaining = 0.0
    for index in range(len(samples) - 2, -1, -1):
        dt = samples[index + 1].timestamp - samples[index].timestamp
        avg_speed = (speeds[index] + speeds[index + 1]) / 2.0
        remaining += (avg_speed / 3600.0) * dt
        distances[index] = remaining
    return distances


def load_braking_curve(path: str | Path, jitter_window: int = 3) -> BrakingCurve:
    samples: List[BrakingSample] = []
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            samples.append(
                BrakingSample(
                    timestamp=float(row["timestamp"]),
                    speed=float(row["speed"]),
                )
            )

    if not samples:
        return BrakingCurve([], [])

    smoothed = _smooth_speeds(samples, jitter_window)
    distances = _integrate_distances(samples, smoothed)
    return BrakingCurve(speeds=smoothed, distances=distances)
