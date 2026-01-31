from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

DEFAULT_CURVE = Path("train_curves/357/decel_linear.csv")


@dataclass(frozen=True)
class BrakeTuning:
    target_speed_mph: float
    safety_margin_miles: float
    response_lag_seconds: float
    max_brake_command: float


def parse_distance_raw(raw: str) -> float:
    cleaned = "".join(ch for ch in raw if ch.isdigit())
    if not cleaned:
        raise ValueError("Distance input must contain digits.")
    return int(cleaned) / 100.0


def load_curve(path: Path) -> list[tuple[float, float]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path} is missing a header row.")
        speed_key = next((name for name in reader.fieldnames if "speed" in name), None)
        accel_key = next((name for name in reader.fieldnames if "decel" in name or "accel" in name), None)
        if not speed_key or not accel_key:
            raise ValueError(f"{path} must include speed and accel/decel columns.")
        points: list[tuple[float, float]] = []
        for row in reader:
            if not row.get(speed_key) or not row.get(accel_key):
                continue
            points.append((float(row[speed_key]), float(row[accel_key])))
    if len(points) < 2:
        raise ValueError(f"{path} must contain at least two points.")
    return sorted(points, key=lambda item: item[0])


def lerp(points: Iterable[tuple[float, float]], speed_mph: float) -> float:
    items = list(points)
    if speed_mph <= items[0][0]:
        return items[0][1]
    if speed_mph >= items[-1][0]:
        return items[-1][1]
    for (s0, a0), (s1, a1) in zip(items, items[1:]):
        if s0 <= speed_mph <= s1:
            ratio = (speed_mph - s0) / (s1 - s0)
            return a0 + ratio * (a1 - a0)
    return items[-1][1]


def required_decel(speed_mph: float, target_speed_mph: float, distance_miles: float) -> float:
    if distance_miles <= 0:
        return float("inf")
    speed_mps = speed_mph / 3600.0
    target_mps = target_speed_mph / 3600.0
    return max(0.0, (speed_mps**2 - target_mps**2) / (2 * distance_miles))


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def compute_brake_command(
    speed_mph: float,
    distance_miles: float,
    curve_points: list[tuple[float, float]],
    tuning: BrakeTuning,
) -> tuple[float, float, float]:
    adjusted_distance = max(0.0, distance_miles - tuning.safety_margin_miles)
    target_decel = required_decel(speed_mph, tuning.target_speed_mph, adjusted_distance)
    curve_decel = lerp(curve_points, speed_mph)
    if curve_decel <= 0:
        return target_decel, curve_decel, tuning.max_brake_command
    command = clamp(target_decel / curve_decel, 0.0, tuning.max_brake_command)
    return target_decel, curve_decel, command


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute a braking command based on a linear braking curve.",
    )
    parser.add_argument("--speed-mph", type=float, required=True, help="Current speed in mph.")
    parser.add_argument(
        "--distance-raw",
        required=True,
        help="Distance readout as 3 digits, where last 2 digits are decimals (e.g., 300 -> 3.00 miles).",
    )
    parser.add_argument(
        "--curve",
        type=Path,
        default=DEFAULT_CURVE,
        help="Path to the linear braking curve CSV.",
    )
    parser.add_argument("--target-speed-mph", type=float, default=0.0, help="Target speed in mph.")
    parser.add_argument(
        "--safety-margin-miles",
        type=float,
        default=0.05,
        help="Subtract this margin from the distance to brake earlier.",
    )
    parser.add_argument(
        "--response-lag-seconds",
        type=float,
        default=0.6,
        help="Response lag in seconds for smoothing command changes.",
    )
    parser.add_argument(
        "--max-brake-command",
        type=float,
        default=1.0,
        help="Maximum brake command (0-1).",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    distance_miles = parse_distance_raw(args.distance_raw)
    curve_points = load_curve(args.curve)
    tuning = BrakeTuning(
        target_speed_mph=args.target_speed_mph,
        safety_margin_miles=args.safety_margin_miles,
        response_lag_seconds=args.response_lag_seconds,
        max_brake_command=args.max_brake_command,
    )
    target_decel, curve_decel, command = compute_brake_command(
        speed_mph=args.speed_mph,
        distance_miles=distance_miles,
        curve_points=curve_points,
        tuning=tuning,
    )
    print(f"Distance: {distance_miles:.2f} miles")
    print(f"Target decel: {target_decel:.6f} miles/s^2")
    print(f"Curve decel: {curve_decel:.6f} miles/s^2")
    print(f"Brake command (0-1): {command:.3f}")


if __name__ == "__main__":
    main()
