#!/usr/bin/env python3
"""Collect line data by OCR'ing HUD fields and integrating speed over time."""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytesseract
from PIL import Image, ImageGrab, ImageOps


@dataclass
class FieldConfig:
    name: str
    roi: tuple[int, int, int, int]
    psm: int = 7
    whitelist: str | None = None
    scale: int = 1
    threshold: int | None = None
    invert: bool = False


@dataclass
class CollectorConfig:
    fields: dict[str, FieldConfig] = field(default_factory=dict)
    window_title: str = "Roblox"
    poll_hz: float = 2.0
    watch_fields: tuple[str, ...] = ("signal_id", "next_stop", "platform")


def _load_config(path: Path) -> CollectorConfig:
    raw = json.loads(path.read_text())
    fields: dict[str, FieldConfig] = {}
    for name, cfg in raw.get("fields", {}).items():
        fields[name] = FieldConfig(
            name=name,
            roi=tuple(cfg["roi"]),
            psm=int(cfg.get("psm", 7)),
            whitelist=cfg.get("whitelist"),
            scale=int(cfg.get("scale", 1)),
            threshold=cfg.get("threshold"),
            invert=bool(cfg.get("invert", False)),
        )
    return CollectorConfig(
        fields=fields,
        window_title=raw.get("window_title", "Roblox"),
        poll_hz=float(raw.get("poll_hz", 2.0)),
        watch_fields=tuple(raw.get("watch_fields", ["signal_id", "next_stop", "platform"])),
    )


def _preprocess(image: Image.Image, cfg: FieldConfig) -> Image.Image:
    if cfg.scale > 1:
        image = image.resize(
            (image.width * cfg.scale, image.height * cfg.scale),
            resample=Image.NEAREST,
        )
    image = ImageOps.grayscale(image)
    if cfg.invert:
        image = ImageOps.invert(image)
    if cfg.threshold is not None:
        image = image.point(lambda p: 255 if p > cfg.threshold else 0)
    return image


def _ocr(image: Image.Image, cfg: FieldConfig) -> str:
    config_parts = [f"--psm {cfg.psm}"]
    if cfg.whitelist:
        config_parts.append(f"-c tessedit_char_whitelist={cfg.whitelist}")
    config = " ".join(config_parts)
    text = pytesseract.image_to_string(image, config=config)
    return text.strip()


def _get_window_bbox(window_title: str) -> tuple[int, int, int, int]:
    try:
        import pygetwindow
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "pygetwindow is required to capture the Roblox window. "
            "Install it with `pip install pygetwindow`.",
        ) from exc

    windows = pygetwindow.getWindowsWithTitle(window_title)
    if not windows:
        raise RuntimeError(f"No window found with title containing '{window_title}'.")
    window = windows[0]
    return (window.left, window.top, window.right, window.bottom)


def _capture_field(cfg: FieldConfig, window_bbox: tuple[int, int, int, int]) -> str:
    left, top, _, _ = window_bbox
    x1, y1, x2, y2 = cfg.roi
    image = ImageGrab.grab(bbox=(left + x1, top + y1, left + x2, top + y2))
    image = _preprocess(image, cfg)
    return _ocr(image, cfg)


def _parse_float(value: str) -> float | None:
    cleaned = value.replace(",", ".").replace("O", "0").replace("o", "0")
    try:
        return float(cleaned)
    except ValueError:
        return None


def _utc_now() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _write_header(writer: csv.DictWriter[str]) -> None:
    writer.writeheader()


def _write_event(
    writer: csv.DictWriter[str],
    event_type: str,
    value: str,
    distance_m: float,
    snapshot: dict[str, Any],
) -> None:
    row = {
        "timestamp": _utc_now(),
        "event_type": event_type,
        "value": value,
        "distance_m": f"{distance_m:.2f}",
    }
    row.update(snapshot)
    writer.writerow(row)


def collect(config_path: Path, output_path: Path) -> None:
    config = _load_config(config_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    last_time = time.monotonic()
    distance_m = 0.0
    last_values: dict[str, str] = {}

    window_bbox = _get_window_bbox(config.window_title)

    with output_path.open("w", newline="") as handle:
        fieldnames = [
            "timestamp",
            "event_type",
            "value",
            "distance_m",
            "speed_mph",
            "speed_limit_mph",
            "distance_to_next_station",
            "signal_id",
            "next_stop",
            "platform",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        _write_header(writer)

        poll_interval = 1.0 / max(config.poll_hz, 0.1)
        print("Starting line data collection. Press Ctrl+C to stop.")

        while True:
            now = time.monotonic()
            dt = now - last_time
            last_time = now

            values: dict[str, str] = {}
            for name, field_cfg in config.fields.items():
                values[name] = _capture_field(field_cfg, window_bbox)

            speed_mph = _parse_float(values.get("speed", ""))
            if speed_mph is None:
                speed_mph = 0.0
            speed_mps = speed_mph * 0.44704
            distance_m += speed_mps * dt

            snapshot = {
                "speed_mph": values.get("speed", ""),
                "speed_limit_mph": values.get("speed_limit", ""),
                "distance_to_next_station": values.get("distance_to_next_station", ""),
                "signal_id": values.get("signal_id", ""),
                "next_stop": values.get("next_stop", ""),
                "platform": values.get("platform", ""),
            }

            for field_name in config.watch_fields:
                current = values.get(field_name, "").strip()
                if not current:
                    continue
                if last_values.get(field_name) != current:
                    _write_event(writer, field_name, current, distance_m, snapshot)
                    handle.flush()
                    print(
                        f"[{field_name}] {current} @ {distance_m:.2f} m",
                    )
                    distance_m = 0.0
                    last_values[field_name] = current

            time.sleep(poll_interval)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect HUD line data via OCR and integrate speed into distances.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/line_data_config.json"),
        help="Path to global config JSON.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("recordings/line_data_events.csv"),
        help="Output CSV path.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    collect(args.config, args.output)


if __name__ == "__main__":
    main()
