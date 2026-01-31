#!/usr/bin/env python3
"""Collect line data by OCR'ing HUD fields and integrating speed over time."""

from __future__ import annotations

import argparse
import csv
import importlib
import importlib.util
import json
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

import pytesseract
from PIL import Image, ImageGrab, ImageOps
from pynput import keyboard


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
    window_bbox: tuple[int, int, int, int] | None = None
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
    window_bbox = raw.get("window_bbox")
    window_bbox_tuple = tuple(window_bbox) if window_bbox else None
    return CollectorConfig(
        fields=fields,
        window_title=raw.get("window_title", "Roblox"),
        window_bbox=window_bbox_tuple,
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


def _load_quartz() -> Any | None:
    if importlib.util.find_spec("Quartz") is None:
        return None
    return importlib.import_module("Quartz")


def _get_window_bbox(window_title: str) -> tuple[int, int, int, int] | None:
    quartz = _load_quartz()
    if quartz is None:
        return None
    options = quartz.kCGWindowListOptionOnScreenOnly | quartz.kCGWindowListExcludeDesktopElements
    window_list = quartz.CGWindowListCopyWindowInfo(options, quartz.kCGNullWindowID)
    for window in window_list:
        window_name = window.get("kCGWindowName", "") or ""
        owner_name = window.get("kCGWindowOwnerName", "") or ""
        if window_title.lower() in window_name.lower() or window_title.lower() in owner_name.lower():
            bounds = window.get("kCGWindowBounds", {})
            left = int(bounds.get("X", 0))
            top = int(bounds.get("Y", 0))
            width = int(bounds.get("Width", 0))
            height = int(bounds.get("Height", 0))
            return (left, top, left + width, top + height)
    raise RuntimeError(f"No window found with title containing '{window_title}'.")


def _capture_field(cfg: FieldConfig, window_bbox: tuple[int, int, int, int] | None) -> str:
    x1, y1, x2, y2 = cfg.roi
    if window_bbox is None:
        image = ImageGrab.grab(bbox=(x1, y1, x2, y2))
    else:
        left, top, _, _ = window_bbox
        image = ImageGrab.grab(bbox=(left + x1, top + y1, left + x2, top + y2))
    image = _preprocess(image, cfg)
    return _ocr(image, cfg)


def _prompt_fullscreen_fallback(reason: str) -> None:
    print(reason)
    try:
        response = input("Fallback to full screen coordinates? [y/N]: ").strip().lower()
    except EOFError as exc:
        raise RuntimeError("Window capture unavailable and no fallback confirmation received.") from exc
    if response not in {"y", "yes"}:
        raise RuntimeError("Aborted line data collection (fullscreen fallback declined).")


def _resolve_window_bbox(config: CollectorConfig) -> tuple[int, int, int, int] | None:
    if config.window_bbox:
        return config.window_bbox
    try:
        window_bbox = _get_window_bbox(config.window_title)
    except RuntimeError as exc:
        _prompt_fullscreen_fallback(str(exc))
        return None
    if window_bbox is None:
        _prompt_fullscreen_fallback(
            "Window lookup unavailable (Quartz backend missing).",
        )
    return window_bbox


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
    key_queue: deque[str] = deque()
    queue_lock = Lock()
    platform_open = False

    window_bbox = _resolve_window_bbox(config)
    if window_bbox is None:
        print(
            "Warning: Using absolute screen coordinates.",
        )

    def _on_key_press(key: keyboard.Key | keyboard.KeyCode) -> None:
        try:
            char = key.char
        except AttributeError:
            return
        if not char:
            return
        char = char.lower()
        if char in {"p", "2", "3", "4", "5", "6", "9"}:
            with queue_lock:
                key_queue.append(char)

    listener = keyboard.Listener(on_press=_on_key_press)
    listener.start()

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
        print(
            "Starting line data collection. Press Ctrl+C to stop. "
            "Hotkeys: P toggles platform start/end; 2-6 mark stop markers; 9 marks S.",
        )

        try:
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

                with queue_lock:
                    pending_keys = list(key_queue)
                    key_queue.clear()
                for key in pending_keys:
                    if key == "p":
                        event_type = "platform_marker"
                        value = "start" if not platform_open else "end"
                        platform_open = not platform_open
                    elif key == "9":
                        event_type = "stop_marker"
                        value = "S"
                    else:
                        event_type = "stop_marker"
                        value = key
                    _write_event(writer, event_type, value, distance_m, snapshot)
                    handle.flush()
                    print(
                        f"[{event_type}] {value} @ {distance_m:.2f} m",
                    )
                    distance_m = 0.0

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
        finally:
            listener.stop()


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
