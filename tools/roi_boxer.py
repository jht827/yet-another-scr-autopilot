#!/usr/bin/env python3
"""Interactively select ROI boxes from the Roblox window and save to config."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from PIL import ImageGrab, ImageTk
import tkinter as tk
from tkinter import messagebox, simpledialog


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


def _load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _save_config(path: Path, config: dict[str, Any]) -> None:
    path.write_text(json.dumps(config, indent=2, sort_keys=True))


class RoiBoxer:
    def __init__(self, config_path: Path) -> None:
        self.config_path = config_path
        self.config = _load_config(config_path)
        self.window_title = self.config.get("window_title", "Roblox")
        self.window_bbox = _get_window_bbox(self.window_title)

        image = ImageGrab.grab(bbox=self.window_bbox)
        self.image = image
        self.photo = ImageTk.PhotoImage(image)

        self.root = tk.Tk()
        self.root.title("ROI Boxer")
        self.canvas = tk.Canvas(self.root, width=image.width, height=image.height)
        self.canvas.pack()
        self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

        self.start_x = 0
        self.start_y = 0
        self.rect_id: int | None = None
        self.end_x = 0
        self.end_y = 0

        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)

    def on_press(self, event: tk.Event) -> None:
        self.start_x = event.x
        self.start_y = event.y
        if self.rect_id is not None:
            self.canvas.delete(self.rect_id)
        self.rect_id = self.canvas.create_rectangle(
            self.start_x,
            self.start_y,
            self.start_x,
            self.start_y,
            outline="red",
            width=2,
        )

    def on_drag(self, event: tk.Event) -> None:
        if self.rect_id is None:
            return
        self.canvas.coords(self.rect_id, self.start_x, self.start_y, event.x, event.y)

    def on_release(self, event: tk.Event) -> None:
        self.end_x = event.x
        self.end_y = event.y
        if self.start_x == self.end_x or self.start_y == self.end_y:
            return
        self._prompt_field_name()

    def _prompt_field_name(self) -> None:
        fields = sorted(self.config.get("fields", {}).keys())
        field_hint = ", ".join(fields) if fields else "speed"
        field_name = simpledialog.askstring(
            "Field Name",
            f"Enter field name to update (existing: {field_hint}):",
            parent=self.root,
        )
        if not field_name:
            return
        self._update_field(field_name)

    def _update_field(self, field_name: str) -> None:
        x1 = min(self.start_x, self.end_x)
        y1 = min(self.start_y, self.end_y)
        x2 = max(self.start_x, self.end_x)
        y2 = max(self.start_y, self.end_y)
        fields = self.config.setdefault("fields", {})
        field_cfg = fields.setdefault(field_name, {})
        field_cfg["roi"] = [x1, y1, x2, y2]
        _save_config(self.config_path, self.config)
        messagebox.showinfo(
            "ROI Updated",
            f"Saved ROI for '{field_name}': [{x1}, {y1}, {x2}, {y2}]",
        )

    def run(self) -> None:
        self.root.mainloop()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select ROIs from the Roblox window and save to config.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/line_data_config.json"),
        help="Path to global config JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not args.config.exists():
        raise FileNotFoundError(f"Config not found: {args.config}")
    boxer = RoiBoxer(args.config)
    boxer.run()


if __name__ == "__main__":
    main()
