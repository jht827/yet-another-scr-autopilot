#!/usr/bin/env python3
"""Interactively select ROI boxes from the Roblox window and save to config."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
from pathlib import Path
from typing import Any

from PIL import ImageGrab


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


def _load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _save_config(path: Path, config: dict[str, Any]) -> None:
    path.write_text(json.dumps(config, indent=2, sort_keys=True))


def _tk_available() -> bool:
    return importlib.util.find_spec("tkinter") is not None


def _get_fullscreen_bbox() -> tuple[int, int, int, int]:
    image = ImageGrab.grab()
    return (0, 0, image.width, image.height)


class RoiBoxer:
    def __init__(self, config_path: Path) -> None:
        self.config_path = config_path
        self.config = _load_config(config_path)
        self.window_title = self.config.get("window_title", "Roblox")
        self.window_bbox = _get_window_bbox(self.window_title)
        if self.window_bbox is None:
            self.window_bbox = self._fallback_bbox()

    def _fallback_bbox(self) -> tuple[int, int, int, int]:
        window_bbox = self.config.get("window_bbox")
        if window_bbox:
            return tuple(window_bbox)
        print("Warning: window lookup unavailable. Using fullscreen coordinates.")
        return _get_fullscreen_bbox()

    def run_gui(self) -> None:
        tkinter = importlib.import_module("tkinter")
        messagebox = importlib.import_module("tkinter.messagebox")
        simpledialog = importlib.import_module("tkinter.simpledialog")
        image_tk = importlib.import_module("PIL.ImageTk")

        image = ImageGrab.grab(bbox=self.window_bbox)
        photo = image_tk.PhotoImage(image)

        root = tkinter.Tk()
        root.title("ROI Boxer")
        canvas = tkinter.Canvas(root, width=image.width, height=image.height)
        canvas.pack()
        canvas.create_image(0, 0, image=photo, anchor=tkinter.NW)

        state = {"start_x": 0, "start_y": 0, "end_x": 0, "end_y": 0, "rect_id": None}

        def on_press(event: tkinter.Event) -> None:
            state["start_x"] = event.x
            state["start_y"] = event.y
            if state["rect_id"] is not None:
                canvas.delete(state["rect_id"])
            state["rect_id"] = canvas.create_rectangle(
                state["start_x"],
                state["start_y"],
                state["start_x"],
                state["start_y"],
                outline="red",
                width=2,
            )

        def on_drag(event: tkinter.Event) -> None:
            if state["rect_id"] is None:
                return
            canvas.coords(
                state["rect_id"],
                state["start_x"],
                state["start_y"],
                event.x,
                event.y,
            )

        def on_release(event: tkinter.Event) -> None:
            state["end_x"] = event.x
            state["end_y"] = event.y
            if state["start_x"] == state["end_x"] or state["start_y"] == state["end_y"]:
                return
            fields = sorted(self.config.get("fields", {}).keys())
            field_hint = ", ".join(fields) if fields else "speed"
            field_name = simpledialog.askstring(
                "Field Name",
                f"Enter field name to update (existing: {field_hint}):",
                parent=root,
            )
            if not field_name:
                return
            self._update_field(
                field_name,
                state["start_x"],
                state["start_y"],
                state["end_x"],
                state["end_y"],
            )
            messagebox.showinfo(
                "ROI Updated",
                f"Saved ROI for '{field_name}'.",
            )

        canvas.bind("<ButtonPress-1>", on_press)
        canvas.bind("<B1-Motion>", on_drag)
        canvas.bind("<ButtonRelease-1>", on_release)

        root.mainloop()

    def run_cli(self, snapshot_path: Path) -> None:
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        image = ImageGrab.grab(bbox=self.window_bbox)
        image.save(snapshot_path)
        print(f"Saved window snapshot to {snapshot_path}")
        print("Open the image and enter ROI coords as: left top right bottom")
        field_name = input("Field name to update: ").strip()
        coords = input("ROI coords (l t r b): ").strip().split()
        if len(coords) != 4:
            raise ValueError("Expected 4 integers for ROI coords.")
        x1, y1, x2, y2 = (int(value) for value in coords)
        self._update_field(field_name, x1, y1, x2, y2)
        print(f"Saved ROI for '{field_name}': [{x1}, {y1}, {x2}, {y2}]")

    def _update_field(self, field_name: str, x1: int, y1: int, x2: int, y2: int) -> None:
        left = min(x1, x2)
        top = min(y1, y2)
        right = max(x1, x2)
        bottom = max(y1, y2)
        fields = self.config.setdefault("fields", {})
        field_cfg = fields.setdefault(field_name, {})
        field_cfg["roi"] = [left, top, right, bottom]
        _save_config(self.config_path, self.config)


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
    parser.add_argument(
        "--mode",
        choices=("auto", "gui", "cli"),
        default="auto",
        help="Use GUI (tkinter) or CLI fallback.",
    )
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=Path("recordings/roi_snapshot.png"),
        help="Where to save the snapshot for CLI mode.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not args.config.exists():
        raise FileNotFoundError(f"Config not found: {args.config}")
    boxer = RoiBoxer(args.config)
    if args.mode == "gui":
        boxer.run_gui()
        return
    if args.mode == "cli":
        boxer.run_cli(args.snapshot)
        return
    if _tk_available():
        boxer.run_gui()
    else:
        boxer.run_cli(args.snapshot)


if __name__ == "__main__":
    main()
