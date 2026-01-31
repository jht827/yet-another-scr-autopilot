"""Keyboard automation utilities."""
from __future__ import annotations

import time

try:
    from pynput.keyboard import Controller
except ImportError as exc:  # pragma: no cover - runtime dependency check
    raise RuntimeError(
        "pynput is required for key automation. Install it with 'pip install pynput'."
    ) from exc


class KeyController:
    """Send key presses and holds using pynput."""

    def __init__(self) -> None:
        self._controller = Controller()

    def hold_key(self, key: str, duration: float) -> None:
        self._controller.press(key)
        time.sleep(duration)
        self._controller.release(key)

    def press_key(self, key: str, times: int = 1, interval: float = 0.1) -> None:
        for index in range(times):
            self._controller.press(key)
            self._controller.release(key)
            if index < times - 1:
                time.sleep(interval)
