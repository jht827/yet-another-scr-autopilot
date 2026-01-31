"""OCR helpers for reading HUD values quickly and reliably."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageGrab

from config import (
    REGION_DOOR_STATUS,
    REGION_SELECT_DESTINATION,
    SELECT_DESTINATION_INTERVAL,
)


@dataclass
class OcrDebug:
    raw_text: Optional[str] = None
    digits: Optional[str] = None
    parsed_value: Optional[float] = None
    note: Optional[str] = None
    region: Optional[Tuple[int, int, int, int]] = None


_LAST_OCR_DEBUG: Dict[str, OcrDebug] = {
    "speed": OcrDebug(),
    "distance": OcrDebug(),
}


last_speed: Optional[float] = None
last_speed_time: float = 0.0
last_select_check_time: float = 0.0

distance_history: list[float] = []


def _update_debug(key: str, **updates) -> None:
    debug = _LAST_OCR_DEBUG.setdefault(key, OcrDebug())
    for field, value in updates.items():
        setattr(debug, field, value)


def get_last_ocr_debug(key: str) -> Dict[str, Optional[str]]:
    debug = _LAST_OCR_DEBUG.get(key, OcrDebug())
    return {
        "raw_text": debug.raw_text,
        "digits": debug.digits,
        "parsed_value": debug.parsed_value,
        "note": debug.note,
        "region": debug.region,
    }


def grab_fullscreen() -> Image.Image:
    return ImageGrab.grab()


def _get_region_image(region: Tuple[int, int, int, int], frame: Optional[Image.Image]) -> Image.Image:
    if frame is None:
        return ImageGrab.grab(bbox=region)
    if isinstance(frame, Image.Image):
        return frame.crop(region)
    raise TypeError("frame must be a PIL Image or None")


def _preprocess_for_ocr(image: Image.Image) -> np.ndarray:
    image_np = np.array(image)
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh


def _read_number_raw(
    region: Tuple[int, int, int, int],
    frame: Optional[Image.Image],
    allow_decimal: bool = True,
) -> Tuple[Optional[str], Optional[str]]:
    img = _get_region_image(region, frame)
    thresh = _preprocess_for_ocr(img)
    whitelist = "0123456789."
    if not allow_decimal:
        whitelist = "0123456789"
    config = f"--oem 3 --psm 7 -c tessedit_char_whitelist={whitelist}"
    text = pytesseract.image_to_string(thresh, config=config).strip()

    cleaned = (
        text.replace("O", "0")
        .replace("o", "0")
        .replace(",", ".")
        .replace("..", ".")
    )
    digits = "".join(ch for ch in cleaned if ch.isdigit())
    return cleaned or None, digits or None


def read_speed(region: Tuple[int, int, int, int], frame: Optional[Image.Image] = None) -> Optional[float]:
    global last_speed, last_speed_time

    raw_text, digits = _read_number_raw(region, frame, allow_decimal=False)
    _update_debug("speed", raw_text=raw_text, digits=digits, region=region)

    if not raw_text:
        _update_debug("speed", parsed_value=None, note="empty")
        return None

    try:
        value = float(raw_text)
    except ValueError:
        _update_debug("speed", parsed_value=None, note="parse-failed")
        return None

    now = time.time()
    if last_speed is not None and last_speed >= 60 and value < 10 and (now - last_speed_time) < 1.5:
        corrected = float(f"7{int(value)}")
        _update_debug("speed", note=f"corrected {value}->{corrected}")
        value = corrected

    if not (0 <= value <= 120):
        _update_debug("speed", parsed_value=None, note="out-of-range")
        return None

    last_speed = value
    last_speed_time = now
    _update_debug("speed", parsed_value=value, note="ok")
    return value


def read_distance(region: Tuple[int, int, int, int], frame: Optional[Image.Image] = None) -> Optional[float]:
    raw_text, digits = _read_number_raw(region, frame, allow_decimal=True)
    _update_debug("distance", raw_text=raw_text, digits=digits, region=region)

    if not raw_text:
        _update_debug("distance", parsed_value=None, note="empty")
        return None

    corrected: Optional[float] = None
    if digits:
        if len(digits) == 3:
            corrected = float(f"{digits[0]}.{digits[1:]}")
            _update_debug("distance", note=f"3-digit {digits} -> {corrected}")
        elif len(digits) == 2:
            corrected = float(f"0.{digits}")
            _update_debug("distance", note=f"2-digit {digits} -> {corrected}")

    if corrected is None:
        try:
            corrected = float(raw_text)
        except ValueError:
            _update_debug("distance", parsed_value=None, note="parse-failed")
            return None

    distance_history.append(corrected)
    if len(distance_history) > 5:
        distance_history.pop(0)

    if len(distance_history) < 3:
        _update_debug("distance", parsed_value=corrected, note="warmup")
        return corrected

    median = sorted(distance_history)[len(distance_history) // 2]
    if abs(corrected - median) > 0.2:
        _update_debug("distance", parsed_value=None, note="jump")
        return None

    _update_debug("distance", parsed_value=corrected, note="ok")
    return corrected


def should_press_door_keys(frame: Optional[Image.Image] = None) -> bool:
    img = _get_region_image(REGION_DOOR_STATUS, frame)
    text = pytesseract.image_to_string(img, config="--oem 3 --psm 6").strip().lower()
    clean_text = re.sub(r"[^a-z ]", "", text)

    if "door" in clean_text and "closed" in clean_text:
        print(f"[✓] Detected 'Door Closed' in: '{text}'")
        return True

    print(f"[x] 'Door Closed' not detected: '{text}'")
    return False


def check_select_destination_trigger(
    now: float,
    frame: Optional[Image.Image] = None,
    interval: float = SELECT_DESTINATION_INTERVAL,
) -> None:
    global last_select_check_time
    if now - last_select_check_time < interval:
        return

    img = _get_region_image(REGION_SELECT_DESTINATION, frame)
    text = pytesseract.image_to_string(img, config="--oem 3 --psm 6").strip().lower()

    if "select destination" in text:
        print("[✓] 'Select Destination' detected — running full sequence")
        import pyautogui
        from pynput.keyboard import Controller

        # Click 57,189 instead of ESC-R-Enter sequence
        pyautogui.click(x=57, y=189)
        time.sleep(3)

        # Wait 1 sec
        time.sleep(1.0)

        # Click 1169,796 → wait 5s
        pyautogui.click(x=1169, y=796)
        time.sleep(5.0)

        # Click 141,495 → wait 0.1s → 145,941 → 0.1s → 2484,1414
        pyautogui.click(x=141, y=495)
        time.sleep(0.1)
        pyautogui.click(x=145, y=941)
        time.sleep(0.1)
        pyautogui.click(x=2484, y=1414)

        # Wait 5s
        time.sleep(5.0)

        # Click 273,297 → 0.1s → 911,417 → 0.1s → 1668,1421
        pyautogui.click(x=273, y=297)
        time.sleep(0.1)
        pyautogui.click(x=911, y=417)
        time.sleep(0.1)
        pyautogui.click(x=1668, y=1421)

        # Wait 10s
        time.sleep(10.0)

        # Hold P for 3s
        keyboard = Controller()
        keyboard.press("p")
        keyboard.press("o")
        time.sleep(3.0)
        keyboard.release("p")
        keyboard.release("o")

        # Final click 1964,1379
        pyautogui.click(x=1964, y=1379)
    else:
        print("[ ] 'Select Destination' not detected")

    last_select_check_time = now
