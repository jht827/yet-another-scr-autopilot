import pytesseract
from PIL import Image, ImageGrab, ImageOps
import time
import re
import pyautogui
from pynput.keyboard import Controller

from config import (
    REGION_DOOR_STATUS,
    REGION_SELECT_DESTINATION,
    SELECT_DESTINATION_INTERVAL,
    DEBUG_OCR,
)

# OCR settings aligned with discarded/3rdgen logic
TESSERACT_SCALE = 2
TESSERACT_THRESHOLD = 160
TESSERACT_INVERT = False
TESSERACT_LANG = "eng"
TESSERACT_PSM = 7
TESSERACT_OEM = 1

NUMERIC_WHITELIST = "0123456789"

_last_ocr_debug = {}


def _record_ocr_debug(label, **data):
    if not DEBUG_OCR:
        return
    entry = _last_ocr_debug.get(label, {})
    entry.update(data)
    entry["timestamp"] = time.time()
    _last_ocr_debug[label] = entry


def get_last_ocr_debug(label):
    return _last_ocr_debug.get(label, {})


def grab_fullscreen():
    return ImageGrab.grab()


def crop_region(frame: Image.Image, region):
    return frame.crop(region)


def preprocess_for_tesseract(
    image: Image.Image,
    threshold=TESSERACT_THRESHOLD,
    invert=TESSERACT_INVERT,
    scale=TESSERACT_SCALE,
):
    if scale > 1:
        image = image.resize(
            (image.width * scale, image.height * scale),
            resample=Image.NEAREST,
        )
    gray = ImageOps.grayscale(image)
    if invert:
        gray = ImageOps.invert(gray)
    if threshold is not None:
        gray = gray.point(lambda p: 255 if p > threshold else 0)
    return gray


# 通用 OCR 读取函数
def read_number_raw(region, frame=None, whitelist=NUMERIC_WHITELIST, label="unknown"):
    img = crop_region(frame, region) if frame is not None else ImageGrab.grab(bbox=region)
    processed = preprocess_for_tesseract(
        img,
        threshold=TESSERACT_THRESHOLD,
        invert=TESSERACT_INVERT,
        scale=TESSERACT_SCALE,
    )
    config = (
        f"--psm {TESSERACT_PSM} --oem {TESSERACT_OEM} "
        f"-c tessedit_char_whitelist={whitelist}"
    )
    text = pytesseract.image_to_string(processed, config=config, lang=TESSERACT_LANG).strip()
    text = text.replace("O", "0").replace("o", "0")
    digits = "".join(c for c in text if c.isdigit())
    attempts = [
        {
            "raw_text": text,
            "digits": digits,
            "threshold": TESSERACT_THRESHOLD,
            "invert": TESSERACT_INVERT,
            "psm": TESSERACT_PSM,
        }
    ]

    if digits:
        _record_ocr_debug(
            label,
            raw_text=text,
            digits=digits,
            region=region,
            whitelist=whitelist,
            attempts=attempts,
        )
        return digits

    _record_ocr_debug(
        label,
        raw_text=text,
        digits="",
        region=region,
        whitelist=whitelist,
        attempts=attempts,
    )

    return None


last_speed = None
last_speed_time = 0


def read_speed(region, frame=None):
    global last_speed, last_speed_time
    raw_text = read_number_raw(region, frame=frame, whitelist=NUMERIC_WHITELIST, label="speed")

    try:
        value = float(raw_text) if raw_text is not None else None
    except Exception:
        _record_ocr_debug("speed", parsed_value=None, note="parse_error")
        return None
    if value is None:
        _record_ocr_debug("speed", parsed_value=None, note="no_digits")
        return None

    now = time.time()

    # Correction logic for probable OCR truncation
    if last_speed is not None and last_speed >= 60 and value < 10 and (now - last_speed_time) < 1.5:
        corrected = float(f"7{int(value)}")
        _record_ocr_debug("speed", note=f"corrected_from_{value}_to_{corrected}")
        value = corrected

    if not (0 <= value <= 120):
        _record_ocr_debug("speed", parsed_value=value, note="out_of_range")
        return None

    last_speed = value
    last_speed_time = now
    _record_ocr_debug("speed", parsed_value=value, note="ok")
    return value


def read_distance(region, frame=None):
    raw_text = read_number_raw(region, frame=frame, whitelist=NUMERIC_WHITELIST, label="distance")
    if raw_text is None:
        _record_ocr_debug("distance", parsed_value=None, note="no_digits")
        return None
    try:
        value = int(raw_text) / 100
    except ValueError:
        _record_ocr_debug("distance", parsed_value=None, note="parse_error")
        return None
    _record_ocr_debug("distance", parsed_value=value, note="ok")
    return value


def should_press_door_keys(frame=None):
    img = crop_region(frame, REGION_DOOR_STATUS) if frame is not None else ImageGrab.grab(bbox=REGION_DOOR_STATUS)
    text = pytesseract.image_to_string(
        img,
        config=f"--psm 6 --oem {TESSERACT_OEM}",
        lang=TESSERACT_LANG,
    ).strip().lower()
    clean_text = re.sub(r'[^a-z ]', '', text)  # remove numbers/symbols

    if "door" in clean_text and "closed" in clean_text:
        return True
    return False


last_select_check_time = 0


def check_select_destination_trigger(now, interval=SELECT_DESTINATION_INTERVAL, frame=None):
    global last_select_check_time
    if now - last_select_check_time < interval:
        return

    img = crop_region(frame, REGION_SELECT_DESTINATION) if frame is not None else ImageGrab.grab(
        bbox=REGION_SELECT_DESTINATION
    )
    text = pytesseract.image_to_string(
        img,
        config=f"--psm 6 --oem {TESSERACT_OEM}",
        lang=TESSERACT_LANG,
    ).strip().lower()

    if "select destination" in text:
        print("[✓] 'Select Destination' detected — running full sequence")

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
        keyboard.press('p')
        keyboard.press('o')
        time.sleep(3.0)
        keyboard.release('p')
        keyboard.release('o')
        # Final click 1964,1379
        pyautogui.click(x=1964, y=1379)

    last_select_check_time = now
