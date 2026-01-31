### file: ocr_reader.py
import pytesseract
from PIL import ImageGrab
import numpy as np
import cv2
import time

# 通用 OCR 读取函数
def read_number_raw(region):
    img = ImageGrab.grab(bbox=region)
    img_np = np.array(img)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresh = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY)
    config = '--psm 7 -c tessedit_char_whitelist=0123456789O.'
    text = pytesseract.image_to_string(thresh, config=config).strip()

    text = text.replace('O', '0').replace('o', '0').replace(',', '.').replace('..', '.')

    try:
        return text
    except:
        print(f"[x] OCR failed: '{text}'")
        return None

last_speed = None
last_speed_time = 0

def read_speed(region):
    global last_speed, last_speed_time
    raw_text = read_number_raw(region)
    if raw_text is None:
        return None

    try:
        value = float(raw_text)
    except:
        print(f"[x] Failed to parse speed: '{raw_text}'")
        return None

    now = time.time()

    # Correction logic for probable OCR truncation
    if last_speed is not None and last_speed >= 60 and value < 10 and (now - last_speed_time) < 1.5:
        corrected = float(f"7{int(value)}")
        print(f"[~] Corrected likely misread: {value} → {corrected} (based on previous {last_speed:.1f})")
        value = corrected

    if not (0 <= value <= 120):
        print(f"[x] Speed out of range: {value}")
        return None

    last_speed = value
    last_speed_time = now
    return value


distance_history = []

def read_distance(region):
    global distance_history
    raw_text = read_number_raw(region)
    if raw_text is None:
        return None

    digits_only = ''.join(c for c in raw_text if c.isdigit())
    corrected = None

    if len(digits_only) == 3:
        corrected = float(f"{digits_only[0]}.{digits_only[1:]}")
        print(f"[~] Corrected 3-digit '{digits_only}' -> '{corrected}'")
    elif len(digits_only) == 2:
        corrected = float(f"0.{digits_only}")
        print(f"[~] Corrected 2-digit '{digits_only}' -> '{corrected}'")
    else:
        try:
            corrected = float(raw_text)
        except:
            print(f"[x] Failed to parse number: '{raw_text}'")
            return None

    distance_history.append(corrected)
    if len(distance_history) > 5:
        distance_history.pop(0)
    if len(distance_history) < 3:
        return corrected
    median = sorted(distance_history)[len(distance_history) // 2]
    if abs(corrected - median) > 0.2:
        print(f"[x] Distance jump: {corrected:.3f} vs median {median:.3f} — discarded")
        return None
    return corrected

import re

def should_press_door_keys():
    region = (1223, 13, 1335, 31)
    img = ImageGrab.grab(bbox=region)
    text = pytesseract.image_to_string(img, config='--psm 6').strip().lower()
    clean_text = re.sub(r'[^a-z ]', '', text)  # remove numbers/symbols

    if "door" in clean_text and "closed" in clean_text:
        print(f"[✓] Detected 'Door Closed' in: '{text}'")
        return True
    else:
        print(f"[x] 'Door Closed' not detected: '{text}'")
        return False

import pyautogui

last_select_check_time = 0

def check_select_destination_trigger(now, interval=60):
    global last_select_check_time
    if now - last_select_check_time < interval:
        return

    region = (676, 57, 854, 80)
    img = ImageGrab.grab(bbox=region)
    text = pytesseract.image_to_string(img, config='--psm 6').strip().lower()

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
        from pynput.keyboard import Controller
        keyboard = Controller()
        keyboard.press('p')
        keyboard.press('o')
        time.sleep(3.0)
        keyboard.release('p')
        keyboard.release('o')
        # Final click 1964,1379
        pyautogui.click(x=1964, y=1379)

    else:
        print("[ ] 'Select Destination' not detected")

    last_select_check_time = now

