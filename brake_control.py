### file: brake_control.py
import time
from pynput.keyboard import Controller

keyboard = Controller()

def issue_brake():
    print("[!] BRAKE NOW — Pressing A + S for 1.5s")
    try:
        keyboard.press('s')
        time.sleep(4.5)
    finally:
        keyboard.release('s')
        print("[✓] Released A + S")

def release_brake_keys():
    keyboard.release('s')
