### file: brake_control.py
import time
from pynput.keyboard import Controller

keyboard = Controller()

def issue_brake():
    print("[!] BRAKE NOW — Pressing A + S for 1.5s")
    try:
        keyboard.press('e')
      #  keyboard.press('s')
        time.sleep(2.1)
    finally:
        keyboard.release('e')
     #   keyboard.release('s')
        print("[✓] Released A + S")

def release_brake_keys():
    keyboard.release('a')
    keyboard.release('s')
