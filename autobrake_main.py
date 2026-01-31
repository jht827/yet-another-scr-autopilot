# autobrake_main.py

import time
from pynput.keyboard import Listener, Controller
from brake_control import issue_brake, release_brake_keys
from ocr_reader import (
    check_select_destination_trigger,
    grab_fullscreen,
    read_distance,
    read_speed,
    should_press_door_keys,
)

from curve_calc import braking_distance
from config import (
    REGION_SPEED,
    REGION_MILES,
    COOLDOWN,
    TQ_PRESS_INTERVAL,
    W_PRESS_COOLDOWN,
    STOPPED_W_PRESS_DELAY,
    STOPPED_W_PRESS_DURATION,
    DOOR_KEY_PRESS_DURATION,
    LOOP_INTERVAL,
)

# State
keyboard = Controller()
brake_disabled = False
last_trigger_time = 0
last_keypress_time = 0
last_nonzero_speed_time = time.time()
last_w_press_time = 0


def on_press(key):
    global brake_disabled
    try:
        if key.char.lower() == 'r':
            brake_disabled = True
            print("[✱] Auto-brake manually RELEASED")
        elif key.char.lower() == 'e':
            brake_disabled = False
            print("[✓] Auto-brake manually RE-ENABLED")
    except AttributeError:
        pass


listener = Listener(on_press=on_press)
listener.start()

print("Braking monitor running... Press Ctrl+C to stop.")
try:
    while True:
        now = time.time()

        frame = grab_fullscreen()

        check_select_destination_trigger(now, frame=frame)

        # Smash T and Q every 1 second
        if now - last_keypress_time > TQ_PRESS_INTERVAL:
            keyboard.press('t')
            keyboard.release('t')
            keyboard.press('q')
            keyboard.release('q')
            last_keypress_time = now

        # Check for 'Door Closed'
        if should_press_door_keys(frame=frame):
            keyboard.press('w')
            keyboard.press('d')
            time.sleep(DOOR_KEY_PRESS_DURATION)
            keyboard.release('w')
            keyboard.release('d')

        # Speed check
        speed_mph = read_speed(REGION_SPEED, frame=frame)
        if speed_mph is None:
            # Assume unreadable speed is 0
            speed_mph = 0.0

        if speed_mph > 0:
            last_nonzero_speed_time = now

        # If speed has stayed at 0 for over 30s, press W
        if (
            speed_mph == 0
            and (now - last_nonzero_speed_time > STOPPED_W_PRESS_DELAY)
            and (now - last_w_press_time > W_PRESS_COOLDOWN)
        ):
            keyboard.press('w')
            time.sleep(STOPPED_W_PRESS_DURATION)
            keyboard.release('w')
            last_w_press_time = now

        if brake_disabled and speed_mph < 1:
            brake_disabled = False

        if brake_disabled:
            time.sleep(LOOP_INTERVAL)
            continue

        # Distance check
        distance_miles = read_distance(REGION_MILES, frame=frame)
        if distance_miles is None:
            time.sleep(LOOP_INTERVAL)
            continue

        required_miles = braking_distance(speed_mph)

        status_line = (
            f"Speed: {speed_mph:5.1f} mph | Dist: {distance_miles:6.3f} mi | "
            f"Req: {required_miles:6.3f} mi"
        )
        print(f"\r{status_line}", end="", flush=True)

        if distance_miles < required_miles and (now - last_trigger_time) > COOLDOWN:
            print()
            issue_brake()
            last_trigger_time = now

        time.sleep(LOOP_INTERVAL)
except KeyboardInterrupt:
    print()
    print("Stopped. Cleaning up...")
    release_brake_keys()
    listener.stop()
