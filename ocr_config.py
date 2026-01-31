"""Configuration for OCR monitor."""

# Screen regions (left, top, right, bottom)
REGION_SPEED = (2144, 1310, 2206, 1341)
REGION_MILES = (180, 1378, 222, 1393)

# OCR settings
TESSERACT_CMD = None  # Set to full path if tesseract is not on PATH.
TESSERACT_PSM = 7
TESSERACT_WHITELIST = "0123456789"

# Performance tuning
TARGET_FPS = 30  # Desired loop frequency.
PRINT_ON_CHANGE_ONLY = True
STATUS_EVERY_SECONDS = 2.0  # Print status even if unchanged.

# Preprocessing
THRESHOLD = 160  # 0-255; tweak for better OCR.
USE_RED_DETECTION = True  # Enable red-on-black digit detection.
RED_THRESHOLD = 40  # Higher means stricter red detection.

# OCR cleanup heuristics
MAX_DISTANCE_DROP_PER_SEC = 2.0  # Maximum allowed drop in distance per second.
MAX_DISTANCE_RISE_PER_SEC = 0.2  # Allow small upward jitter.
MAX_DISTANCE_DIGITS = 3  # Distance should be 3 digits (e.g., 061).

MIN_SPEED_STABLE_FOR_RESET_SEC = 10.0  # Allow distance reset after prolonged stop.
STOP_SPEED_THRESHOLD = 1  # Consider stopped if speed <= this value.

MAX_SPEED_DELTA_PER_SEC = 15.0  # Speed should change roughly linearly.
SINGLE_DIGIT_SPEED_IF_PREV_HIGH = 80  # Reject single digit if previous >= this.
SINGLE_DIGIT_CONFIRM_FRAMES = 2  # Accept single digit only after N repeats.

# Correction backoff
MAX_SPEED_REJECT_FRAMES = 6  # Allow realtime speed after too many rejections.
MAX_DISTANCE_REJECT_FRAMES = 6  # Allow realtime distance after too many rejections.
