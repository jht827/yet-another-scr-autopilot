"""Configuration for OCR monitor."""

# Screen regions (left, top, right, bottom)
REGION_SPEED = (2144, 1310, 2206, 1341)
REGION_MILES = (180, 1378, 222, 1393)
REGION_NEXT_SIGNAL_NUMBER = (269, 1375, 330, 1394)
REGION_NEXT_SIGNAL_DISTANCE = (286, 1398, 305, 1417)
REGION_PLATFORM_NUMBER = (94, 1403, 121, 1418)
REGION_ROUTE_NUMBER = (182, 1293, 228, 1308)
REGION_NEXT_STATION = (17, 1354, 244, 1370)

# OCR settings
TESSERACT_CMD = None  # Set to full path if tesseract is not on PATH.
TESSERACT_PSM = 7
TESSERACT_WHITELIST = "0123456789"
ALPHANUM_WHITELIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

# Performance tuning
TARGET_FPS = 30  # Desired loop frequency.
PRINT_ON_CHANGE_ONLY = True
STATUS_EVERY_SECONDS = 2.0  # Print status even if unchanged.
SLOW_READ_INTERVAL = 5.0  # Seconds between slow OCR reads.

# Preprocessing
THRESHOLD = 160  # 0-255; tweak for better OCR.
USE_RED_DETECTION = True  # Enable red-on-black digit detection.
RED_THRESHOLD = 40  # Higher means stricter red detection.

# OCR cleanup heuristics
MAX_DISTANCE_DIGITS = 3  # Distance should be 3 digits (e.g., 061).

MIN_SPEED_STABLE_FOR_RESET_SEC = 10.0  # Allow distance reset after prolonged stop.
STOP_SPEED_THRESHOLD = 1  # Consider stopped if speed <= this value.

MAX_SPEED_DELTA_PER_SEC = 15.0  # Speed should change roughly linearly.
SINGLE_DIGIT_SPEED_IF_PREV_HIGH = 80  # Reject single digit if previous >= this.
SINGLE_DIGIT_CONFIRM_FRAMES = 2  # Accept single digit only after N repeats.

# Distance prediction (high-speed tolerant)
SPEED_LAG_SEC = 0.2  # Approximate delay of speed readout vs distance.
DISTANCE_BASE_TOLERANCE = 0.8  # Base tolerance for OCR distance vs prediction.
DISTANCE_TOLERANCE_PER_MPH = 0.01  # Extra tolerance per mph of speed.

# OCR misread fixups (distance)
DISTANCE_07X_FIX_LOOKBACK = 20  # Lookback window for detecting 07X misreads.
DISTANCE_03X_REQUIRED = 2  # Require this many 03X readings in the window to fix 07X -> 02X.

# Correction backoff
MAX_SPEED_REJECT_FRAMES = 4  # Allow realtime speed after too many rejections.
MAX_DISTANCE_REJECT_FRAMES = 2  # Allow realtime distance after too many rejections.
