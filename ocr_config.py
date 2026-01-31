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
