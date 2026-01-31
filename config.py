from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# Screen regions
REGION_SPEED = (804, 1350, 860, 1379)
REGION_MILES = (1168, 1388, 1225, 1409)
REGION_DOOR_STATUS = (1223, 13, 1335, 31)
REGION_SELECT_DESTINATION = (676, 57, 854, 80)

# Unit conversion
MPH_TO_MPS = 0.44704
MILES_TO_METERS = 1609.34

# Timing and input
COOLDOWN = 3.0
TQ_PRESS_INTERVAL = 1.0
W_PRESS_COOLDOWN = 35
STOPPED_W_PRESS_DELAY = 30
STOPPED_W_PRESS_DURATION = 2.5
DOOR_KEY_PRESS_DURATION = 2.0

# Curve settings
DECEL_CURVE_PATH = BASE_DIR / "class_357_decel.csv.csv"
CURVE_SMOOTHING_WINDOW = 5

# Additive distance adjustment (meters) to compensate for short-stop bias
DISTANCE_ADJUST_M = 0.0

# Select Destination automation
SELECT_DESTINATION_INTERVAL = 60
