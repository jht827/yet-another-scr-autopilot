### file: utils.py
DECELERATION = 1.55
HEADTIME = -0.25

def braking_distance(speed_mps):
    return speed_mps**2 / (2 * DECELERATION) + speed_mps * HEADTIME
