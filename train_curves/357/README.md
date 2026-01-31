# Class 357 linear curves

These CSVs store linear acceleration/deceleration curves as speed vs. acceleration,
using **miles per second²** for acceleration. The values are intended to be adjusted
after collecting raw curve data for the 357 and fitting a line.

- `accel_linear.csv`: throttle/acceleration curve.
- `decel_linear.csv`: braking/deceleration curve.

Both files expect two endpoints for a linear fit; additional points can be added if
the automation should interpolate a multi-point curve instead.
