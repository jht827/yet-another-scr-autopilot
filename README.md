# yet-another-scr-autopilot

Project documentation:
- [SCR Autopilot Plan](docs/plan.md)

Developer notes:
- Vision helpers live in `scr_autopilot/vision/` (see `screen_capture.py`, `roi_selector.py`).
- macOS Roblox capture demo: `roblox_capture.py`.
- HUD OCR coordinates and OCR tuning live in the root `scr_autopilot.toml` config.

## HUD reader usage

Run the HUD readout script to print speed/limit/distance from the Roblox window:

```bash
python scr_autopilot/hud_readout.py
```

### Speed OCR with OpenCV templates

The OpenCV template OCR backend only affects the speed readout. It expects a folder of
digit templates (PNG files) where the filename stem is the digit/label (for example
`0.png`, `1.png`, etc.).

```bash
python scr_autopilot/hud_readout.py \
  --speed-ocr opencv \
  --speed-template-dir templates/speed_digits
```

Useful tuning/debug flags:

```bash
python scr_autopilot/hud_readout.py \
  --speed-ocr opencv \
  --speed-template-dir templates/speed_digits \
  --opencv-threshold 160 \
  --opencv-invert \
  --opencv-min-area 30 \
  --opencv-min-height 10 \
  --opencv-debug \
  --debug-dir hud_debug \
  --debug-interval 1.0
```

Debug output includes the thresholded speed ROI, contour overlays, per-digit crops,
and a `speed_matches.txt` summary file inside `hud_debug/opencv_speed_<timestamp>/`.

## Acceleration curve capture

Capture the train acceleration curve (speed vs. time) and write a CSV of speed and
acceleration estimates:

```bash
python scr_autopilot/accel_curve.py --output-csv accel_curve.csv
```

Adjust the sample interval, smoothing window, and OCR settings in `scr_autopilot.toml`.

## Auto braking helper

Compute a braking command from the HUD speed and 3-digit distance readout (where the
last two digits are decimal places, e.g. `300` → 3.00 miles):

```bash
python auto_brake.py --speed-mph 45 --distance-raw 300
```

The script reads a linear braking curve CSV (default:
`train_curves/357/decel_linear.csv`) and exposes tweakable parameters via CLI flags.
