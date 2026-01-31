# yet-another-scr-autopilot

Project documentation:
- [SCR Autopilot Plan](docs/plan.md)

Developer notes:
- Vision helpers live in `scr_autopilot/vision/` (see `screen_capture.py`, `roi_selector.py`).
- macOS Roblox capture demo: `roblox_capture.py`.
- HUD OCR coordinates live in `scr_autopilot/hud_config.py` (update `HUD_ROIS` after boxing).

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
