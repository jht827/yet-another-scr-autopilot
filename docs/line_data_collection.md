# Line Data Collection Tool

Use the OCR collector to build line data from live HUD fields when no official LineData files exist.
The collector watches the HUD for speed, speed limit, signal ID, next stop name, platform, and
distance-to-next-station, then integrates speed over time to estimate distance between nodes.

## Quick Start
1. Update the global config (window title + ROI values):
   - `config/line_data_config.json`
2. (Recommended) Use the ROI boxing tool to select areas on the Roblox window:
   ```bash
   python tools/roi_boxer.py --config config/line_data_config.json
   ```
   If `tkinter` is unavailable, run the CLI fallback which saves a snapshot and lets you type
   the ROI coordinates:
   ```bash
   python tools/roi_boxer.py --mode cli --snapshot recordings/roi_snapshot.png
   ```
3. Run the collector:
   ```bash
   python tools/line_data_collector.py \
     --config config/line_data_config.json \
     --output recordings/line_data_events.csv
   ```
   While collecting, use hotkeys for manual markers:
   - `P` toggles platform start/end markers.
   - `2`-`6` record stop marker positions.
   - `9` records the `S` stop marker.

## Output
The CSV output logs an event when a watched field changes (signal ID, next stop, platform).
Each event includes:
- Timestamp (UTC)
- Event type (which field changed)
- Value (OCR result)
- Distance since the last event (meters, computed by integrating speed)
- Full snapshot of HUD fields at the time of the event

Use the event distances to assemble `SIG`, `DIST`, and `STOP` nodes for LineData files.

## ROI Config Notes
- `roi` is `[left, top, right, bottom]` in screen pixels.
- `scale` is an integer multiplier for OCR readability.
- `threshold` applies a simple binary threshold for numeric fields.
- `whitelist` limits OCR characters for speed and stability.
- `window_title` must match the Roblox window title so capture is scoped to that window.
  The matcher looks for the title as a substring (case-insensitive).
- `window_bbox` can be set manually as `[left, top, right, bottom]` if window lookup fails.
  If the window cannot be found, the collector prompts before falling back to full-screen
  coordinates.

## Dependencies
- `pytesseract` + Tesseract installation for OCR.
- `pygetwindow` for window-scoped capture.
- `tkinter` for the ROI boxing UI (optional; CLI fallback works without it).

## Next Steps
- Add a review step to merge OCR noise into clean IDs and stop names.
- Add a post-processor that converts the event CSV into LineData text format.
