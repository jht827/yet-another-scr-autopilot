# SCR Autopilot Plan

## Detailed Software Plan

### Goals
- Build an autopilot for Stepford County Railway (SCR) that can safely transition between **full automation** and **reactive fallback** using Signal IDs and distance anchors instead of flaky OCR.
- Implement a **multi-tiered intelligence model**:
  - **Full Line Data (FLD)**: deterministic driving from LineData files.
  - **Limited Line Data (LLD)**: partial route knowledge with physics-based stopping.
  - **RedCap (Reduced Capability)**: reactive-only mode using HUD speed/signal state.
  - **Upgrade path**: template-matching a signal sequence to select LineData and upgrade to FLD.

### Architectural Overview

#### 1) Core Components
- **Capture Pipeline**
  - Thread A: HUD capture (speed, speed limits, signal state).
  - Thread B: Signal ID box capture for template matching.
  - Shared frame buffer with timestamps to keep data in sync.
- **Perception Layer**
  - Signal ID recognizer using icon/character template matching.
  - HUD parser for speed, current limit, and signal color/state.
  - OpenCV-based OCR-like detection for HUD elements using:
    - ROI templates + normalized cross-correlation for digits and icons.
    - Adaptive thresholding and contour filtering for digit segmentation.
    - Frame-to-frame voting to stabilize real-time readings.
- **World Model**
  - Route state machine with current section, next signal, platform stops.
  - Distance accumulator driven by physics integration (speed * delta_t).
- **Control System**
  - Braking curve generator using \(a = \frac{v^2 - u^2}{2d}\).
  - Throttle and brake output to match target acceleration with response lag.
- **Mode Manager**
  - Determines active intelligence tier (FLD/LLD/RedCap).
  - Handles upgrades/downgrades based on confidence signals and data availability.
- **Vehicle Profile Loader**
  - Train-specific constants: brake force, response lag, acceleration curve.

#### 2) Data Model
Proposed LineData format (example):
```
SECTION BENTONSIDINGS TO BENTON

SIG D308 SHUNT

DIST 300

SIG M209

DIST 500

SIG M210 4ASP PC

DIST 300

SIG M211 3ASP C

DIST 200

PLATFORM BEGIN

DIST 100

STOP 3

DIST 80

STOP 4

DIST 120

STOP 5

STOP 6

DIST 120

STOP S

DIST 10

PLATFORM END

SIG M380
```
- **Nodes**:
  - `SIG <ID> [SIGNAL TYPE] [DEFAULT STATE]`
  - `DIST <meters>`
  - `PLATFORM BEGIN/END`
  - `STOP <marker>`
- **Parsing rules**:
  - `DIST` applies to the previous node and defines its distance to the next node.
  - Multiple `STOP` markers can map to identical distance positions.
  - Missing optional fields implies a 4-aspect signal with a default state of proceed.

#### Signal Command Usage
- **Format**: `SIG [ID] [SIGNAL TYPE] [DEFAULT STATE]`
  - `[SIGNAL TYPE]` is optional (ex: `SHUNT`, `4ASP`, `3ASP`).
  - `[DEFAULT STATE]` is optional (ex: `PC`, `C`).
- **Valid examples**:
  - `SIG D003` (default 4-aspect signal with default state set as proceed)
  - `SIG D004 SHUNT` (shunt signal, proceed)
  - `SIG D005 4ASP PC` (4-aspect signal, proceed-caution default)
  - `SIG D008 3ASP C` (3-aspect signal, caution default)
- **Invalid examples**:
  - `SIG 4ASP` (missing ID)
  - `SIG D009 C` (missing signal type when default state is present)

### Feature Planning

#### Recommended Build Order
1. **Data ingestion + parsing**
   - Implement the LineData parser and validation first to establish the canonical route model.
2. **Instrumentation + data collection tools**
   - Build a braking-curve collection harness per train profile (logging speed, decel, response lag).
   - Build a LineData capture tool to place signals, distances, and platform markers.
3. **Perception layer**
   - Add OpenCV OCR-like parsing and signal ID template matching after the route model exists.
4. **Control loop + safety**
   - Wire in braking curves, response lag modeling, and conservative fallbacks.
5. **Mode manager + upgrades**
   - Enable FLD/LLD/RedCap switching and signal-sequence-based upgrades last.

#### Multi-File LineData Strategy
- **Segmented files**: Allow each file to describe a section or corridor with explicit boundaries.
- **Linking fields**:
  - Add metadata headers to each file: `SECTION`, `START_SIG`, `END_SIG`, `NEXT_SECTION`.
  - Use signal IDs (or platform IDs) as join keys to stitch adjacent sections.
- **Runtime stitching**:
  - Preload all LineData files into a route index keyed by `START_SIG`/`END_SIG`.
  - When FLD detects a boundary signal, automatically load/activate the next section.
  - If the next section is missing, downgrade to LLD/RedCap but keep a soft retry loop.
- **Conflict handling**:
  - If multiple candidate sections share the same boundary signal, rank by recent signal sequence history.
  - Keep a short rolling window of observed signals to disambiguate the correct continuation.

#### Phase 1: Foundations
1. **Telemetry + Capture**
   - HUD capture loop with image normalization.
   - Signal ID ROI capture with fixed crop box.
2. **Template Matching**
   - Offline template library for alphanumeric signal characters.
   - Confidence thresholding and debounce on recognitions.
3. **OpenCV OCR-Like Parsing**
   - ROI-based digit segmentation (adaptive threshold + contours).
   - Template-matched digits for speed, limits, and HUD indicators.
   - Temporal smoothing for stable, real-time readouts.
4. **LineData Parser**
   - Read files into a route graph of ordered nodes.
   - Validate sections and compute cumulative distances.
5. **Physics Core**
   - Speed integration and distance tracking.
   - Braking curve computation from current speed to target speed at distance.

#### Phase 2: Intelligence Modes
1. **FLD Mode**
   - Use LineData signals and distance to preemptively brake.
   - React to signal defaults and platform segments.
2. **LLD Mode**
   - Use known route structure without exact stop markers.
   - Apply “End of Platform” target using Physics Offset.
3. **RedCap (Reduced Capability)**
   - Reactive control based on HUD speed limits and signal color.
   - Conservative braking without map knowledge.
4. **Upgrade Logic**
   - Use detected signal sequences (e.g., D308 → M209) to select LineData.
   - Promote mode to FLD after successful match and validation.

#### Phase 3: Control Tuning
- Train profile selection and calibration (brake force, lag).
- Soft transitions between throttle and brake to avoid jerk.
- Fail-safe logic for uncertainty (treat unknown as restrictive).

### Mode Transition Rules

#### Entry Conditions
- **FLD**: LineData loaded + signal sequence confirmed.
- **LLD**: Section known, but stop markers missing or unreliable.
- **RedCap**: No route data or confidence in signal IDs is too low.

#### Upgrade/Downgrade Criteria
- **RedCap → FLD**:
  - Match a known signal sequence with confidence and timing consistency.
- **FLD → LLD**:
  - LineData mismatch with repeated signal ID failures.
- **Any → RedCap**:
  - Loss of confidence in both LineData alignment and signal IDs.

### Control Algorithm Details

#### Braking Curve
- Use target speed \(u\) (e.g., 0 at stop or new speed limit) and current speed \(v\).
- Compute required acceleration \(a\) for distance \(d\):
  - \(a = \frac{v^2 - u^2}{2d}\)
- Clamp acceleration to train’s braking capabilities and response lag.

#### Response Lag Modeling
- Apply smoothing:
  - `cmd = cmd_prev + (target_cmd - cmd_prev) * (dt / response_lag)`
- Limit jerk by bounding delta command per tick.

### Error Handling and Safety
- If any mismatch is detected, favor **over-braking** and **speed limit compliance**.
- Log all signal ID matches and mode changes with timestamps.
- Provide a manual override hotkey to force downgrade to RedCap.

### Telemetry and Debugging
- Real-time overlay: current mode, next signal, distance to target, brake curve.
- Log format:
  - `[timestamp] MODE=FLD SIG=M209 DIST=480 SPEED=43 LIMIT=40 BRAKE=0.35`
- Offline replay by feeding recorded HUD frames into the perception pipeline.

### Testing Strategy
- **Unit tests**:
  - LineData parser and distance accumulation.
  - Braking curve outputs across a matrix of speeds/distances.
- **Simulation tests**:
  - Mock speed inputs to verify stop accuracy.
  - Randomized signal sequences to validate upgrade logic.
- **Regression tests**:
  - Ensure “platform stop” behavior remains stable after tuning.

### Milestones
1. **MVP**
   - Signal ID detection + RedCap reactive control.
2. **Route Awareness**
   - LineData ingestion + FLD mode.
3. **Robust Fallback**
   - LLD mode + upgrade/downgrade logic.
4. **Polish**
   - UI overlay, logging, and train profile tuning.
