# CyberRunner state estimator

This is **the** state estimator. It reports the marble's position on the maze and
the board's tilt angles, and publishes on `/cyberrunner_state_estimation/*`.

The older HSV + omnidirectional-camera pipeline (`estimator`, `estimator_sub`)
has been removed — see [What was removed](#what-was-removed).

## How it works

Two independent measurements per frame:

- **Marble position.** A learned ONNX detector finds the marble in the image.
  The four blue dots on the moving plate give a fresh pixel-to-metres homography
  every frame, so the map follows the plate as it tilts. HSV is used *only* to
  track the eight reference dots, never as a marble fallback.
- **Board angles.** `solvePnP` on the moving dots gives the plate pose; the four
  dots on the fixed outer frame give the world reference it is measured against.
  (Verified fixed: over 119 frames spanning the full tilt range they move 0.9 px
  peak-to-peak, versus 15.1 px for the moving dots.)

Marble position depends on exactly four things — the four dot pixel positions,
`MOVING_MARKER_SPACING_X_M/Y_M`, the PnP camera height, and
`marble_radius_m`/`marker_plane_height_m` for parallax. It does **not** use the
camera intrinsics, the hole layout, or the fixed markers. Angles additionally
depend on the intrinsics and the fixed-frame geometry.

## Running it

```bash
# 1. camera
python3 fast_camera_publisher_v2.py

# 2. estimator  (add -p show_image:=true for the annotated camera window)
./run_ai_map_estimator.sh

# 3. optional: map overlay
python3 overlay_map_view_simple.py
```

Every parameter defaults to its calibrated value, so pass nothing unless you mean
to override. Health check:

```bash
ros2 topic hz /cyberrunner_state_estimation/estimate           # expect ~40 Hz
ros2 topic echo /cyberrunner_state_estimation/status --once    # expect: valid
```

## Topics

| topic | contents |
|---|---|
| `estimate` | centered marble x/y, x/y velocity, board alpha/beta |
| `estimate_subimg` | the same state plus a 64x64 marble crop (what the RL env consumes) |
| `position_map` | lower-left map x/y, AI confidence in z |
| `valid` | true only when both marker quads, the board pose and the marble detection are all valid |
| `status` | validity or rejection reason |
| `ai_confidence` | raw AI confidence |
| `camera_height_m` | PnP camera height |

On marble loss, position and velocity go `NaN` immediately and `valid` goes
false — no stale position is ever published. A detection inside a hole is
rejected on the same frame (`hole_rejection_delay_sec=0.0`).

Set `publish_legacy_topics:=false` to publish under `/cyberrunner_ai_map/*`
instead, e.g. to A/B two estimators. Do not run two publishers on the production
prefix at once.

## Calibration

Three separate things, in the order you should do them.

### 1. Markers — whenever the camera or board moves

```bash
./run_select_estimator_markers.sh
```

Click in this order: fixed frame `F1`-`F4` (lower-left, lower-right, upper-right,
upper-left), then moving board `M1`-`M4` in the same order. `u`/Backspace undo,
`r` restart, `q`/Escape quit. Then rebuild:

```bash
colcon build --packages-select cyberrunner_state_estimation --symlink-install
```

**Clicks only need to be within ~14 px.** `MarkerQuadGuard` snaps to the true dot
centroids on its first frame (`marker_acquire_radius_px`), then tightens to a
per-frame motion gate. Expect the drawn crosses to shift a few pixels off your
clicks at startup — that is the acquisition working, not drift.

The fixed crosses sit *outside* the moving ones horizontally and *inside* them
vertically (313 x 191 mm versus 249 x 222 mm). It looks wrong; it is correct.

Watch which shell you run this from. `install/setup.bash` *appends* to
`AMENT_PREFIX_PATH`, so another workspace sourced earlier wins and
`select_markers` will write `markers.csv` there instead. The node logs the file
it actually loaded, with its mtime, at startup — read that line.

### 2. Camera intrinsics

Record while tilting, then fit offline. Offline keeps it fast and sidesteps any
interactive lag:

```bash
python3 tools/record_frames.py --seconds 90 --out /tmp/calib_frames
# tilt the plate slowly through its full range on both axes while it records
python3 tools/calibrate_camera_holes.py --frames_dir /tmp/calib_frames
```

It uses the 21 maze holes as a planar target — their metric positions are known
from the DXF layout, and the dot homography predicts where each should be, so
correspondences are solved before any blob detection runs.

**Check two numbers before installing: RMS reprojection under 1 px, and camera
height within 5% of your measured lens-to-dot-plane distance.** It warns on both.
Then rebuild as above; `pinhole_calib.json` is picked up automatically. Without
it the node falls back to `f = 300 px` and no distortion.

Useful flags: `--num_dist_coeffs 2` for `k1+k2`, `--free_principal_point`, or omit
`--frames_dir` to run live against the camera instead.

### 3. Marker geometry — only if positions look scaled

```bash
python3 tools/measure_hole_layout.py --frames_dir /tmp/calib_frames   # verify
python3 tools/fit_marker_geometry.py --frames_dir /tmp/calib_frames   # fit
```

`measure_hole_layout` checks the layout file against the installed maze.
`fit_marker_geometry` jointly fits the dot spacing and the dot-plane height.
They have to be fitted together: with the camera nearly overhead the two are
algebraically degenerate, and only a range of plate tilts separates them. The fit
reports the camera's board-frame sweep so you can see whether it had the leverage
(about 99 x 90 mm is plenty; a few mm is not).

## Diagnosing

**`status` tells you why a frame was dropped.** Tally it rather than guessing:

```bash
ros2 topic echo /cyberrunner_state_estimation/status
```

- `fixed_marker_jump_rejected` / `moving_marker_jump_rejected` — the guard refused
  a marker substitution. Persistent means the seeds are wrong; re-select markers.
- `moving_markers_timeout` — the moving quad was lost past the grace period,
  usually at extreme tilt.
- `ai_marble_missing` — the detector did not find the marble.
- `ai_hole_rejected_N` — the detection fell inside hole N. Disable with
  `-p hole_rejection_enabled:=false` if you do not want hole logic at all.
- `speed_gate` — implied marble speed exceeded `max_marble_speed_mps`.

**If the marble blinks in the overlay while the AI clearly sees it, suspect QoS,
not detection.** Camera frames are 640x400x3 = 768 KB, which DDS fragments over
many UDP datagrams; under `BEST_EFFORT` one lost fragment discards the whole
frame with no retransmission, and the loss arrives in bursts. Measured with the
camera at 40.6 Hz, a `BEST_EFFORT` subscriber received 13.1 Hz with stalls to
1669 ms — 45% of wall-clock time inside a gap longer than the overlay's 250 ms
staleness cutoff — while every frame it *did* receive was valid at 0.994
confidence. `RELIABLE` gives 38-43 Hz and a 60 ms worst case. Several other
scripts here still subscribe `BEST_EFFORT` and will show the same symptom.

**The overlay's status panel tracks per-axis session extremes.** Roll the marble
into all four walls, then read them: symmetric overshoot on an axis means that
axis's dot spacing is wrong, while one end over by *d* and the other short by *d*
means the dot-quad centroid is offset from the playable centre. The extremes only
reflect where the marble has actually been, so touch every wall first.

## Parameters worth knowing

| parameter | default | note |
|---|---|---|
| `publish_legacy_topics` | `true` | false publishes under `/cyberrunner_ai_map/*` |
| `ai_confidence_threshold` | `0.90` | |
| `marker_plane_height_m` | `0.010` | dot plane above the play surface; ruler and fit agree |
| `marble_radius_m` | `0.006` | |
| `camera_height_m` | `0.29` | fallback only; PnP normally supplies it |
| `marker_acquire_radius_px` | `14.0` | one-shot marker acquisition window |
| `hole_rejection_enabled` | `true` | |
| `show_image` | `false` | annotated camera window |

## Known open items

- Dot spacing is `249.2 x 222.3 mm`, fitted against the DXF hole positions. This
  makes hole residuals 2.19 mm and the reach overshoot 0.2 mm, but leaves PnP
  camera height at 265 mm against a 290 mm ruler. Measuring the leftmost-to-
  rightmost hole centre span (DXF says 240.7 mm) decides which is right.
- Positions are ~7% smaller in radius and angles ~1.9x smaller than before this
  calibration, so RL checkpoints trained earlier are off-distribution.

## What was removed

`estimator` / `estimator_sub` and their chain — `estimation_pipeline.py`,
`measurements.py`, `estimator.py`, `opencv_acceleration.py`, the `utils/`
visualisation helpers, `ocam_model.py` and `calib/calib_results_cyberrunner.txt`.

That calibration file described a 1920x1200 capture while the pipeline runs
1280x720 downscaled to 640x400, so its polynomial did not transfer. Measured
against `markers.csv`, the ocam pass changed the dot spacing by 1.93x and left
both aspect ratios identical to four decimal places — no shape correction at all,
only a wrong magnification. And because `get_pose_T__C_P` ran `solvePnP` with the
same `f` the reprojection used, `f` cancelled and the effective focal length
became the polynomial's on-axis 673.2 px instead of the true ~298, placing the
camera at 0.556 m rather than a measured 0.290 m and scaling every derived angle
by 2.26x. No rescale fixes it: `OcamModel.scale` is angle-preserving, and scaling
its coefficients is algebraically just a change of `f`.

Recover any of it from git history if needed.
