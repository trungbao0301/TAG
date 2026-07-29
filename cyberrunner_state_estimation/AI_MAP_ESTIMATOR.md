# AI map estimator

This estimator uses the learned ONNX detector for the marble. HSV is used only
to track the eight colored reference dots; it is never used as a fallback
marble detector.

## Marker calibration

Run:

```bash
./run_select_estimator_markers.sh
```

Select the dots in this order:

1. Fixed frame: lower-left, lower-right, upper-right, upper-left (`F1`-`F4`)
2. Moving board: lower-left, lower-right, upper-right, upper-left (`M1`-`M4`)

Use `u` or Backspace to undo, `r` to restart, and `q` or Escape to quit.
After all eight clicks, press any key to save
`cyberrunner_state_estimation/markers.csv`. Rebuild after changing markers:

```bash
colcon build --packages-select cyberrunner_state_estimation --symlink-install
```

## Shadow validation

Start the camera, then run:

```bash
./run_ai_map_estimator.sh
```

The default shadow topics do not replace the production estimator:

- `/cyberrunner_ai_map/estimate`: centered marble x/y, marble x/y velocity,
  board alpha/beta
- `/cyberrunner_ai_map/estimate_subimg`: the same state plus a 64x64 marble crop
- `/cyberrunner_ai_map/position_map`: lower-left map x/y; confidence in z
- `/cyberrunner_ai_map/valid`: true only when both marker quads, board pose, and
  the current AI marble detection are valid
- `/cyberrunner_ai_map/status`: rejection or validity reason
- `/cyberrunner_ai_map/ai_confidence`: current raw AI confidence
- `/cyberrunner_ai_map/camera_height_m`: PnP camera height, with 0.20 m fallback

No stale marble position is published after AI loss: position and velocity
become `NaN` immediately and `valid` becomes false. A detection inside a hole
is also rejected on that same frame (`hole_rejection_delay_sec=0.0`).

The geometry assumes a 12 mm marble and a 0.20 m camera-to-board distance.
The four moving dots provide a fresh pixel-to-map homography every frame, and
the marble-center height is compensated before publishing map coordinates.

## Production switch

Only after shadow validation, stop the old estimator and run:

```bash
./run_ai_map_estimator.sh models/marble_detector_v5_moredata.onnx \
  -p publish_legacy_topics:=true
```

This changes the output prefix to `/cyberrunner_state_estimation`, so it must
not run alongside another publisher on those production topics.
