# Marble detection, plate calibration and anti-cheat fixes

Investigation of two reported symptoms — *"it shows a detection with no marble on
the board"* and *"the marble is reported lost when the board tilts"* — plus the
anti-cheat firing on episodes where nothing was cheating.

All numbers below are measured on this rig, not estimated. Reproduction commands
are given so each claim can be re-checked.

---

## 1. Plate pose: fixed-frame geometry was wrong

`PlatePoseEstimator.MODEL_POINTS_FIXED_CORNERS` assumed the four fixed frame dots
spanned `L_EXT_INT_X + 2r = 325 mm` in x and sat **50 mm** inset in y (span
174 mm), a ratio of 1.868. The undistorted image gives a ratio of **1.6357**.

That mismatch drove `camera_localization()` into a wrong PnP minimum.

| | before (317 / 50 mm) | after (305 / 41.3 mm) |
|---|---|---|
| fixed-marker anisotropy | 12.4% | **0.02%** |
| PnP reprojection (mean/max) | 4.587 / 7.500 px | **0.369 / 0.388 px** |
| camera x in `T__W_C` | 0.015 m | **0.151 m** (board centre ≈ 0.157) |
| off-axis rotation term | 0.2584 | **0.0195** |
| resting `beta` | −10.63° | **+3.22°** |

The outer frame span was **measured as 305 mm** (was 317). The y inset was then
solved from the image: two independent methods agree — the analytic span-ratio
solve gives 41.32 mm, and sweeping the inset for minimum reprojection bottoms out
at 41.3 mm.

**Independent confirmation:** `markers.csv` predicts a resting
`alpha = −0.20°, beta = +3.22°` after the fix; the live system then measured
**−0.65° / +3.99°**. Two separate paths agreeing.

Changed: `core/plate_pose.py` — `L_EXT_INT_X = 0.305`, new named constant
`FIXED_DOT_INSET_Y = 0.0413` replacing two hardcoded `0.05` literals.

The moving markers were already correct (`C2C_X/C2C_Y`, reprojection 0.497 px,
anisotropy 1.0%) and were not touched. Note `L_EXT_INT_X` also sizes the
processing mask in `measurements.py`, which narrows 325 → 313 mm in x.

**Falsifiable prediction:** the four fixed dots should measure **313.0 mm**
centre-to-centre horizontally and **191.4 mm** vertically. If the vertical reads
~174 mm, 305 is not the right x span.

---

## 2. AI ROI was fixed in image space — the tilt loss

The AI detector's ROI (`ai_roi_x/y_min/max`) is normalised **image** coordinates,
so it did not move as the plate tilted. Measured margin between the board's
corner dots and that rectangle:

```
ROI    cols 160-460   rows  60-320
board  cols 170-453   rows  66-319
margin left +10px  right +7px  top +6px  bottom +1px
```

Past roughly 10° of tilt the raised board edge moves ~9–10 px in the image (it
comes ~35 mm closer to a camera 550 mm away), which exceeds that margin. The
heatmap is masked to `-1e9` outside the ROI **before** argmax
(`ai_marble_common.decode_heatmap`), so a marble in the excluded strip is
undetectable and the reported confidence is just background.

Measured loss rate versus tilt, and after making the ROI follow the tracked
corner dots:

| tilt | fixed ROI | ROI follows board |
|---|---|---|
| 0–6° | 0.00% (n=129) | — |
| 6–8° | 2.83% (n=530) | — |
| 8–10° | 16.96% (n=230) | — |
| 10–12° | **77.89%** (n=1420) | **16.43%** (n=1601) |
| 12–14° | 77.19% (n=57) | **14.95%** (n=107) |
| 14–20° | **85.31%** (n=143) | **2.98%** (n=973) |

Overall loss **52.8% → 15.0%**, and `corr(lost, tilt)` flipped from **+0.578 to
−0.308** — the tilt dependence is gone.

Changed:
- `ai_marble_common.py` — `detect()` accepts a per-frame `valid_roi` override.
- `core/detection.py` — `_ai_roi_from_corners()` derives the ROI from the tracked
  corner dots each frame. Returns `None` when corners are missing, so a
  lost-corner frame falls back to the static ROI rather than silently widening
  the searchable region out to the robot arm.
- New params: `ai_roi_follows_corners` (default `True`),
  `ai_roi_corner_inset_px` (default `0.0`).

### Why the ROI is not inset

An inset was the first attempt and it **fails**. A marble travelling along the
top edge sits at the same image *rows* as the top corner dots — observed marble
at row 70, dots at rows 66–71 — because they are separated horizontally, not
vertically. No rectangle can include one and exclude the other.

The blue dots are the same colour family as the marble, so they are masked
separately as **discs** (`ai_corner_mask_radius_px`, default 12 px). This gives
the AI the protection `detect_ball()` already had via `mask_corner=True`.

---

## 3. Colour ranges overlap (not yet changed)

```python
DEFAULT_HSV_CORNERS = ((43, 140), (125, 255), (9, 255))   # hue 43-140
DEFAULT_HSV_BALL    = ((60, 116), (162, 255), (50, 243))  # hue 60-116
```

**The marble's entire hue range sits inside the corner range.** This is why
`detect_ball` needs `mask_corner=True` and why `hybrid_ball.py` refuses to trust
an HSV-only candidate. The marble measures BGR `[55,21,0]` → hue ≈ 108.

Scene occupancy: walls yellow/orange (~8–20), LED border pink/red (~0, ~150–179),
surface white (low sat), holes dark brown (low val). **Green (~60) is the only
free slot.** A disjoint split would be corners `(45, 75)` and ball `(95, 125)`.

These are hardcoded constants, *not* ROS parameters — recolouring the dots
requires editing `core/detection.py`. Changing one range does not affect the
other. Practical risk: the pink LED wash tints the dots (they currently read
`[149,125,251]`), so test one dot before repainting all four.

---

## 4. Playable-area gate: dead parameters, and a residual +y bias

`playable_half_x/y` were hardcoded to `0.14`/`0.13`, so `playable_width`,
`playable_height` and `playable_edge_tolerance` did nothing. An older build
artifact showed this *had* been computed from the parameters with
`tolerance = 0.015` before being replaced by literals.

Restored, with a **per-axis** tolerance because the error is not symmetric:

- **x is healthy** — p0.5 span 237.6 mm vs 242.0 mm reachable; 0.05% of samples
  beyond the board half-width. Tolerance 5 mm.
- **y is biased** — a marble on the top edge reports `y = +0.126…+0.129` where
  the board half-height is **0.1145**, i.e. 12–15 mm above a physically
  impossible position (a 6 mm-radius marble cannot get closer than 8.5 mm to the
  playable edge). Tolerance 20 mm.

After the ROI fix, all six remaining dropouts in a 60 s recording were
`lost_outside`, each clipped by only **0.1–3.8 mm** against the old 0.1295 gate,
with x nowhere near its limit. Raising the y tolerance 15 → 20 mm clears them.

> **This is a mitigation, not a fix.** The y tolerance being 4× the x tolerance
> exists solely to absorb the bias. The bias survived the frame-geometry fix, so
> it is not the fixed-corner model.
>
> **Open item — needs a physical measurement.** Hold the marble at known
> positions (each corner of the playable area, plus centre) and record reported
> vs actual. That separates a scale error from a plane-height error from a
> tilt-coupled error, and would let the y tolerance come back down to match x.

---

## 5. Anti-cheat: nothing on this board can be shortcut

A wall-crossing check over all 157 wall segments (inflated by ball + wall radius),
for hop distances of 30, 40 and 60 mm:

```
EXPLOITABLE shortcuts (hop <= 60mm, no wall between):
  none -- every near-self-approach is blocked by a wall
```

The path does return near itself — index 5900 (81.8%) passes **23.5 mm** from
index 3348 (46.4%), which would skip 510 mm of course — but a wall blocks it
every time. So a progress shortcut is physically impossible here, and **every
anti-cheat trigger this system ever produced was a bad state estimate.**

The trigger investigated in detail (`0.8% → 4.9%`, i.e. index 57 → 353, 58.3 mm)
matched a detector flip:

| | position |
|---|---|
| `pts[57]` (the "from") | (0.1271, 0.2196) |
| the real marble | (0.1353, 0.2268) — **11 mm away** |
| `pts[353]` (the "to") | (0.0696, 0.2101) |
| a false lock at pixel (235,76) | (0.0583, 0.2221) — **17 mm away** |

Pixel (235,76) is blank white board surface, BGR `[255,230,250]`, where the
surface meets the pink LED glow — while the real marble (teal, BGR `[55,21,0]`)
sat 83 px away at (318,70).

### Changes to `env_tcp.py`

1. **Termination off by default.** `CYBERRUNNER_ANTICHEAT_ENABLED=0`. Implausible
   jumps are still logged and still **denied progress credit**, so a glitch
   cannot earn reward — it just no longer kills the episode or applies −0.5.
   Set `=1` to restore termination on a board where shortcuts are reachable.
2. **Confirm-before-terminate** (`CYBERRUNNER_ANTICHEAT_CONFIRM_STEPS`, default
   5). Sized from recorded speeds: runs above 1.0 m/s lasting ≥5 steps occur once
   per ~7500 samples (~1 per 160 episodes) versus once per ~2900 at 3 steps —
   the runs cluster, so 3 was too few. `prev_pos_path` is left alone during
   strikes so a one-frame flip that reverts resumes from the real position.
3. **Time-scaled budget:** `min(57 mm, max(10 mm, 1.0 m/s × step_dt))`. Measured
   marble speed is p50 0.024 / p95 0.122 / p99 0.277 m/s, so 1.0 m/s is ~4× p99.
   The old fixed 57 mm/step meant 2.0 m/s at 35 fps but only 0.86 m/s at the
   15 fps this rig dips to. The 10 mm floor matches the checkpoint pass radius
   and only binds above ~100 fps.
4. **`prev_pos_path` resync after an occlusion gap.** `was_occluded` was captured
   and used only for a log line; the gap's travel was charged to one step.
   *(Dormant unless a non-zero grace is configured — see below.)*

Verified against the real event: old code terminated with −0.5, patched code
ignores it and resumes; a genuine cheat (position stays relocated) still
terminates.

---

## 6. Ball-loss grace is already zero

Both `CYBERRUNNER_BALL_LOSS_GRACE_SEC` and `CYBERRUNNER_OCCLUSION_GRACE_SEC`
default to `0.0` and nothing overrides them, so:

```python
within_grace = last_valid_obs is not None and missing_sec < 0.0   # always False
self.ball_detected = within_grace                                 # False
done = not self.ball_detected                                     # True
```

The episode already ends on the **first** missing frame. No change was needed.

This also means the occlusion machinery and the `prev_pos_path` resync above are
dormant on this setup. Worth knowing: measured gaps ran **3.8 s median** (max
7.9 s), so grace could never honestly bridge them anyway — any grace long enough
would mean feeding the policy seconds of `ball_velocity`-extrapolated position.

---

## 7. Overlay showed a stale ball forever

`overlay_map_view_simple.py` stored `self.latest` and never expired it, and the
draw path only checked `isnan`. An estimator that stalled, crashed or restarted
looked exactly like a marble sitting still.

Added `STALE_AFTER_SEC = 0.25`; the panel now shows `NO DATA` plus the silence
duration instead of a frozen ball.

---

## 8. Two workspaces shadow each other

`~/cyberrunner_ws` is sourced from the shell profile and sits **before** this
repo in `AMENT_PREFIX_PATH`. So:

```
ros2 run cyberrunner_state_estimation estimator_ai_map
```

silently resolves the **other** workspace's executable — different node name, no
AI topics, none of this repo's code — while logging a normal-looking startup.

Use the existing launchers, which `unset` the inherited overlay so it cannot win:

```
./run_estimator_cyber.sh --ros-args -p ai_mode:=hybrid -p ai_model_path:=.../marble_detector.onnx
./run_ai_marble_detector.sh
```

`markers.csv` also differs between the two workspaces, so ad-hoc scripts pick up
the wrong marker positions and calibration file.

---

## 9. Model A/B: v5_moredata vs production

Both models over all 749 labelled frames, production ROI:

| | current (prod) | v5_moredata |
|---|---|---|
| detection rate @0.90 (634 positives) | 100.00% | 99.84% |
| localisation error, median | 1.54 px | **1.33 px** |
| localisation error, p90 | 3.37 px | **2.76 px** |
| within 5 px | 98.7% | **99.4%** |
| false positives @0.90 (115 negatives) | 0.87% | 0.87% |
| negative confidence, median | 0.3129 | **0.0102** |

v5 is modestly better on localisation with a 30× lower baseline confidence on
empty frames. **Caveat: v5 trained on these 749 images**, so this is train-set
performance and biased in its favour. `marble_detector_v3_holdout.onnx` suggests
held-out splits have been used before; that is the test that would settle it.

The gap that matters: offline localisation error is ~1.3 px, but the live false
lock was **83 px**. The marble pressed against the top wall under the pink LED
wash is evidently not represented in the training set. That is a labelling gap,
not an epochs gap.

v5 was **not** promoted. `models/marble_detector.onnx` is untouched. To test:

```
CYBERRUNNER_AI_MODEL=$PWD/models/marble_detector_v5_moredata.onnx ./run_estimator_cyber.sh
```

---

## Reproducing the measurements

```bash
# loss rate vs tilt (records estimate + ball_source together)
python3 record_tilt.py out.csv 60

# which source accompanies each dropout
ros2 topic echo /cyberrunner_state_estimation/ball_source

# empty-board false-positive rate
ros2 topic echo /cyberrunner_state_estimation/ai_confidence

# marker geometry / PnP residuals
#   undistort markers.csv through the OcamModel, solve PnP against
#   MODEL_POINTS_FIXED_CORNERS and MODEL_POINTS_CORNERS, compare side scales
```

`ball_source` values worth recognising:

| value | meaning |
|---|---|
| `fused` | HSV and AI agree within 12 px — healthy |
| `kalman_occlusion` | no measurement; Kalman predicting |
| `lost_uncertain` | prediction std exceeded `ai_max_prediction_std_m` |
| `lost_outside` | position rejected by the playable-area gate |
| `lost` | tracker gave up after the occlusion grace |
| `ai_hole_pending_N` / `ai_hole_rejected_N` | timed hole rejector |

---

## Open items

1. **The +y bias** (§4) — top priority. Needs the marble held at known positions.
   Everything else is working around it.
2. **Label frames with the marble at the board edges** under the pink LED wash
   (§5, §9) — the false-lock failure mode is unrepresented in the dataset.
3. **Recolour the corner dots** to a disjoint hue (§3) — optional hardening now
   that the dots are disc-masked.
4. **The y-tolerance bump is unverified live** (§4) — the confirming run had the
   marble 12 mm inside the top edge, so the new tolerance was never exercised.
   Needs a run with the marble tracking the top edge.
