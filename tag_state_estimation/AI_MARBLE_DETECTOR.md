# AI Marble Detector (TAG)

A learned full-camera marble detector that augments the classical HSV state
estimator when HSV struggles: pendulum-arm occlusion, reflections, holes, corner
markers, blue hardware, or objects outside the board. **It never controls
motors** — it only produces a marble pixel detection / confidence, or assists the
existing estimator through guarded fusion + Kalman prediction.

Diagnostic outputs only (no motor/control topics):
- `/tag_ai_marble/pixel` (`geometry_msgs/PointStamped`) and
  `/tag_ai_marble/confidence` (`std_msgs/Float32`) from the standalone node
- `/tag_state_estimation/{ball_source, ai_confidence, detection_disagreement_px}`
  when integrated into the estimator

> ⚠️ **Validation status:** the shipped model has known evaluation weaknesses
> (see *Validation caveats*). Safe in `off` (default) and `shadow`. Do **not**
> enable `hybrid` on the robot until shadow-mode acceptance criteria pass.

---

## Files & model
| File | Purpose |
|---|---|
| `tag_state_estimation/ai_marble_common.py` | ONNX inference (OpenCV DNN, no PyTorch at runtime) + heatmap decode + ROI |
| `tag_state_estimation/core/hybrid_ball.py` | AI-authoritative HSV/AI fusion + confirmed-reacquire state machine |
| `tag_state_estimation/ai_marble_detector_node.py` | Standalone diagnostic node (`ai_detector`) — pixel + confidence only |
| `tag_state_estimation/train_ai_marble.py` / `ai_dataset_labeler.py` | trainer (`ai_train`) + labeler (`ai_labeler`) |
| `models/marble_detector.onnx` | Deployed weights |

**Model:** 320×200 RGB in → 80×50 heatmap out, output stride **4**, ~510 KB.
**SHA-256:** `0a09032fb6a62c680dcc16f1411973aebe7e1d77771e094cfbd828adbdeb154b`

> The AI **library + standalone diagnostic + trainer + model** ship here. The
> full estimator wiring (the `detection.py` / `subimg.py` hooks that drive the
> behavior below) is the deployment target — integrate it only after shadow
> validation.

---

## Modes
- **off** (default): HSV-only; original behavior, byte-for-byte.
- **shadow**: AI runs and publishes diagnostics; HSV still drives the estimator.
  `ai_check_every_n_frames` throttles AI work in this mode.
- **hybrid**: **AI-authoritative.** AI is evaluated **every frame**.

## How hybrid detection behaves
- **Agreeing HSV + AI** (within `ai_agreement_radius_px`, 12 px) → **fused**.
- **AI-only** detection → accepted only through the continuity/confirmation gates.
- **HSV-only** detection (no AI support) → **treated as missing.** This is what
  stops a blue board marker from resetting the marble-loss timer.
- **Disagreement** → AI is authoritative.
- **Reacquisition confirmation:** after *any* missing frame, a replacement
  position must stay spatially consistent for **three frames** before it is
  accepted (both fused and AI-only paths → `fused_reacquired_confirmed` /
  `ai_reacquired_confirmed`). A rejected candidate **cannot** reset the loss timer.
- **Marker masking:** the moving *and* fixed blue reference markers are masked at
  their freshly-detected pixel positions **every frame before HSV** detection. The
  AI detector receives an **unmodified** copy of the image, so it can still find a
  real marble near a marker.
- **HSV crop reset:** after **six** consecutive missing HSV measurements the
  predictive HSV crop resets to a full-board search, while the bounded Kalman
  occlusion grace continues (the same "cover/uncover by hand" recovery, automated).
- **Kalman occlusion:** during `kalman_occlusion` the KF gets a **missing**
  measurement and the node publishes its bounded prediction, only while position
  uncertainty ≤ `ai_max_prediction_std_m` (0.03 m). After **90 frames** or
  excessive uncertainty, the normal finite/NaN loss contract resumes.

**`ball_source` values:** `hsv`, `hsv_hold`, `fused`, `fused_reacquired_confirmed`,
`ai_reacquired`, `ai_reacquired_confirmed`, `kalman_occlusion`, `lost`,
`lost_uncertain`, `lost_outside`.

**Coordinates:** classical detector is `[row, col]`; the AI reports `[x, y]` and is
converted to `[y, x]` before fusion. When lost, the KF receives **NaN** (no fake
repeated measurement).

---

## Reproducible training (stride 4)
The deployed model is **stride 4**; the trainer defaults to stride 8, so you
**must** pass `--output-stride 4`:
```bash
ros2 run tag_state_estimation ai_train ai_marble_dataset \
  --output models/marble_detector.onnx --epochs 60 --output-stride 4
```
Config: input 320×200, 60 epochs, batch 32, lr 3e-4, AdamW,
BCEWithLogitsLoss(pos_weight=20), brightness/contrast + H/V flips, seed 7.
Collect ≥500 images across the full maze, angles, lighting, moving/stationary
marble, arm occlusions, and true ball-loss frames (holes, blue markers, cables,
highlights). The trainer also inpaints one synthetic negative per visible click —
useful, but not a substitute for real `N` (occluded / off-board) examples.

## Safe commands
```bash
# collect labels (left-click=center, N=not visible, Space=freeze, Q=quit)
ros2 run tag_state_estimation ai_labeler ai_marble_dataset

# standalone diagnostic (subscribe-only; publishes only pixel + confidence)
ros2 run tag_state_estimation ai_detector \
  --ros-args -p model_path:=models/marble_detector.onnx

# shadow (after estimator integration; same camera view the model was trained on)
ros2 launch tag_camera camera_estimation_gpu.launch.py \
  ai_mode:=shadow \
  ai_model_path:=/home/trungbao/CYBER/tag/models/marble_detector.onnx

# hybrid (ONLY after shadow criteria pass)
ros2 launch tag_camera camera_estimation_gpu.launch.py ai_mode:=hybrid \
  ai_model_path:=/home/trungbao/CYBER/tag/models/marble_detector.onnx

# immediate rollback to HSV-only
ros2 launch tag_camera camera_estimation_gpu.launch.py ai_mode:=off
```
Inspect diagnostics:
```bash
ros2 topic echo /tag_state_estimation/ball_source
ros2 topic echo /tag_state_estimation/ai_confidence
ros2 topic echo /tag_state_estimation/detection_disagreement_px
```
> ⚠️ The full launch starts **both** a camera and an estimator. Never run it while
> another camera already owns the video device. To run against an existing feed,
> launch the estimator only and remap + pass pose-zero:
> ```bash
> ros2 run tag_state_estimation estimator_sub --ros-args \
>   -r tag_camera/image:=/cyberrunner_camera/image -p ai_mode:=off \
>   -p pose_zero_alpha_deg:=<deg> -p pose_zero_beta_deg:=<deg>
> ```
> Don't run the standalone `ai_detector` at the same time as hybrid mode — both
> perform the same ONNX inference.

---

## Geometry (important)
The model + ROI (`x=[0.25,0.72]`, `y=[0.15,0.80]`) were trained on the CyberRunner
camera view (640×400, that crop/border/board placement). They are valid **only**
if the estimator receives the same resolution, crop, border, and board placement.
A different tag camera/mount requires new labels / ROI recalibration.

## Validation caveats (from the technical review)
- **Train/test leakage:** random per-image split; capture sessions straddle the
  train/test boundary (adjacent ~0.18 s frames are near-duplicates). Reported test
  error (~2.6 px) and 100% recall are **optimistic** — use a session/group split.
- **No negative/occlusion coverage in the test set** (150 all-visible) → false-
  positive rate essentially unvalidated.
- **Threshold** 0.90 not chosen on an independent validation split.
- **Synthetic negatives** dominate the real ones (risk: "blurred patch = empty").

**Before enabling hybrid:** run shadow for several hours across real lighting and
deliberate occlusions / off-board / holes / corners, and require:
`detection_disagreement_px` median < ~5 px (95th < ~12 px) when both detectors see
the marble; **zero** `ai_confidence ≥ 0.90` when the marble is truly absent/occluded;
no spurious reacquisitions. Ideally collect more real negatives/occlusions and
re-evaluate on a group split first.
