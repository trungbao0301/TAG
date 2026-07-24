# AI Marble Detector (TAG)

A small CNN marble detector that augments the classical HSV state estimator when
the HSV detector struggles: pendulum-arm occlusion, reflections, holes, corner
markers, blue hardware, or objects outside the board. **It never controls
motors** — it only produces a marble pixel detection / confidence, or assists the
existing estimator through guarded fusion + Kalman prediction.

> ⚠️ **Validation status (read before deploying):** the shipped model has known
> evaluation weaknesses (see *Validation caveats* below). It is safe to run in
> `off` (default) and `shadow` modes. Do **not** enable `hybrid` on the real
> robot until the shadow-mode acceptance criteria pass.

---

## Files in this repo

| File | Purpose |
|---|---|
| `tag_state_estimation/ai_marble_common.py` | ONNX inference + heatmap decode + ROI (OpenCV DNN, no PyTorch at runtime) |
| `tag_state_estimation/core/hybrid_ball.py` | Guarded HSV/AI fusion + reacquire state machine (pure logic) |
| `tag_state_estimation/ai_marble_detector_node.py` | Standalone diagnostic node (pixel + confidence only) |
| `tag_state_estimation/train_ai_marble.py` | Trainer (`ai_train`) |
| `tag_state_estimation/ai_dataset_labeler.py` | Click-to-label tool (`ai_labeler`) |
| `models/marble_detector.onnx` | Deployed weights (see hash below) |

**Model:** 320×200 RGB in → 80×50 heatmap out, output stride **4**, ~510 KB.
**SHA-256:** `0a09032fb6a62c680dcc16f1411973aebe7e1d77771e094cfbd828adbdeb154b`

> Full **hybrid estimator integration** (wiring the detector into
> `core/detection.py` / `tag_state_estimation_subimg.py` with `ball_source`,
> `ai_confidence`, `detection_disagreement_px` diagnostics) lives in the
> Multi-maze deployment. This repo ships the AI **library + standalone
> diagnostic + trainer + model**; integrate into the estimator only after
> shadow-mode validation.

---

## Modes

- **off** (default): HSV-only; original behavior, byte-for-byte.
- **shadow**: AI runs and publishes diagnostics; HSV still drives the estimator.
- **hybrid**: guarded HSV/AI fusion, AI reacquisition, bounded Kalman prediction
  during occlusion. (Requires the estimator integration; keep off until validated.)

---

## Reproducible training (stride 4)

The deployed model is **stride 4**. The trainer default is stride 8, so you
**must** pass `--output-stride 4` or you will get a different architecture:

```bash
ros2 run tag_state_estimation ai_train ai_marble_dataset \
  --output models/marble_detector.onnx \
  --epochs 60 \
  --output-stride 4
```

Recorded config: input 320×200, epochs 60, batch 32, lr 3e-4, AdamW,
BCEWithLogitsLoss(pos_weight=20), brightness/contrast + H/V flip aug, seed 7.

---

## Safe commands

**Collect more data (labeler):**
```bash
ros2 run tag_state_estimation ai_labeler ai_marble_dataset
# left-click = marble center · N = not visible · Space = freeze · Q = quit
```

**Retrain on a GPU** (server): run the `ai_train` command above with
`--output-stride 4`; the deployed model came from physical GPU 5.

**Standalone diagnostic detector** (subscribe-only; publishes only
`/tag_ai_marble/pixel` and `/tag_ai_marble/confidence`):
```bash
ros2 run tag_state_estimation ai_detector \
  --ros-args -p model_path:=models/marble_detector.onnx
```

**Shadow mode** (after estimator integration; camera must be the same view the
model was trained on — see *Geometry*):
```bash
ros2 launch tag_camera camera_estimation_gpu.launch.py \
  ai_mode:=shadow \
  ai_model_path:=/home/trungbao/CYBER/tag/models/marble_detector.onnx
```

**Hybrid mode** (only after shadow criteria pass):
```bash
ros2 launch tag_camera camera_estimation_gpu.launch.py \
  ai_mode:=hybrid \
  ai_model_path:=/home/trungbao/CYBER/tag/models/marble_detector.onnx
```

**Immediate rollback to HSV-only:**
```bash
ros2 launch tag_camera camera_estimation_gpu.launch.py ai_mode:=off
```

> ⚠️ The full launch starts **both** a camera and an estimator. Do **not** run it
> while another camera already owns the video device — you'll get a device
> conflict. To run the estimator against an existing camera feed, launch the
> estimator only and remap the image topic + pass pose-zero params, e.g.:
> ```bash
> ros2 run tag_state_estimation estimator_sub --ros-args \
>   -r tag_camera/image:=/cyberrunner_camera/image \
>   -p ai_mode:=off \
>   -p pose_zero_alpha_deg:=<deg> -p pose_zero_beta_deg:=<deg>
> ```

---

## Geometry (important)

The model and its ROI (`x=[0.25,0.72]`, `y=[0.15,0.80]`, normalized) were trained
on the **CyberRunner** camera view (640×400, that specific crop/border/board
placement). They are valid **only** if the estimator receives the same
resolution, crop, border, and board placement. If the tag camera geometry
differs, you need new labels / an explicit ROI recalibration before the model is
trustworthy.

---

## Validation caveats (from the technical review)

- **Train/test leakage:** the split is random per-image, and capture sessions
  straddle the train/test boundary (adjacent ~0.18 s frames are near-duplicates).
  Reported test error (~2.6 px) and 100% recall are **optimistic** — not a true
  generalization estimate. Use a **session/group split** for any real evaluation.
- **No negative/occlusion coverage in the test set:** the 150 test frames are all
  visible-marble; false-positive rate is essentially unvalidated.
- **Threshold** (0.90 deploy) was not chosen on an independent validation split.
- **Synthetic negatives** (inpainted marbles) dominate the real negatives and may
  teach a "blurred patch = empty" shortcut.

**Before enabling hybrid:** run shadow mode for several hours across real lighting
and deliberate occlusions / off-board / holes / corners, and require:
`detection_disagreement_px` median < ~5 px (95th < ~12 px) when both detectors see
the marble; **zero** `ai_confidence ≥ 0.90` events when the marble is truly
absent/occluded; no spurious reacquisitions. Ideally also collect more real
negatives/occlusions and re-evaluate on a group split first.
