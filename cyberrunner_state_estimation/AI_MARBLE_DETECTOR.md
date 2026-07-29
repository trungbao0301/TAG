# AI marble detector

This adds a learned full-camera marble detector without replacing the current
estimator. Train it on images from the exact CyberRunner camera, compare it
against the current state estimate, and only integrate it after validation.

The inference node never publishes motor commands. Its optional outputs are
diagnostic pixel coordinates and confidence only:

- `/cyberrunner_ai_marble/pixel` (`geometry_msgs/msg/PointStamped`)
- `/cyberrunner_ai_marble/confidence` (`std_msgs/msg/Float32`)

The dataset labeler has no application publishers (ROS2 may still expose its
standard internal `/rosout` endpoint).

## 1. Build

```bash
cd /home/trungbao/CYBER/cyberruner-main
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-select cyberrunner_state_estimation
source install/setup.bash
```

## 2. Collect labels passively

Run alongside the existing camera and state estimator:

This machine has another workspace containing an older package with the same
name, so use the project launcher to avoid the old overlay:

```bash
cd /home/trungbao/CYBER/cyberruner-main
./run_ai_marble_labeler.sh
```

An optional first argument changes the dataset directory:

```bash
./run_ai_marble_labeler.sh /path/to/dataset
```

- Left-click the center of the marble to save a visible example.
- Press `N` when the marble is absent or completely occluded.
- Press Space to freeze/unfreeze the displayed frame and `Q` to quit.

Collect at least 500 images across the complete maze, different board angles,
lighting, stationary and moving marble states, partial arm occlusions, and true
ball-loss frames. Include hard negatives such as holes, blue corner markers,
cables, highlights, and the pendulum arm. Do not train only on easy images.

The trainer also creates one inpainted not-visible example from every visible
click. This is useful for the first model, but it does not replace collecting
real `N` examples under the pendulum arm and with the marble off the board.

## 3. Train and export ONNX

Training requires a working PyTorch installation. It can run on the training
server; runtime inference on the ROS machine uses OpenCV and does not require
PyTorch.

```bash
ros2 run cyberrunner_state_estimation ai_train ai_marble_dataset \
  --output models/marble_detector.onnx --epochs 60
```

The current model predicts an 80x50 heatmap from a 320x200 RGB input
(`--output-stride 4`). Brightness, contrast, and horizontal/vertical flips are
augmented during training.

## 4. Run in parallel without changing Dreamer

```bash
cd /home/trungbao/CYBER/cyberruner-main
./run_ai_marble_detector.sh
```

The launcher defaults to `models/marble_detector.onnx`. Pass another model as
its first argument when needed.

The diagnostic launcher uses a `0.90` confidence threshold, excludes pixels
outside the playable-board ROI, and retains the last accepted position for 90
camera frames during a brief occlusion. This prevents blue supports and corner
hardware outside the maze from being reported as the marble.

Set `publish_diagnostics:=false` for a viewer-only test. The node subscribes to
the camera and has no connection to the Dynamixel command topic.

## Validation before integration

Record both the existing estimate and AI diagnostic output for at least 10
minutes. Measure labeled pixel error, false positives on holes/markers,
false-negative duration, confidence calibration, inference FPS, and behavior
during pendulum occlusion. A recommended acceptance target is at least 99%
visible/not-visible accuracy and a 95th-percentile center error below one marble
radius on a held-out, manually labeled test set.

Do not enable the hybrid production path merely because training loss is low.

In hybrid mode, AI candidates are also rejected when their board-relative
position falls inside a known maze hole. The exclusion zones follow the four
moving board markers, so they remain aligned while the maze tilts. The default
radius is the 7.5 mm physical hole plus a 2.5 mm guard margin. A candidate is
still accepted while it crosses a hole; it is rejected only after remaining in
the same hole continuously for 2 seconds. Configure this with
`ai_hole_rejection_enabled`, `ai_hole_rejection_margin_m`, and
`ai_hole_rejection_delay_sec`. During the delay, `ball_source` reports
`ai_hole_pending_N`; afterward it reports `ai_hole_rejected_N` and publishes a
missing marble observation.
The configurable fallback in `core/detection.py` remains disabled by default
until its shadow-mode diagnostics pass on the live system.

## Hybrid estimator integration

The estimator now supports three modes. `off` is the default and preserves the
existing HSV-only behavior. `shadow` runs AI diagnostics without changing the
selected marble measurement. `hybrid` is AI-authoritative: agreeing HSV and AI
detections are fused, an AI-only detection can be accepted through the
continuity/confirmation gates, and an HSV-only detection is treated as
missing. This prevents a blue board marker from resetting the marble-loss
timer. Hybrid mode evaluates AI on every frame; `ai_check_every_n_frames` only
reduces work in shadow mode.

Do not run the standalone AI viewer at the same time unless its visual output
is needed; both processes would perform the same ONNX inference.

Start a shadow comparison on the next planned estimator launch:

```bash
cd /home/trungbao/CYBER/cyberruner-main
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch cyberrunner_camera camera_estimation_gpu.launch.py \
  ai_mode:=shadow \
  ai_model_path:=/home/trungbao/CYBER/cyberruner-main/models/marble_detector.onnx
```

Inspect the diagnostics:

```bash
ros2 topic echo /cyberrunner_state_estimation/ball_source
ros2 topic echo /cyberrunner_state_estimation/ai_confidence
ros2 topic echo /cyberrunner_state_estimation/detection_disagreement_px
```

After shadow validation, enable guarded fallback by changing only the mode:

```bash
ros2 launch cyberrunner_camera camera_estimation_gpu.launch.py \
  ai_mode:=hybrid \
  ai_model_path:=/home/trungbao/CYBER/cyberruner-main/models/marble_detector.onnx
```

Hybrid source values include `hsv`, `fused`, `ai_reacquired`,
`ai_reacquired_confirmed`, `kalman_occlusion`, and `lost`. During
`kalman_occlusion`, the Kalman filter receives a missing measurement and the
node publishes its bounded prediction. After 90 frames or excessive
uncertainty, the existing finite/NaN loss contract resumes.

The moving blue reference markers are masked at their freshly detected pixel
positions on every frame before HSV marble detection. The AI detector receives
an unmodified copy of the image, so it can still detect a real marble near a
marker.

After any missing frame, a replacement position must remain spatially
consistent for three frames before it is accepted. After six consecutive
missing HSV measurements, the predictive HSV crop is reset to a full-board
search while the bounded Kalman occlusion grace continues. This provides the
same recovery that temporarily covering and uncovering the marble previously
triggered by hand, without allowing a rejected HSV-only candidate to reset the
loss timer.

Rollback requires no code change:

```bash
ros2 launch cyberrunner_camera camera_estimation_gpu.launch.py ai_mode:=off
```
