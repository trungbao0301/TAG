# TAG — Labyrinth Marble Robot

A model-based reinforcement-learning robot that learns to play the labyrinth
marble maze. A camera looks down at the board, a state estimator tracks the
marble and board tilt, two Hiwonder servos tilt the board, and a DreamerV3 agent
learns to solve the maze from real experience.

This is a slimmed, rebranded workspace (packages use the `tag_` prefix).

---

## Architecture

```
                 ROBOT MACHINE                              SERVER (GPU)
 ┌──────────────────────────────────────────┐        ┌────────────────────┐
 camera ──▶ fast_camera_publisher_v2.py       │        │  DreamerV3 training │
             │  /tag_camera/image             │        │  (dreamerv3.train)  │
             ▼                                │  TCP   │                     │
        tag_state_estimation (estimator_sub)  │◀──────▶│  gym env tag-ros-v0 │
             │  /tag_state_estimation/estimate│  5555  └────────────────────┘
             ▼                                │
        tcp_ros_bridge.py ◀───────────────────┘
             │  /tag_hiwonder/cmd
             ▼
        tag_hiwonder (hiwonder_compat_node.py) ──▶ servos
```

The robot side collects experience in real time; the server runs the learner
and sends actions back over TCP.

---

## Prerequisites

- Ubuntu 22.04 + **ROS 2 Humble**
- `colcon`, `python3`, and Python deps: `numpy`, `opencv-python`, `scipy`,
  `gym`, `cv_bridge`
- `v4l-utils` (for the camera color controls)
- A See3CAM (or compatible V4L2) camera at `/dev/video*`
- Hiwonder LX-16A servos over USB HID

---

## Build

Build the **same workspace on both machines** — the robot (ROS nodes) and the
GPU server (training). Steps are identical.

```bash
# 1. get the code onto the machine
cd ~/CYBER/tag                      # workspace root

# 2. source ROS 2
source /opt/ros/humble/setup.bash

# 3. build all packages (tag_interfaces builds first automatically)
colcon build                       # add --symlink-install to skip rebuilds on python edits

# 4. source the built workspace
source install/setup.bash
```

Notes:
- **Source `install/setup.bash` in every new terminal** before running any node.
- If you moved/renamed the workspace folder, run `rm -rf build install log`
  then `colcon build` again (the install embeds absolute paths).
- Build a single package after an edit: `colcon build --packages-select <pkg>`
  (e.g. `tag_state_estimation`). With `--symlink-install`, python edits need only
  a node restart, no rebuild.
- The server also needs the DreamerV3 python deps (JAX + CUDA); install them in
  the training environment per `dreamerv3/requirements.txt`.

---

## Run — robot side

Open a terminal for each step (source ROS + the workspace in each).

**1. Camera** (publishes `/tag_camera/image`; applies the locked color settings
for stable marble detection):
```bash
python3 fast_camera_publisher_v2.py
```

**2. Servo driver** (drives the two Hiwonder servos, listens on
`/tag_hiwonder/cmd`):
```bash
ros2 run tag_hiwonder hiwonder_compat_node.py
```

**3. State estimation** (detects the marble + board tilt, publishes
`/tag_state_estimation/estimate`):
```bash
ros2 run tag_state_estimation estimator_sub
```

**4. Bridge to the training server** (connect to the server's IP; default port
5555):
```bash
python3 tcp_ros_bridge.py <SERVER_IP> 5555
```

> First-time calibration: if the estimator can't find the board corners, set the
> marker positions once with
> `ros2 run tag_state_estimation select_markers`.

---

## Train on the GPU server

The learner runs on the server; the robot streams observations/actions to it
over TCP (default port **5555**). The **server listens** and the **robot's
bridge connects** to it. If the server is remote, tunnel the port over SSH.

### How the connection works
```
robot: tcp_ros_bridge.py ──connect──▶ 127.0.0.1:5555
                                          │  (SSH tunnel)
server: dreamerv3.train (env binds 0.0.0.0:5555) ◀──────┘
```

### Step 0 — get the code onto the server and build it

Sync the workspace to the server, then build it there.

**Option A — via GitHub (recommended):**
```bash
# on the server — first time:
git clone git@github.com:trungbao0301/TAG.git ~/CYBER/tag
# later, to update after you push changes:
cd ~/CYBER/tag && git pull
```

**Option B — rsync straight from the robot machine** (no GitHub round-trip):
```bash
# run on the robot; copies source only (build artifacts excluded)
rsync -az --delete \
  --exclude build --exclude install --exclude log --exclude __pycache__ \
  ~/CYBER/tag/ <user>@<server>:~/CYBER/tag/
```

**Build on the server** (needs ROS 2 + the DreamerV3 python env with JAX/CUDA):
```bash
cd ~/CYBER/tag
source /opt/ros/humble/setup.bash
colcon build            # or: colcon build --packages-select tag_dreamer tag_interfaces
source install/setup.bash
```

> Re-run the sync + `colcon build` (or `--packages-select` the changed package)
> whenever you edit code on the robot and want it on the server.

### Step 1 — start training on the server
With the workspace built and sourced (Step 0), launch:

```bash
python3 -m dreamerv3.train \
  --configs tag large \
  --task gym_tag_dreamer:tag-ros-v0 \
  --logdir ~/tag_logs/run1 \
  --replay_size 1e6 \
  --run.script train_top5 \
  --run.train_ratio 256 \
  --run.save_every 20 \
  --run.log_every 1 \
  --jax.policy_devices 0 \
  --jax.train_devices 0
```

- `--configs tag` — the maze config profile.
- `--task gym_tag_dreamer:tag-ros-v0` — the real-robot gym env.
- `--logdir` — where checkpoints, replay chunks, and TensorBoard logs are written.
- `--run.train_ratio` — how many times each collected sample is replayed for
  learning. Higher = more learning per real step, but in this **synchronous**
  loop too high slows data collection. **256 is a good default; drop to 128 if
  the robot steps slower** (watch that chunks still save regularly).
- `--jax.policy_devices` / `--jax.train_devices` — GPU indices for acting vs
  learning (same GPU is fine).

### Step 2 — open the TCP tunnel on the robot (only if the server is remote)
```bash
ssh -N -L 5555:127.0.0.1:5555 <user>@<server>
```
Leave it running. This forwards the robot's `localhost:5555` to the server.

### Step 3 — start the robot side
Bring up the robot nodes (see **Run — robot side**), and point the bridge at the
tunnel:
```bash
python3 tcp_ros_bridge.py 127.0.0.1 5555     # 127.0.0.1 because of the SSH tunnel
```
(Use the server's real IP instead of `127.0.0.1` if you're on the same LAN with
no tunnel.)

Training now runs: the server learns while the robot plays in real time. Motors
pause periodically to cool down (`TAG_REST_*`).

### Watch training in TensorBoard
On the server:
```bash
tensorboard --logdir ~/tag_logs --port 6006
```
Tunnel the UI port from the robot/your laptop, then open it in a browser:
```bash
ssh -N -L 6006:127.0.0.1:6006 <user>@<server>   # then browse http://localhost:6006
```
Watch **`episode/score`** — that's the learning signal (should trend up over
real-world hours). Also useful: losses and `fps`.

> If TensorBoard isn't installed on the server, copy just the event files down
> and view them locally:
> `rsync -az <user>@<server>:'~/tag_logs/run1/events.out.tfevents.*' ~/tag_logs/run1/`
> then `tensorboard --logdir ~/tag_logs`.

---

## Debug / tuning tools (subscribe-only, safe to run alongside)

| Tool | Purpose |
|---|---|
| `python3 marble_hsv_picker.py` | Click the marble to sample its HSV; press `p` to print a `DEFAULT_HSV_BALL` block for `tag_state_estimation/.../core/detection.py`. `d` checks the blob passes the detector gates. |
| `python3 camera_tuner_live.py` | Live sliders for camera white-balance / exposure / saturation via `v4l2-ctl`. Lock WB + exposure so the marble's color is stable at every board position. |
| `python3 grayscale_marble_detector.py` | Experimental grayscale + size detector (for comparison). |
| `python3 safe_image_viewer.py` | View the camera feed (subscribe-only, never opens the device). |
| `python3 overlay_map_view_simple.py` | Overlay the maze map, waypoints, and tracked marble. |

---

## AI marble detector (optional)

An optional learned detector augments the HSV estimator during pendulum-arm
occlusion, reflections, holes, blue markers, and off-board false detections. It
**never controls motors** — it only produces a marble pixel/confidence or assists
the estimator through guarded fusion + Kalman prediction. Default mode is `off`
(pure HSV); `shadow` runs diagnostics only; `hybrid` is AI-authoritative.

📄 **See [`tag_state_estimation/AI_MARBLE_DETECTOR.md`](tag_state_estimation/AI_MARBLE_DETECTOR.md)** for
modes, the stride-4 training command, safe shadow/hybrid/rollback commands, and
validation caveats. Model: `models/marble_detector.onnx`.

> Do not enable `hybrid` on the robot until shadow-mode acceptance criteria pass.

---

## Configuration (environment variables)

Tunable knobs are read from `TAG_*` environment variables (see
`tag_dreamer/tag_dreamer/env_tcp.py` and `tcp_ros_bridge.py`). Common ones:

| Variable | Meaning | Default |
|---|---|---|
| `TAG_TCP_PORT` | training/bridge TCP port | 5555 |
| `TAG_MAX_ANGLE_VEL` | max servo angular velocity | 240 |
| `TAG_MAX_CMD_1` / `TAG_MAX_CMD_2` | per-motor command clamp | 240 |
| `TAG_REST_AFTER_SEC` / `TAG_REST_DURATION_SEC` | motor cooldown schedule | 3600 / 240 |
| `TAG_BALL_LOSS_GRACE_SEC` / `TAG_OCCLUSION_GRACE_SEC` | how long to hold a briefly-lost marble | 0.35 / 1.50 |

### Camera color
The camera locks white balance + exposure on startup (baked into
`fast_camera_publisher_v2.py`) so the marble's blue is consistent and the frame
rate stays high. Re-tune with `camera_tuner_live.py`, then re-run
`marble_hsv_picker.py` and paste the new range into `detection.py`.

### Servo tilt range
The maximum board tilt is set by `servo_min_*` / `servo_max_*` in
`tag_hiwonder/scripts/hiwonder_compat_node.py` (defaults 100–900 around a 500
center). Widen that window to allow a larger tilt angle.

---

## Topics

| Topic | Type | Direction |
|---|---|---|
| `/tag_camera/image` | `sensor_msgs/Image` | camera → estimator |
| `/tag_state_estimation/estimate` | `tag_interfaces/StateEstimate` | estimator → consumers |
| `/tag_state_estimation/estimate_subimg` | `tag_interfaces/StateEstimateSub` | estimator → bridge |
| `/tag_hiwonder/cmd` | `tag_interfaces/HiwonderVel` | bridge → servos |
| `/tag_hiwonder/reset` | `tag_interfaces/HiwonderReset` (srv) | reset board to level |
# TAG
# TAG
