# Runbook — bringing the whole system up

Start order matters where noted. Every local command runs from the repo root
(`/home/trungbao/CYBER/cyberruner-main`) with ROS sourced:

```bash
cd /home/trungbao/CYBER/cyberruner-main
source /opt/ros/humble/setup.bash
source install/setup.bash
```

---

## Read this first: two traps that cost real time

### 1. A second workspace shadows this one

`~/cyberrunner_ws` is sourced from the shell profile and sits **before** this repo
in `AMENT_PREFIX_PATH`. These packages exist in **both**, so `ros2 run` may
silently resolve the wrong one — it starts, logs a normal-looking banner, and
runs none of this checkout's code:

```
cyberrunner_camera   cyberrunner_dreamer   cyberrunner_interfaces
cyberrunner_state_estimation
```

`cyberrunner_dynamixel` is **not** in the other workspace, so `ros2 run` is safe
for it.

For the estimator, always use the wrapper — it `unset`s the inherited overlay so
the other workspace cannot win:

```bash
./run_estimator_cyber.sh --ros-args ...
```

Verify with the line it prints:
```
AMENT head: /home/trungbao/CYBER/cyberruner-main/install/cyberrunner_state_estimation
```
Also check `ros2 node list` shows `/cyberrunner_state_estimation` (this repo) and
**not** `/cyberrunner_state_estimation_subimg` (the other workspace), and that
`/cyberrunner_state_estimation/ball_source` exists.

### 2. Exactly ONE tcp_ros_bridge

The trainer accepts whichever bridge reconnects first. If two are running, both
hold ESTABLISHED sockets, both answer `obs`, and both publish motor commands.
Killing "the spare" can pull the socket out from under a live run and crash it
with `ConnectionError: TCP bridge disconnected`.

Before starting anything, check:
```bash
pgrep -af tcp_ros_bridge.py        # must be 0 or 1 lines
```

---

## Local startup

### 1. Camera

```bash
python3 fast_camera_publisher_v2.py
```
Must be first — the estimator subscribes to `/cyberrunner_camera/image`.
Check: `ros2 topic hz /cyberrunner_camera/image` (~37–44 Hz).

### 2. Servos

```bash
ros2 run cyberrunner_dynamixel hiwonder_compat_node.py
```

### 3. State estimator

```bash
./run_estimator_cyber.sh --ros-args \
  -p ai_mode:=hybrid \
  -p ai_model_path:=$PWD/models/marble_detector.onnx
```

To A/B a different model, just swap the path — nothing else changes:
```bash
  -p ai_model_path:=$PWD/models/marble_detector_v5_moredata.onnx
```

Check:
```bash
ros2 topic echo /cyberrunner_state_estimation/ball_source     # want: fused
ros2 topic hz   /cyberrunner_state_estimation/estimate        # ~55 Hz
```

### 4. Arduino ball-loss bridge

```bash
python3 scripts/arduino_ball_loss_bridge.py
```

### 5. SSH tunnel to the training server

```bash
ssh -N -L 5555:127.0.0.1:5555 tbt589@aere-a83514.ae.utexas.edu
```
Local port 5555 forwards to the server's 5555. The bridge is a **client** that
connects out through this, so the tunnel must exist before the bridge is useful.

### 6. TCP bridge

```bash
python3 -u tcp_ros_bridge.py 127.0.0.1 5555
```
Expect on startup:
```
TCP command clamp: vel_1=[-300.0, 300.0], vel_2=[-300.0, 300.0]
```
It will log `Connecting to server ...` once per second until the trainer is up —
that is normal, and the 1 s backoff is deliberate (see *Known fixes* below).

### 7. Optional views

```bash
python3 -u overlay_map_view_simple.py --display_hz 15 --checkpoint_radius_m 0.010
./run_ai_marble_detector.sh          # defaults to models/marble_detector.onnx
```

Both are subscribe-only and safe to start/stop mid-run. The AI detector costs
~190% CPU and drops the trainer's step rate — close it if the rate matters more
than the picture.

**The AI view's green marker does not mean "detecting now."** It holds the last
position for up to `miss_grace_frames` (90 ≈ 1.6 s). `misses=49` means 49
consecutive frames with **no** detection.

---

## Server: the training run

Always in tmux, or it dies when SSH closes.

### Fresh run (from scratch)

A new logdir means no checkpoint, so training starts from zero:

```bash
ssh tbt589@aere-a83514.ae.utexas.edu
RUN=maprawv2_fresh_$(date +%Y%m%d_%H%M%S)
tmux new-session -d -s cyberrunner_fresh \
  "~/cyberruner-main/run_fresh_train.sh $RUN 2>&1 | tee ~/cyberrunner_logs/${RUN}_stdout.log"
```

### Continue the latest run

```bash
RUN=$(basename "$(ls -td ~/cyberrunner_logs/maprawv2_fresh_* | head -1)")
tmux new-session -d -s cyberrunner_fresh \
  "~/cyberruner-main/run_fresh_train.sh $RUN 2>&1 | tee -a ~/cyberrunner_logs/${RUN}_stdout.log"
```

`run_fresh_train.sh` sets `PATH`/`LD_LIBRARY_PATH` for the micromamba env, sources
the ROS setup files and `install/setup.bash`, exports `CYBERRUNNER_TCP_BIND=0.0.0.0`,
`CYBERRUNNER_TCP_PORT=5555`, `CYBERRUNNER_RUN_NAME`, `CYBERRUNNER_LOGDIR`,
`CYBERRUNNER_REST_AFTER_SEC=3600`, `CYBERRUNNER_REST_DURATION_SEC=180`, then
`exec ros2 run cyberrunner_dreamer train`.

### Verify it came up

```bash
tmux list-sessions                              # cyberrunner_fresh present
ss -ltnp | grep 5555                            # LISTEN 0.0.0.0:5555
tmux capture-pane -p -t cyberrunner_fresh | tail -20
```

The banner should show the current settings:
```
[TCP ENV] progress anti-cheat: cheat=False, anticheat_termination=OFF,
          max_single_step=0.057m, max_speed=1.00m/s, confirm_steps=5
[TCP ENV] ball_loss_grace=0.00s, occlusion_grace=0.00s
```

Then from the local machine:
```bash
ss -tnp | grep 5555 | grep python                # bridge ESTAB
ros2 topic echo /cyberrunner_dynamixel/cmd       # commands flowing, |v| up to 300
```

---

## Full-stack health check

```bash
ros2 node list
# /arduino_ball_loss_bridge  /cyberrunner_camera  /cyberrunner_hiwonder_compat
# /cyberrunner_state_estimation  /overlay_map_view_simple  /tcp_ros_bridge
# (+ /cyberrunner_ai_marble_detector if the view is open)

cat /proc/loadavg                                # ~9-13 of 24 cores is normal
```

`ball_source` values:

| value | meaning |
|---|---|
| `fused` | HSV and AI agree within 12 px — healthy |
| `kalman_occlusion` | no measurement; Kalman predicting |
| `lost_uncertain` | prediction std exceeded `ai_max_prediction_std_m` |
| `lost_outside` | rejected by the playable-area gate |
| `lost` | tracker gave up |
| `ai_hole_pending_N` | timed hole rejector counting down |

---

## Shutdown

Reverse order. Stop the trainer first so the bridge is not yanked mid-episode:

```bash
# server
ssh tbt589@aere-a83514.ae.utexas.edu 'tmux kill-session -t cyberrunner_fresh'

# local — kill by PID, not by pattern
pgrep -af "tcp_ros_bridge.py|estimator_ai_map|overlay_map_view_simple|ai_detector"
kill <pid> ...
```

> Do **not** use `pkill -f tcp_ros_bridge.py` from a shell whose own command line
> contains that string — it matches itself and kills the shell (exit 144), leaving
> the intended target running. Get the PID first, then `kill <literal-pid>`.

---

## After editing code

```bash
colcon build --packages-select cyberrunner_state_estimation   # estimator changes
colcon build --packages-select cyberrunner_dreamer            # env changes
```

`build/<pkg>/<pkg>/` holds **copies**, not symlinks, so a source edit does nothing
until the rebuild. Then restart the affected node.

`env_tcp.py` runs on the **server**, so local edits need syncing:
```bash
scp cyberrunner_dreamer/cyberrunner_dreamer/env_tcp.py \
  tbt589@aere-a83514.ae.utexas.edu:'~/cyberruner-main/cyberrunner_dreamer/cyberrunner_dreamer/env_tcp.py'
```
The server's `build/cyberrunner_dreamer/cyberrunner_dreamer` is a **symlink** to the
source, so no rebuild is needed there — but the trainer must restart.

`tcp_ros_bridge.py` is a plain script: restart, no build.

---

## Known fixes already in place

| area | state |
|---|---|
| Plate frame geometry | `L_EXT_INT_X = 0.305`, `FIXED_DOT_INSET_Y = 0.0413` |
| `H_BORDERS = 0.0036` | fixes the tilt-coupled +y bias (43.6% → 0% unreachable) |
| AI ROI | follows the tracked corner dots; blue dots disc-masked 12 px |
| Anti-cheat | termination **OFF** (no shortcut is reachable on this board) |
| Command limit | 300 in env *and* bridge (they used to disagree: 240 vs 180) |
| Bridge reconnect | 1 s backoff (was spinning at 250/s, 27% CPU) |
| Overlay | 0.25 s staleness expiry; 22 mm margin; `!! UNREACHABLE` flag |

See [detection_and_anticheat_fixes.md](detection_and_anticheat_fixes.md) for the
measurements behind each.

## Still open

1. **`playable_edge_tolerance_y` is 20 mm** — only existed to absorb the
   `H_BORDERS` bias, now fixed. Should come down to 5 mm to match x.
2. **`H_BORDERS = 0.0036` slightly overshoots** — residual slope +9.4 mm/rad;
   the zero-crossing suggests ~0.0102 m. Nothing exceeds the reach limit at
   0.0036, so it is not urgent.
3. **Actions are not clipped to [−1, 1]** before scaling by `max_angle_vel`.
4. **Label frames with the marble at the board edges** under the pink LED wash —
   the false-lock failure mode is unrepresented in the training set.
