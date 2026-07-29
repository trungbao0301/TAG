# CyberRunner passive hardware recorder

This ROS2 package records receipt timing and numeric telemetry from the running
CyberRunner system. It is intentionally subscriber-only. It does not publish
messages, provide or call services, use actions, reset the board, or contain
motor-control logic. ROSout and parameter services are disabled, and the
automatic `/parameter_events` publisher created internally by ROS2 Humble is
removed during initialization so the live node graph also contains only
subscriptions.

The recorder uses `time.monotonic_ns()` at the beginning of every callback. The
same `receipt_monotonic_ns` clock therefore appears in camera, state, command,
and inferred-episode outputs.

## Live interface inventory

Observed on 2026-07-24 with ROS domain 0 and `rmw_fastrtps_cpp`:

| Purpose | Topic | Type | Live publisher QoS / observed rate |
|---|---|---|---|
| Camera | `/cyberrunner_camera/image` | `sensor_msgs/msg/Image` | reliable, volatile; approximately 44.3 Hz in a 12 s CLI snapshot and 34.8 Hz received by the recorder in a loaded 10 s smoke run |
| Ball and board state | `/cyberrunner_state_estimation/estimate` | `cyberrunner_interfaces/msg/StateEstimate` | reliable, volatile; approximately 59.2 Hz |
| TCP observation | `/cyberrunner_state_estimation/estimate_subimg` | `cyberrunner_interfaces/msg/StateEstimateSub` | reliable, volatile; consumed by `tcp_ros_bridge` |
| Hiwonder command / Dreamer action proxy | `/cyberrunner_dynamixel/cmd` | `cyberrunner_interfaces/msg/DynamixelVel` | reliable, volatile; event-driven |
| Estimator target output | `/ball/target` | `geometry_msgs/msg/Point` | estimator diagnostic |
| Estimator target input | `/target_pixel` | `geometry_msgs/msg/Point` | tuner input; not recorded |

`StateEstimate` contains `x_b`, `y_b`, `x_b_dot`, `y_b_dot`, `alpha`, and
`beta`, all `float64`. The ball is considered missing when either position is
non-finite. Board angles are `alpha` and `beta` and arrive in the same message,
so their rate is the state rate.

`DynamixelVel` contains `vel_1` and `vel_2`, both `float64`. The active
`tcp_ros_bridge.py` defaults to clamping each field to `[-180, 180]`. The
Hiwonder compatibility node maps them to servo targets using
`position = 500 + 1.5 * command`, giving `[230, 770]` at the bridge limits,
inside the configured `[100, 900]` servo range. Its internal output loop is
30 Hz, with a 30 ms move time, maximum 20 position units per tick, deadband 1,
and a 1 s command timeout that returns home. The ROS command topic is
event-driven and can be silent when Dreamer is waiting, resetting, training, or
disconnected.

There is no separate ROS Dreamer action topic; the clamped motor command topic
is the observable action proxy. There is no ROS Dreamer episode-event topic.
`episodes.csv` therefore contains clearly labeled ball-visibility inferences,
not official Dreamer episode boundaries, rewards, or success events.

The TCP bridge subscribes to `estimate_subimg` and publishes the command topic.
The current connection is a local tunnel on `127.0.0.1:5555`. For a Dreamer
`step` request, the bridge publishes the received command, then waits up to
18 ms for a state message newer than the pre-command state; on timeout it
returns the most recent state. Both ends set `TCP_NODELAY`. There is no separate
ROS topic carrying TCP request, response, episode, or round-trip timing, so
`state_subimg` and command receipt timing are the passive observables.

The live camera message is 640×400 `bgr8`, frame ID `camera`, 1,920 bytes per
row, and 768,000 payload bytes per frame. Resolution, encoding, stride, frame
ID, and payload size are also read from the first message and stored in
`session_metadata.json`. The default mode records timing and metadata only; it
does not copy image payloads to disk.

## Build

From the workspace root:

```bash
source /opt/ros/humble/setup.bash
colcon build --packages-select cyberrunner_hardware_recorder --symlink-install
source install/setup.bash
```

Building this package does not start the recorder or touch the hardware.

## Record

Record for 10 minutes without saving images:

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 run cyberrunner_hardware_recorder record \
  --duration-sec 600 \
  --output-root hardware_recordings
```

Record until interrupted and save one JPEG every 30 seconds:

```bash
ros2 run cyberrunner_hardware_recorder record \
  --frame-every-sec 30 \
  --output-root hardware_recordings
```

Every session folder contains:

- `session_metadata.json`
- `camera_timing.csv`
- `state.csv`
- `motor_commands.csv`
- `episodes.csv`
- `topic_report.txt`

Sparse JPEGs are optional and disabled by default.

## Stop

Press `Ctrl-C` in the recorder terminal. This only stops the recorder. It does
not send a zero command, reset the board, stop Dreamer, or alter any other ROS2
process. A duration-limited run exits automatically.

## Analyze

```bash
ros2 run cyberrunner_hardware_recorder analyze \
  hardware_recordings/session_YYYYMMDD_HHMMSS
```

The analyzer adds `analysis_summary.json` and `analysis_summary.md`. It reports
topic rates and jitter, estimated gaps, ball-observation loss, state and command
ranges, command saturation, inferred episode statistics, and a preliminary
closed-loop command-to-angle fit.

The delay, gain, and cross-axis estimates are not active system identification.
Dreamer commands are correlated with measured state, resets and saturation may
be present, and the recorder does not excite the hardware. Do not treat these
fits as causal plant parameters.

Export a small sanitized excerpt suitable for Git:

```bash
ros2 run cyberrunner_hardware_recorder export_sample \
  hardware_recordings/session_YYYYMMDD_HHMMSS \
  cyberrunner_hardware_recorder/sample_outputs/session_YYYYMMDD_HHMMSS \
  --max-rows 200
```

This retains summaries and at most 200 rows from each CSV. It excludes image
payloads and sanitizes the absolute session path, host name, and process ID.

## Safety verification

Run:

```bash
python3 -m pytest -q cyberrunner_hardware_recorder/test
```

The safety test parses the recorder source and rejects publisher, service,
service-client, action-client, and message-publication calls.
