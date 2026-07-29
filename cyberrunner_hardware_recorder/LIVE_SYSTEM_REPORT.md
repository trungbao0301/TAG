# CyberRunner ROS2 passive hardware interface report

Inspection date: 2026-07-24  
ROS domain: 0  
RMW implementation: `rmw_fastrtps_cpp`

All inspection was passive. It used graph queries, topic subscriptions, process
environment reads, TCP socket-state inspection, and source review. No message
was published, no service or action was called, and no process, board, or
controller was stopped, reset, or reconfigured.

## Running components

The relevant observed nodes were:

- `/cyberrunner_camera`
- `/cyberrunner_state_estimation_subimg`
- `/tcp_ros_bridge`
- `/cyberrunner_hiwonder_compat`
- `/overlay_map_view_simple`
- `/safe_cyberrunner_image_viewer`
- `/arduino_ball_loss_bridge`

The DreamerV3 process runs on the existing remote training host and communicates
through an existing SSH tunnel. The local bridge endpoint is loopback TCP port
5555; remote host details are intentionally omitted from committed artifacts.

## Topics and message types

| Topic | Type | Role |
|---|---|---|
| `/cyberrunner_camera/image` | `sensor_msgs/msg/Image` | Raw camera |
| `/cyberrunner_state_estimation/estimate` | `cyberrunner_interfaces/msg/StateEstimate` | Ball state and board angles |
| `/cyberrunner_state_estimation/estimate_subimg` | `cyberrunner_interfaces/msg/StateEstimateSub` | State plus 64×64 estimator crop for TCP |
| `/cyberrunner_dynamixel/cmd` | `cyberrunner_interfaces/msg/DynamixelVel` | Dreamer action proxy and Hiwonder command |
| `/ball/target` | `geometry_msgs/msg/Point` | Estimator diagnostic target |
| `/target_pixel` | `geometry_msgs/msg/Point` | Tuner input |
| `/tf` | `tf2_msgs/msg/TFMessage` | Estimator transforms |
| `/tf_static` | `tf2_msgs/msg/TFMessage` | Static estimator transforms |

`StateEstimate` fields:

- `x_b`, `y_b`: ball position in metres
- `x_b_dot`, `y_b_dot`: ball velocity
- `alpha`, `beta`: board angles in radians

The message has no header or sequence number. A missing ball is represented by
non-finite `x_b` or `y_b`.

`DynamixelVel` fields are `vel_1` and `vel_2`, both `float64`.

## Camera

- Topic: `/cyberrunner_camera/image`
- Resolution: 640×400
- Encoding: `bgr8`
- Frame ID: `camera`
- Row stride: 1,920 bytes
- Payload: 768,000 bytes per message
- Publisher QoS: reliable, volatile
- CLI snapshot: approximately 44.3 Hz over 12 seconds
- Subscriber-only smoke recording: approximately 34.8 Hz over 10 seconds

The difference between the two FPS measurements is load and subscriber
dependent. The recorder reports both source-header and monotonic receipt
timing. It defaults to metadata/timing only and does not write image payloads.

## State and board angles

- Primary topic: `/cyberrunner_state_estimation/estimate`
- TCP topic: `/cyberrunner_state_estimation/estimate_subimg`
- Publisher QoS: reliable, volatile
- Primary snapshot: approximately 59.2 Hz
- Smoke recording: approximately 60.0 Hz for both state topics
- Board-angle rate: the same as the state rate because `alpha` and `beta` are
  fields of every state message

The `StateEstimateSub.subimg.header.stamp` is zero in the inspected system.
Consequently, exact camera-to-estimator latency cannot be recovered by matching
source stamps. The analyzer reports a clearly labeled latest-camera-receipt age
instead; missed best-effort camera receipts can overestimate it.

## Hiwonder command path

- Topic: `/cyberrunner_dynamixel/cmd`
- Type: `cyberrunner_interfaces/msg/DynamixelVel`
- Publisher: `/tcp_ros_bridge`
- Subscriber: `/cyberrunner_hiwonder_compat`
- Topic QoS: reliable, volatile
- Topic rate: event-driven by Dreamer; it can be silent during training,
  waiting, reset, or disconnection
- Bridge clamps: `vel_1` and `vel_2` each default to `[-180, 180]`

No live process environment override for either bridge clamp was present.

The active Hiwonder node had no command-line parameter overrides, so its source
defaults apply:

- `home_pos_1 = home_pos_2 = 500`
- `servo_min_1 = servo_min_2 = 100`
- `servo_max_1 = servo_max_2 = 900`
- `scale_1 = scale_2 = 1.5`
- target mapping: `position = 500 + 1.5 * command`
- target range at bridge limits: `[230, 770]`
- output loop: 30 Hz
- move time: 30 ms
- slew limit: 20 position units per output tick per axis
- deadband: 1 position unit
- command timeout: 1 second, then return home

The recorder captures the received command values and timing, not private
Hiwonder HID writes or servo telemetry.

## DreamerV3 topics and episodes

There is no separate ROS Dreamer action topic. The closest observable action is
the clamped command on `/cyberrunner_dynamixel/cmd`.

No ROS episode, reward, terminal, reset-event, or success topic was present.
Those values exist inside the remote Gym/Dreamer process. The recorder therefore
does not claim official Dreamer episode statistics. It writes
`episodes.csv` using finite-ball intervals and labels every row
`inference_source=ball_visibility`.

## TCP bridge timing

The local bridge:

1. subscribes to `/cyberrunner_state_estimation/estimate_subimg`;
2. receives newline-delimited JSON from the existing TCP connection;
3. for a `step`, publishes the requested clamped command;
4. waits up to 18 ms for a state receipt newer than the pre-command state;
5. returns that state, or the most recent state when the wait expires.

Both TCP ends use `TCP_NODELAY`. The bridge also accepts `obs`, `action`, and
`reset` requests, but the passive recorder neither observes raw TCP payloads nor
participates in that connection. There is no ROS TCP-request or round-trip
timing topic. Any timing inferred from state and command receipts is therefore
preliminary.

## Identification limits

The analysis performs no excitation. Command-to-angle delay, gain, and
cross-axis coupling are least-squares fits to closed-loop policy data. Policy
feedback, saturation, estimator filtering, missing observations, episode
boundaries, and the Hiwonder slew limiter can bias all estimates. Separate
approval is required before any step, sweep, impulse, or other active
system-identification command.
