# Training runbook

Training is split across two machines:

- **PC** (the one wired to the labyrinth) runs the camera, the state estimator,
  the motors, and a TCP bridge.
- **Server** (the GPU box) runs DreamerV3. Its gym environment *listens* on
  port 5555; the PC's bridge connects out to it.

So **start the server first**, then the PC bridge.

## On the server

```bash
cd ~/cyberruner-main
./run_server_dreamer_stuck_gpu0.sh
```

It sets the reward/anti-cheat environment, then runs
`python -m dreamerv3.train --configs cyberrunner large --task
gym_cyberrunner_dreamer:cyberrunner-ros-v0`. Override paths with env vars rather
than editing it:

```bash
PROJECT_ROOT=/home/tbt589/cyberruner-main \
CYBERRUNNER_LOGDIR=/home/tbt589/cyberrunner_logs/my_run \
./run_server_dreamer_stuck_gpu0.sh
```

Wait for `[TCP ENV] waiting for PC bridge` before starting the PC side.

## On the PC — four terminals

Open each in a **new** terminal so `~/.bashrc` is applied (it puts this
workspace ahead of `~/cyberrunner_ws` for `cyberrunner_state_estimation`).

```bash
# 1. camera
cd ~/CYBER/cyberruner-main
python3 fast_camera_publisher_v2.py

# 2. state estimator
cd ~/CYBER/cyberruner-main
./run_ai_map_estimator.sh

# 3. motors
cd ~/CYBER/cyberruner-main
ros2 run cyberrunner_dynamixel cyberrunner_dynamixel

# 4. TCP bridge to the server   (replace with the server's IP)
cd ~/CYBER/cyberruner-main
python3 tcp_ros_bridge.py 192.168.1.50 5555
```

Then place the marble at the start position. Training begins once the server
logs step 0.

Optional fifth terminal, to watch what the policy sees:

```bash
python3 overlay_map_view_simple.py
```

## Check the estimator before you commit hours to a run

A bad estimator trains a bad policy, and the failure is silent. Two commands:

```bash
ros2 topic hz /cyberrunner_state_estimation/estimate         # want ~40 Hz
ros2 topic echo /cyberrunner_state_estimation/status --once  # want: valid
```

If the rate is ~13 Hz instead of ~40, a subscriber is using `BEST_EFFORT` QoS
and silently dropping half the 768 KB frames. If `status` is anything other than
`valid`, see the troubleshooting table in
[`cyberrunner_state_estimation/AI_MAP_ESTIMATOR.md`](../cyberrunner_state_estimation/AI_MAP_ESTIMATOR.md).

Recalibrate markers (`./run_select_estimator_markers.sh`) whenever the camera or
board has been moved. The fixed marker quad is now anchored to its calibrated
position, so after physically moving the rig it will refuse to follow until you
re-select.

## Two traps specific to this workspace

**`cyberrunner_dreamer` resolves to `~/cyberrunner_ws`, not here.** `~/.bashrc`
only promotes `cyberrunner_state_estimation`, and `install/setup.bash` *appends*
to `AMENT_PREFIX_PATH`, so the earlier workspace wins. `ros2 run
cyberrunner_dreamer train` therefore runs the OTHER copy of `env.py`. Check
before assuming:

```bash
python3 -c "from ament_index_python.packages import get_package_share_directory as g; print(g('cyberrunner_dreamer'))"
```

The server-side path avoids this entirely — `run_server_dreamer_stuck_gpu0.sh`
sets `AMENT_PREFIX_PATH` and `PYTHONPATH` explicitly, and the PC only runs
`tcp_ros_bridge.py` by file path. Prefer the TCP flow.

**Checkpoints from before tag `estimator-calibration-20260729` are
off-distribution.** That calibration made reported positions ~7% smaller in
radius and `alpha`/`beta` ~1.9x smaller. Policies trained on the old scale will
behave differently; retrain or at least re-evaluate rather than resuming.

## Useful environment variables

Set on the **server**, since the reward and termination logic lives in the gym
env. `run_server_dreamer_stuck_gpu0.sh` sets sensible values for all of them.

| variable | purpose |
|---|---|
| `CYBERRUNNER_LOGDIR` | checkpoint and metrics directory |
| `CYBERRUNNER_REWARD_ON_FAIL` | penalty when the marble is lost |
| `CYBERRUNNER_TIMEOUT_STEPS` / `_PENALTY` | episode cap and its penalty |
| `CYBERRUNNER_ANTICHEAT_MAX_STEP_M` / `_PENALTY` | rejects implausible position jumps |
| `CYBERRUNNER_STUCK_WINDOW_SEC` / `_RADIUS_M` / `_PENALTY` | stuck-marble detection |
| `CYBERRUNNER_BALL_LOSS_GRACE_SEC` | grace before a loss counts |
| `CYBERRUNNER_TCP_PORT` | must match the port given to `tcp_ros_bridge.py` |

On the **PC**, `tcp_ros_bridge.py` reads `CYBERRUNNER_MAX_CMD_1`/`_2` (motor
command clamps) and `CYBERRUNNER_BALL_LOST_RESET_FRAMES`.

## Evaluating a trained policy

```bash
ros2 run cyberrunner_dreamer eval      # note the workspace trap above
```

Motor-only smoke test, no learning:

```bash
ros2 run cyberrunner_dreamer test
```
