# Pendulum occlusion-zone selector

This passive tool subscribes only to `/cyberrunner_camera/image`. It does not
publish ROS messages or send motor commands.

Run it after the camera, maze, pendulum, and corner-marker calibration are in
their final positions:

```bash
cd /home/trungbao/CYBER/cyberruner-main
./run_pendulum_zone_selector.sh
```

Keep the board near its home position. In the camera window:

- Drag with the left mouse button to add a pendulum-covered rectangle.
- Press `U` to undo the last rectangle.
- Press `C` to clear all rectangles.
- Press `S` to save.
- Press `Q` or Escape to quit.

The default output is `pendulum_occlusion_zones.json`. The red regions shown
after selection are projected through the current moving-marker geometry, so
they represent board-relative areas rather than fixed image pixels.

To use the saved zones in `env_tcp`, copy the JSON to the training server and
set these variables before starting Dreamer:

```bash
export CYBERRUNNER_BALL_LOSS_GRACE_SEC=0
export CYBERRUNNER_OCCLUSION_GRACE_SEC=1.5
export CYBERRUNNER_OCCLUSION_ZONES_FILE=/absolute/path/pendulum_occlusion_zones.json
```

Loss outside every saved zone remains immediate. Loss beginning inside a zone
is tolerated only for `CYBERRUNNER_OCCLUSION_GRACE_SEC`; if the marble is not
reacquired before that time, the episode ends with the normal failure penalty.

If the camera or physical marker positions move, recalibrate the markers and
select the pendulum zones again.
