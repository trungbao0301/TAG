# Simulated board

An Isaac Sim stand-in for the physical rig. It speaks the same TCP protocol the
robot bridge speaks, so `tag_dreamer`'s environment, reward, checkpoint and
anti-cheat logic all run unchanged — the learner cannot tell which one it is
talking to.

Use it to pretrain a policy before touching the hardware, and to debug reward
changes at 30 steps/s without having to pick the marble out of a hole by hand.

## What is here

| Path | What it is |
|---|---|
| `sim/assets/tag_board.usd` | The board: plate, maze, marble, the eight blue markers, the overhead camera, and the two tilt joints |
| `sim/assets/map.usd` | The maze itself, referenced by `tag_board.usd` (keep the two together) |
| [`tools/isaac_tcp_server.py`](../tools/isaac_tcp_server.py) | The bridge: drives the board and answers the learner |

The maze mesh came from `maprawv2.STEP`; the same geometry appears in
`tag_dreamer/data/map.DXF` and `tag_state_estimation/.../maze_layout.py`, and all
three agree to within 0.1 mm (checked by rasterising the mesh and matching its
21 circular voids against the layout's hole centres).

## Requirements

- **Isaac Sim 4.5** as a pip package, in its own Python 3.10 environment. Isaac
  Lab is *not* needed.
- A CUDA GPU for rendering. Physics runs on the **CPU** here, deliberately: with
  GPU dynamics the SDF collider fills the 15 mm holes, so a marble never falls
  in (measured: 0/4 drops with GPU, 4/4 with CPU). GPU physics is also 2.5x
  *slower* for a scene this small.
- No ROS. The link to the learner is a TCP socket carrying JSON lines.

## Running it

The learner listens; the simulator dials in, exactly as `tcp_ros_bridge.py` does
on the robot. Start the learner first (or don't -- the simulator retries).

```bash
# 1. the learner, on one GPU
CUDA_VISIBLE_DEVICES=0 TAG_LOGDIR=~/tag_logs/sim TAG_TCP_PORT=5556 \
    ./run_server_dreamer_stuck_gpu0.sh

# 2. the board, on another
CUDA_VISIBLE_DEVICES=1 <isaac-python> -u tools/isaac_tcp_server.py \
    --stage sim/assets/tag_board.usd \
    --port 5556 \
    --image composite \
    --overlay-port 8088
```

Then open `http://<host>:8088` for a live map view: the maze, the 21 holes, the
target path, and the marble where the learner sees it. `http://<host>:8088/camera.png`
serves the full-resolution camera frame when `--image camera` is on.

## The options that matter

**`--image`** — what goes into the observation's image channel.

| Mode | Cost | What the world model's CNN learns |
|---|---|---|
| `camera` | 38 steps/s | A real render. Use for recording, or if the policy must work from pixels |
| `composite` | ~free | The maze rendered *once*, with the marble drawn on each step. Recommended |
| `black` | fastest | Nothing: a constant frame has no gradient |

Rendering costs the same per frame whatever the resolution — 38 steps/s at
640x400 against 53 at 128x80, versus 289 with no render at all — so lowering the
resolution is not the way to buy speed; not rendering every step is.

**`--reset-mode`** — where each episode starts.

`fixed` puts the marble where the robot's reset puts it. `spread` starts it
anywhere along the first 60% of the path (still 30% of the time at the true
start, so the opening is not forgotten). Use `spread` once the policy reliably
reaches some point and dies there: on this map 74 of 200 episodes died between
40% and 50% of the path, a stretch that passes within 8 mm of a hole, and which
a from-the-start-only run only reaches after surviving everything before it.

**`--axis-map`** — which board axis servo 1 drives. `1beta` matches the only
experimental evidence available. Settle it on the real rig: with the marble
removed, hold `cmd_1 = +200` for two seconds and see which of alpha/beta moves.

## Fidelity: what is faithful and what is not

Taken from the hardware and checked:

- The command chain is the Hiwonder driver's, including its rate limit:
  `pos = clip(500 + 1.5*vel, 100, 900)`, moved at most 20 units per 1/30 s tick.
- Board, maze, marble and marker geometry match the measured board to <0.1 mm.
- The camera sits at the calibrated 0.29 m and renders at the real 640x400.
- `x_b, y_b` are board-centre metres and the image is 64x64 grey, because that
  is what `StateEstimate` carries and what `imgmsg_to_gray64` produces.

Known gaps, all of which matter for sim-to-real:

- **`DEG_PER_UNIT = 0.018`** (servo units to degrees of tilt) is inferred from
  the angle range of one successful run, not measured. Everything about how the
  board *feels* scales with it.
- **The marble position is ground truth**, not an estimate. The real policy sees
  the output of the vision stack, including its dropouts and its 3-frame
  reacquisition delay. A policy trained here has never seen the marble go
  missing for a frame.
- **No sensor noise, blur, or lighting variation** in the rendered image.
