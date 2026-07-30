#!/usr/bin/env bash
set -eo pipefail

# Where the TAG workspace is checked out ON THE SERVER. Override as needed.
PROJECT_ROOT="${PROJECT_ROOT:-$HOME/tag}"
LOGDIR="${TAG_LOGDIR:-$HOME/tag_logs/run1}"
# NOTE: this is a conda/micromamba ENVIRONMENT name on the server, not a ROS
# package -- it is deliberately not renamed to tag_*. Override with TAG_PYTHON.
PYTHON_BIN="${TAG_PYTHON:-/home/tbt589/micromamba/envs/cyberrunner_ros/bin/python3}"

cd "$PROJECT_ROOT"

export PYTHONUNBUFFERED=1
# Only tag_dreamer is needed: the TCP gym env imports no ROS messages, just
# ament_index_python to find its share dir. So the server never has to build
# tag_interfaces (it has no ROS/colcon anyway).
export AMENT_PREFIX_PATH="$PROJECT_ROOT/install/tag_dreamer"
# $PROJECT_ROOT/dreamerv3 MUST come first: dreamerv3 is pip-installed editable
# on this server pointing at the old cyberruner-main checkout, so without this
# shadow "-m dreamerv3.train --configs tag" loads that copy and fails with an
# unknown config (it only has a "cyberrunner:" profile).
export PYTHONPATH="$PROJECT_ROOT/dreamerv3:$PROJECT_ROOT/install/tag_dreamer/lib/python3.11/site-packages${PYTHONPATH:+:$PYTHONPATH}"

# One frame without a detected marble used to end the episode and charge
# reward_on_fail: measured 166 losses over 100 episodes, every one reported
# grace=0.00s. At the measured 25 Hz control rate a frame is 40 ms, so 0.10 s
# absorbs a two-frame detector dropout and nothing longer -- a marble actually
# down a hole stays undetected far past that and still terminates.
#
# The cost is real and is why this is 0.10 and not the 0.35 the README quotes:
# inside the grace window env_tcp feeds last_valid_ball_pos + velocity * dt
# instead of a measurement, so the window's frames are extrapolated, not
# observed. 0.10 s caps that at ~3 frames per episode end.
export TAG_BALL_LOSS_GRACE_SEC=0.10
# Left at 0 on purpose: the longer occlusion grace only applies inside the
# zones configured below, and both zone lists are empty, so all 166 losses
# were classified source=outside_zone and never consulted this value.
export TAG_OCCLUSION_GRACE_SEC=0
export TAG_OCCLUSION_XY_ZONES=""
export TAG_OCCLUSION_CHECKPOINT_RANGES=""
unset TAG_OCCLUSION_ZONES_FILE

# How far off the path line the marble may be and still be given a progress
# index. This defaults to ball_radius (6 mm), which turns out to be the single
# biggest throttle on the reward signal: measured over 242664 recorded marble
# positions the distance to the nearest path point is 5.5 mm (p50) but 10.7 mm
# (p90), so 6 mm scores only 52.6% of frames. The other 47.4% get no progress
# credit AND a zeroed goal vector, i.e. no steering hint either. The
# precomputed grid inside path_custom.pkl is meant to cover the corridor and
# would make this moot, but it returns a valid index for only 5.5% of real
# positions, so this tolerance is carrying the whole lookup.
#
# A position is ambiguous when a second corridor more than 150 indices away
# sits within 3 mm of the nearest one, i.e. detector noise could pick the wrong
# corridor and the goal vector would point down it. Measured over 246760 real
# positions, aggregated across everything the tolerance accepts:
#
#    <=12 mm   93.3% of frames scored   0.0% of them ambiguous
#    <=24 mm   97.8% of frames scored   0.8% of them ambiguous
#
# So 24 mm buys the last 4.5% of frames for 0.8% ambiguity overall -- the added
# 12-24 mm band is 4.4% of frames at 9.5% ambiguity, i.e. 0.42% of all frames
# misassigned. Worth it: a scored frame with an approximate corridor beats an
# unscored one, and the anti-cheat's path-per-metre-rolled test independently
# refuses to pay for a claim the marble did not roll for.
#
# Do not go further and drop the limit entirely to match
# overlay_map_view_simple.py:420, which falls back to an unbounded argmin. The
# farthest position ever observed is 45.6 mm, and the 24-45 mm range is a marble
# in a hole or off the board -- real junk, worth excluding. Unbounded is fine
# for the overlay because nothing is scored from it.
#
# The worst-case geometric bound is far tighter -- |A-B| <= 2T against a 20 mm
# minimum corridor separation argues for T < 10 mm -- but that bounds a marble
# sitting exactly between two corridors, and the measured distribution says that
# is 0.8% of frames at 24 mm, not the common case.
export TAG_PATH_TOLERANCE_M=0.024

# Penalty for losing the marble down a hole. Sized against the reward scale: the
# full path is worth only 9294 pts * 0.004/16 = 2.324, so -0.20 was 8.6% of a
# perfect run and needed 160 mm of path to earn back. At the ~3% progress reached
# early in training that made EVERY episode return negative no matter how well the
# marble played, which is what the -0.16..-0.20 returns in the log showed. -0.05 is
# 2.2% of a full run (40 mm of path) and leaves an early episode net positive, so
# progress reward can actually drive learning.
export TAG_REWARD_ON_FAIL=-0.05
export TAG_TIMEOUT_STEPS=3000
# TAG_TIMEOUT_PENALTY deliberately unset: env_tcp.py defaults it to
# reward_on_fail, so it tracks the line above. It used to be pinned at -0.20,
# which is the same inversion the comment above warns about, one level up:
# 3000 steps is ~120 s of surviving at the measured 25 Hz, and that was charged
# 4x what falling down a hole after 4 s costs. Under that ranking the cheapest
# way out of a stretch the policy cannot solve is to drop the marble
# immediately, so keeping it alive was the punished behaviour.
# Anti-cheat. The budget is dt-scaled on purpose: allowed advance along the path
# per step = min(MAX_STEP_M, max(MIN_STEP_M, MAX_SPEED_MPS * step_dt)). Do NOT
# express "flag a 10 mm skip" as MAX_STEP_M=0.010 -- step_dt here ranges 40-800 ms
# and the marble legitimately rolls 49 mm in 400 ms (p95 speed 0.122 m/s), so a
# flat cap fires on real motion. MIN_STEP_M is already the 10 mm floor below which
# nothing is ever flagged.
#
# MAX_SPEED_MPS 1.0 -> 0.3 tightens the normal-frame budget from ~50 mm to ~15 mm
# (2.5x the measured p95) while still relaxing on slow frames.
export TAG_ANTICHEAT_MAX_STEP_M=0.057
export TAG_ANTICHEAT_MIN_STEP_M=0.010
# MEASURED, do not tighten without re-checking: live single steps reach 21-31 mm
# in 35-45 ms, i.e. 0.6-0.7 m/s. The "p95 0.122 m/s" figure in env_tcp.py is stale
# for this board. At 0.3 m/s this produced 17 false triggers in 100 s and killed
# every episode at ~30 steps with the -0.50 penalty. 1.0 m/s is the known-good
# value; revisit only after marble detection is reliable.
export TAG_ANTICHEAT_MAX_SPEED_MPS=1.0
export TAG_ANTICHEAT_CONFIRM_STEPS=5
# Termination OFF. The maze does have a reachable shortcut (a 59.5 mm hop across
# the open centre skips 836 mm of path, 43.6% -> 88.6%), but credit for it is
# already denied by the corridor + frozen prev_pos_path + per-step budget. Turning
# termination on while marble detection sits near 40% valid just charges -0.50 for
# camera faults. Set to 1 once detection is fixed.
export TAG_ANTICHEAT_ENABLED=0
export TAG_ANTICHEAT_PENALTY=-0.50

export TAG_STUCK_WINDOW_SEC=5
export TAG_STUCK_RADIUS_M=0.003
export TAG_STUCK_PENALTY=0

exec "$PYTHON_BIN" -m dreamerv3.train \
  --configs tag large \
  --task gym_tag_dreamer:tag-ros-v0 \
  --logdir "$LOGDIR" \
  --replay_size 1e6 \
  --run.script train_top5 \
  --run.train_ratio 256 \
  --run.save_every 20 \
  --run.log_every 1 \
  --jax.policy_devices 0 \
  --jax.train_devices 0 \
  "$@"
