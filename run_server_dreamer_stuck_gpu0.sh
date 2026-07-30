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

# How long a marble may go undetected before the episode is failed. This started
# at 0, where a single missed frame ended the episode -- 166 losses over 100
# episodes, every one reporting grace=0.00s.
#
# 0.20 s, raised from 0.10 s. 0.10 s was sized against a 25 Hz loop, i.e. 2-3
# frames, but the loop actually runs at p10 19.3 / p50 25.3 fps, so a frame is
# 31-52 ms and 0.10 s buys only two of them. Every measured loss confirmed at
# 0.10-0.15 s, which is the first frame past the grace rather than evidence the
# marble was really gone -- and episodes were ending with the marble plainly
# still on the board and nowhere near a hole. 0.20 s is 5-6 frames.
#
# The cost is that the grace window feeds predicted positions rather than
# measured ones, so it doubles from ~3 to ~6 substituted frames per gap. That is
# safer than it was: the reference can no longer be teleported on reacquire,
# because the resync now refuses an index further away than max_speed * gap.
export TAG_BALL_LOSS_GRACE_SEC=0.20
# The grace that actually decides it, in frames rather than seconds. A dropout is
# a count of frames the detector missed, so tolerating it in seconds means the
# tolerance moves with the loop rate: at p10 19.3 fps the 0.20 s above is 3.9
# frames and at p90 32.2 fps it is 6.4, so the same dropout ended the episode or
# not depending on how busy the machine was. 6 frames is fixed. The seconds value
# stays as a ceiling, because the grace window substitutes predicted positions
# for measured ones and on a very slow frame that prediction is worthless.
export TAG_BALL_LOSS_GRACE_FRAMES=6

# Paid once the first time each of the 61 waypoints is passed, on top of the
# per-millimetre progress term. The progress term is smooth but gives nothing
# extra for clearing the specific spots the marble keeps failing at, and the
# measured distribution says it fails at a few specific spots rather than
# uniformly. 0.02 each is 1.22 over a full run against 2.324 of progress reward,
# and about a third of what an episode currently earns, so it is worth aiming at.
# Gated on the best checkpoint reached in the episode, so rolling back and forth
# across a boundary pays nothing.
export TAG_CHECKPOINT_BONUS=0.02
# Left at 0 on purpose: the longer occlusion grace only applies inside the
# zones configured below, and both zone lists are empty, so all 166 losses
# were classified source=default_grace and never consulted this value.
export TAG_OCCLUSION_GRACE_SEC=0
export TAG_OCCLUSION_XY_ZONES=""
export TAG_OCCLUSION_CHECKPOINT_RANGES=""
unset TAG_OCCLUSION_ZONES_FILE

# 0 disables the geometric repair in _closest_point, so a cell the grid marks
# -1 stays -1 instead of being handed the nearest path point anyway.
#
# That repair was only ever needed because the shipped grid was broken: it
# credited 16.6% of recorded marble positions, so without a tolerance the maze
# was unplayable, and the tolerance had to be pushed to 24 mm to reach 97.8%.
# But the repair reads a path index for cells the grid deliberately blanked --
# including the ridges between corridors -- which is exactly how the 25 mm hop
# onto a corridor 811 mm further along became payable. The workaround defeated
# the mechanism it was working around.
#
# tools/rebuild_path_grid.py fixes the grid instead. Restoring cells within
# 10 mm of their own corridor takes credited marble frames from 16.6% to 80.4%
# and the path's own centreline from 89.7% to 100%, while leaving zero adjacent
# credited pairs whose index jumps more than 57 mm of path -- so no hop is
# payable and no tolerance is needed. Thresholds past 10 mm buy almost nothing
# and break that: 12 mm adds 0.9 points of marble coverage and 2 jump pairs.
export TAG_PATH_TOLERANCE_M=0

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
# 1, not 3: a confirmed jump ends the episode on the frame it happens. Asked for
# directly. The reason it was 3 no longer holds the same weight -- back then this
# rule was the only thing standing between the marble and an 811 mm hop, so a
# false strike had to be traded against missing a real one. Now the grid blanks
# the crossings, so this rule is left guarding detector flips, and a flip that
# only lasts one frame will now end the episode.
#
# Watch [Done]: ANTI-CHEAT PROGRESS JUMP against [Done]: OFFPATH in the log. At 3
# it fired on 1-2 of every ~22 episodes. If at 1 it climbs past that, the flips
# are being charged, and 2 is the smaller step back rather than 3.
export TAG_ANTICHEAT_CONFIRM_STEPS=1
# 0: the 57 mm one-step rule no longer ends an episode or charges anything. It is
# not needed to stop shortcuts any more -- the grid blanks the crossings and
# off_path ends the episode on them -- and what it was actually doing was charging
# -0.15 for detector flips, on 4 of 23 episodes, which is about -0.026 of average
# return against the +0.05 an episode earns.
#
# TAG_ALLOW_CHEAT stays 0, so the rule still REFUSES TO PAY for a 57 mm one-step
# advance even though it no longer punishes it. That half costs nothing and is
# worth keeping: a detector flip that lands on a credited cell 100 mm along the
# path would otherwise be paid +0.125, which is more than an episode earns, and
# fabricated reward is the failure mode this whole file has been chasing. Set
# TAG_ALLOW_CHEAT=1 to remove the rule completely.
export TAG_ANTICHEAT_ENABLED=0
# -0.15, three times the -0.05 charged for a hole. It has to be the heavier of
# the two: they both end the episode, so if cheating cost the same or less there
# would be no reason to prefer driving the maze. Against a full path worth 2.324
# it is 6.5%, and against what an episode currently earns (+0.03 to +0.07) it is
# two to five episodes -- enough to deter without one false strike dominating the
# value function the way -0.50 would.
export TAG_ANTICHEAT_PENALTY=-0.15
export TAG_ALLOW_CHEAT=0

# Ratio test off: the grid now blocks corridor hops structurally, so the only
# numeric rule still needed is the flat one-step cap the original env used,
# which TAG_ANTICHEAT_MAX_STEP_M above already sets to its 0.057 m. Setting the
# ratio to 0 selects exactly that. The ratio was a workaround for a grid that
# could not block hops itself, and it never found a threshold that both caught
# the hop and left honest motion alone -- at 3.0 with confirm 3 it ended 4 of 23
# episodes.
export TAG_ANTICHEAT_TRAVEL_RATIO=0

# Off-path termination, which is the original env's entire anti-shortcut rule:
# a cell the grid will not credit ends the episode. A hop between corridors has
# to cross those cells, so it cannot be taken at all rather than having to be
# detected.
#
# 1 frame: touching a blanked cell ends the episode, which is what the original
# env did and what makes the trap have teeth. At anything higher the marble can
# cross a blanked ridge and come out the other side unpunished -- at 10 it had
# 0.4 s of grace, which is longer than a crossing takes.
#
# 10 was a stopgap from when reported y was wrong. At 2 back then it ended 633 of
# 644 episodes with a median length of 3 steps, because 47.6% of episode-start
# frames came back above the board's 229 mm edge, outside the grid entirely, so
# episodes began off-path. marker_plane_height_m 0 fixed that: starts inside the
# corridor went 52.0% -> 92.9% and episodes ended by OFFPATH went 49/57 -> 0.
#
# That 0 was measured at 10, not at 1, so this value is not yet backed by a
# measurement. If OFFPATH starts firing on single glitch frames, raise to 3 or 4
# rather than back to 10.
export TAG_OFFPATH_CONFIRM_STEPS=1
export TAG_OFFPATH_PENALTY=-0.05

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
