"""Generate a random path to follow, for map-agnostic path-tracking training.

env_tcp.py's TagGym scores progress against one path baked once from the real
board's DXF (tag_dreamer/data/path_custom.pkl). A policy trained only against
that one path has no reason to generalize to a path it has never seen. Here
the path itself is the domain-randomization axis: a fresh one is generated
every episode, so the only way to do well on average is to track a path's
*shape* rather than memorize one corridor.

Coordinates are lower-left board metres, matching LinearPath and the maze
layouts' convention.
"""
import numpy as np

MARBLE_RADIUS_M = 0.006


def generate_waypoints(
    rng,
    board_width,
    board_height,
    start=None,
    margin=MARBLE_RADIUS_M + 0.01,
    num_segments=12,
    min_step=0.02,
    max_step=0.05,
    max_turn_rad=1.0,
):
    """A random-walk polyline with heading persistence, kept inside the board.

    `max_turn_rad` bounds how sharply the path can turn per segment, so the
    result stays inside what a tilted plate could plausibly steer a rolling
    ball along -- not a zig-zag no policy could ever track exactly.
    """
    lo = np.array([margin, margin])
    hi = np.array([board_width - margin, board_height - margin])
    pos = np.asarray(start, dtype=np.float64) if start is not None else (lo + hi) / 2.0
    pos = np.clip(pos, lo, hi)
    heading = float(rng.uniform(-np.pi, np.pi))
    waypoints = [pos.copy()]
    for _ in range(num_segments):
        for _attempt in range(8):
            candidate_heading = heading + rng.uniform(-max_turn_rad, max_turn_rad)
            step = rng.uniform(min_step, max_step)
            nxt = pos + step * np.array([np.cos(candidate_heading), np.sin(candidate_heading)])
            if np.all(nxt >= lo) and np.all(nxt <= hi):
                heading = candidate_heading
                break
        else:
            # Every heading tried this step left the board -- turn back toward
            # the centre instead of clipping onto the rim and bunching points
            # there, which would make "the path" mostly a wall-hugging line.
            centre = (lo + hi) / 2.0
            delta = centre - pos
            heading = float(np.arctan2(delta[1], delta[0]))
            nxt = pos + min_step * np.array([np.cos(heading), np.sin(heading)])
        waypoints.append(nxt.copy())
        pos = nxt
    return np.asarray(waypoints, dtype=np.float32)
