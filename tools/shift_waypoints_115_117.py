#!/usr/bin/env python3
"""Shift the 115-117 leg sideways, keeping it straight.

The physical maze is not changed. Waypoints 115, 116 and 117 form the vertical
leg at x = 228.0 mm that the path takes to get around hole 8; this moves all
three by the same amount so the leg stays straight and only its distance from
that hole changes.

Measured before writing this: the leg is not where the path is tight. The
narrowest points in this region are 112->113 and 113->114, at 0.60 and 0.58 mm
of marble-edge-to-hole-edge clearance, and neither is touched by moving 115-117.
The leg itself already has 4.4-5.7 mm. Moving it RIGHT closes on hole 8 at
(249, 189) mm: +4 mm leaves 3.9 mm, +6 mm leaves 1.9 mm, and +8 mm puts the path
through the hole. So the safe range in that direction is small, and the script
refuses to write a path whose clearance anywhere in 113->118 drops below the
1.74 mm that the marble is known to have cleared elsewhere.

    python3 tools/shift_waypoints_115_117.py --path in.pkl --out out.pkl --shift_mm 4
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "tag_dreamer"))

from tag_dreamer.path import LinearPath  # noqa: E402
import tag_dreamer.tag_layout_custom as layout_module  # noqa: E402

LAYOUT = layout_module.tag_dxf_layout
HOLES = np.asarray(LAYOUT["holes"], dtype=np.float64)
HOLE_RADII = np.asarray(LAYOUT["hole_radii"], dtype=np.float64)
BALL_R = float(LAYOUT["ball_radius"])

TARGETS = (115, 116, 117)
# The tightest gap the marble is known to have actually cleared, measured over
# the whole path. A new gap below this is a new bottleneck, not a fix.
KNOWN_PASSABLE_MM = 1.74


def clearance_mm(start, end, step=0.0002):
    """Smallest marble-edge to hole-edge clearance along a segment."""
    span = float(np.linalg.norm(end - start))
    samples = max(2, int(span / step))
    worst = float("inf")
    for t in np.linspace(0.0, 1.0, samples):
        point = start + t * (end - start)
        gaps = np.linalg.norm(HOLES - point, axis=1) - HOLE_RADII - BALL_R
        worst = min(worst, float(gaps.min()) * 1000.0)
    return worst


# Segments whose geometry this move actually changes. 112->113 and 113->114 are
# printed for context but are not the script's business: they are upstream of
# the leg and stay exactly where they were.
EDITED = (114, 115, 116, 117)


def report(waypoints, label):
    print(f"  {label}")
    gaps = {}
    for index in range(112, 118):
        gap = clearance_mm(waypoints[index], waypoints[index + 1])
        gaps[index] = gap
        mark = "" if index in EDITED else "   (khong doi)"
        print(f"    wp{index:3d}->{index + 1:3d} : {gap:6.2f} mm{mark}")
    return min(gaps[i] for i in EDITED)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--shift_mm",
        type=float,
        required=True,
        help="Positive moves the leg right (+x), toward hole 8.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Write even if the result is tighter than a known-passable gap.",
    )
    args = parser.parse_args()

    with open(args.path, "rb") as stream:
        old = pickle.load(stream)

    waypoints = np.asarray(old.orig_waypoints, dtype=np.float32).copy()
    if len(waypoints) <= max(TARGETS):
        raise RuntimeError(f"path has only {len(waypoints)} waypoints")

    xs = {index: float(waypoints[index][0]) for index in TARGETS}
    if max(xs.values()) - min(xs.values()) > 1e-6:
        raise RuntimeError(
            f"waypoints {TARGETS} are not collinear in x: {xs}; refusing to "
            "shift them as a straight leg"
        )

    before = report(waypoints, "before:")
    for index in TARGETS:
        waypoints[index][0] += args.shift_mm / 1000.0
    after = report(waypoints, f"after {args.shift_mm:+.1f} mm:")

    print(f"  narrowest AMONG THE EDITED segments: {before:.2f} -> {after:.2f} mm")
    # Judge the move on what it changes. The 0.58 mm at 113->114 is upstream of
    # the leg and identical either way, so testing against it would refuse every
    # shift including the ones that improve things.
    if after < KNOWN_PASSABLE_MM and not args.force:
        raise SystemExit(
            f"refusing: this move leaves {after:.2f} mm on an edited segment, "
            f"tighter than the {KNOWN_PASSABLE_MM} mm the marble is known to "
            "clear. Re-run with --force to override."
        )

    new = LinearPath(
        waypoints,
        distance=old.distance,
        board_width=old.width,
        board_height=old.height,
        wall_r=old.wall_r,
    )
    new.closest_dim_x = int(old.width / old.distance) + 1
    new.closest_dim_y = int(old.height / old.distance) + 1
    if old.closest_idx is None:
        raise RuntimeError("input path has no progress grid to preserve")
    # Same treatment as adjust_checkpoint_104_corner: keep which cells are
    # valid, so off-path termination behaviour does not change too, and only
    # remap the stored indices onto the new points.
    old_to_new = cKDTree(new.points).query(old.points)[1].astype(np.int32)
    new.closest_idx = np.full(old.closest_idx.shape, -1, dtype=np.int32)
    valid = old.closest_idx >= 0
    new.closest_idx[valid] = old_to_new[old.closest_idx[valid]]

    with open(args.out, "wb") as stream:
        pickle.dump(new, stream, protocol=pickle.HIGHEST_PROTOCOL)

    for index in TARGETS:
        print(
            f"  wp{index}: x {xs[index] * 1000:.2f} -> "
            f"{waypoints[index][0] * 1000:.2f} mm"
        )
    print(
        f"  checkpoints {len(old.orig_waypoints) - 1} -> "
        f"{len(new.orig_waypoints) - 1}; points {old.num_points} -> "
        f"{new.num_points}; valid cells {100.0 * valid.mean():.2f}%"
    )
    print(f"  wrote {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()
