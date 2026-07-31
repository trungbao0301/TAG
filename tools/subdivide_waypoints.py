#!/usr/bin/env python3
"""Even out the checkpoint spacing by splitting over-long path segments.

The checkpoint bonus pays once per waypoint passed, so how evenly the waypoints
are spread decides how evenly that reward arrives. Measured on this path they are
not spread evenly at all: 61 segments with a median of 23.9 mm but a maximum of
120.7 mm, and eleven of them over 40 mm. The marble crosses that 120 mm stretch
earning nothing extra while elsewhere a bonus lands every 10 mm.

Raising the density uniformly is the wrong fix. The progress term already pays
0.00025 for every 0.2 mm rolled, which is 50x finer than a 10 mm waypoint, so
there is no gradient to add -- and going to 10 mm spacing would take the bonus
total from 1.22 to 4.36 against 2.324 of progress reward, making the bonus the
dominant term. Capping the long segments evens the spacing while barely moving
the count, and the per-waypoint value is then scaled to hold the total.

Only orig_waypoints is rewritten. The inserted points are collinear with the
segment they split, so the path shape is identical, and points and closest_idx
are left exactly as they are -- which matters because closest_idx carries the
hand-painted shortcut traps and is indexed to the existing path points.

    python3 tools/subdivide_waypoints.py --cap_mm 40            # report
    python3 tools/subdivide_waypoints.py --cap_mm 40 --write
"""
import argparse
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "tag_dreamer"))

PROGRESS_PER_POINT = 0.004 / 16.0


def subdivide(waypoints, cap_m):
    """Insert collinear waypoints so no segment exceeds cap_m."""
    out = [waypoints[0]]
    for start, end in zip(waypoints[:-1], waypoints[1:]):
        length = float(np.linalg.norm(end - start))
        pieces = max(1, int(np.ceil(length / cap_m))) if cap_m > 0 else 1
        for k in range(1, pieces + 1):
            out.append(start + (end - start) * (k / pieces))
    return np.asarray(out, dtype=waypoints.dtype)


def describe(label, waypoints, distance, num_points, bonus):
    seg = np.linalg.norm(np.diff(waypoints, axis=0), axis=1) * 1000.0
    total = (len(waypoints) - 1) * bonus
    progress = num_points * PROGRESS_PER_POINT
    print(
        f"  {label:<10} {len(waypoints):4d} waypoints   segment p50 {np.median(seg):5.1f} "
        f"max {seg.max():6.1f} mm   bonus {bonus:.4f} each, {total:.2f} total "
        f"({100.0 * total / progress:.0f}% of the {progress:.2f} progress reward)"
    )
    return len(waypoints) - 1


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--path",
        default=os.path.join(HERE, "..", "tag_dreamer", "data", "path_custom.pkl"),
    )
    ap.add_argument("--out", default=None, help="default: overwrite --path")
    ap.add_argument("--cap_mm", type=float, default=20.0)
    ap.add_argument(
        "--base_path", default=None,
        help="take the waypoints to subdivide from this pickle instead of --path, "
             "so re-running with a different cap starts from the original spacing "
             "rather than splitting an already-split path",
    )
    ap.add_argument(
        "--bonus", type=float, default=0.02,
        help="current per-waypoint bonus, used to report the scaled value",
    )
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    with open(args.path, "rb") as fh:
        p = pickle.load(fh)

    if args.base_path:
        with open(args.base_path, "rb") as fh:
            before = np.asarray(pickle.load(fh).orig_waypoints)
    else:
        before = np.asarray(p.orig_waypoints)
    after = subdivide(before, args.cap_mm / 1000.0)

    n_before = describe("before", before, p.distance, p.num_points, args.bonus)
    scaled = args.bonus * n_before / max(1, len(after) - 1)
    describe("after", after, p.distance, p.num_points, scaled)
    print(f"\n  set TAG_CHECKPOINT_BONUS={scaled:.4f} to hold the total where it is")

    # The path itself must not move: every inserted point is on the segment it
    # splits, so the polyline is unchanged and closest_idx stays valid.
    length_before = float(np.linalg.norm(np.diff(before, axis=0), axis=1).sum())
    length_after = float(np.linalg.norm(np.diff(after, axis=0), axis=1).sum())
    print(
        f"  polyline length {1000 * length_before:.2f} -> {1000 * length_after:.2f} mm "
        f"(must be unchanged)"
    )
    # 1 micron. A nanometre bound fails on float32 accumulation across a hundred
    # segments while the geometry is identical: measured, 123 waypoints drift the
    # total by 0.0007 um and no inserted point sits more than 0.0075 um off the
    # segment it splits.
    if abs(length_after - length_before) > 1e-6:
        print("  refusing to write: the path shape moved", file=sys.stderr)
        return 1

    if args.write:
        p.orig_waypoints = after
        out = args.out or args.path
        with open(out, "wb") as fh:
            pickle.dump(p, fh, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"  wrote {os.path.abspath(out)} (points and closest_idx untouched)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
