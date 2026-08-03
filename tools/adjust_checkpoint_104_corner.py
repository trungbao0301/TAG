#!/usr/bin/env python3
"""Move only the virtual corner before checkpoint 104 away from hole 2.

The physical maze is not changed. The old right-angle corner at
(206.20, 196.24) mm leaves less clearance than a 12 mm marble needs around the
15 mm hole. Moving that one virtual waypoint left makes the policy begin its
left turn earlier while preserving every other waypoint and checkpoint count.

The old grid's valid/invalid mask is preserved so this focused geometry fix
does not also change off-path termination behavior. Stored path indices are
remapped to the nearest point on the new path.
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "tag_dreamer"))

from tag_dreamer.path import LinearPath


OLD_CORNER = np.asarray([0.20619848, 0.19624366], dtype=np.float32)
NEW_CORNER = np.asarray([0.20080000, 0.19624366], dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    with open(args.path, "rb") as stream:
        old = pickle.load(stream)

    waypoints = np.asarray(old.orig_waypoints, dtype=np.float32).copy()
    distances = np.linalg.norm(waypoints - OLD_CORNER, axis=1)
    index = int(np.argmin(distances))
    if float(distances[index]) > 0.0005:
        raise RuntimeError(
            f"expected corner not found; nearest waypoint is "
            f"{1000.0 * distances[index]:.3f} mm away"
        )
    waypoints[index] = NEW_CORNER

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
    # Map every old path index to the spatially nearest point on the new path,
    # then apply that lookup without changing which grid cells are valid.
    old_to_new = cKDTree(new.points).query(old.points)[1].astype(np.int32)
    new.closest_idx = np.full(old.closest_idx.shape, -1, dtype=np.int32)
    valid = old.closest_idx >= 0
    new.closest_idx[valid] = old_to_new[old.closest_idx[valid]]

    with open(args.out, "wb") as stream:
        pickle.dump(new, stream, protocol=pickle.HIGHEST_PROTOCOL)

    print(
        f"moved waypoint array index {index} from "
        f"({OLD_CORNER[0]:.6f}, {OLD_CORNER[1]:.6f}) to "
        f"({NEW_CORNER[0]:.6f}, {NEW_CORNER[1]:.6f})"
    )
    print(
        f"checkpoint count {len(old.orig_waypoints) - 1} -> "
        f"{len(new.orig_waypoints) - 1}; path points "
        f"{old.num_points} -> {new.num_points}"
    )
    print(
        f"preserved progress-grid valid cells: "
        f"{100.0 * valid.mean():.2f}%"
    )
    print(f"wrote {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()
