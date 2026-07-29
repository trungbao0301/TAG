#!/usr/bin/env python3
"""Verify the installed maze's hole positions against the layout file.

What this is for
----------------
Cross-checking that cyberrunner_dxf_layout / maze_layout.py actually describe
the maze that is physically installed. On this rig it confirmed they DO:
maprawv2.DXF (sha256 bf2632c8..., matching the layout's source_map_sha256) has
21 circles of r=7.5 mm spanning x 8.75-249.46, y 8.52-220.78 mm, and the
measured board holes land a median 4.1 mm away with no fitting at all.

Do not trust a large mismatch here without first checking the dot detection. An
earlier version of this tool reported a 21.7 mm median and "the maze does not
match", which was false: a bare Detector with its default
corner_subimage_half_size of 25 locked corner 2 onto a FIXED frame marker 26 px
away, inflating the dot x span from 277 to 301 px and skewing every board
coordinate. MarkerQuadGuard rejects that, and the mismatch dropped to 4.1 mm.

How
---
The four blue dots have a known centre spacing (269 x 237 mm), so
getPerspectiveTransform gives a metric image->board map involving NO camera
model -- it cannot inherit the intrinsics error this repo is still resolving.
Map every detected hole blob through it and average across frames.

Two limitations worth knowing:

  * The result inherits the 269/237 dot-spacing constants as a global scale. If
    those are wrong, every measured position scales with them. Measure your dots
    with calipers to pin this down.
  * The holes are on the floor, the dots slightly above it, so a tilted view
    adds a radial parallax offset of about r * h / H. This prefers the most
    fronto-parallel frames, where that term is smallest, and reports the
    cross-frame spread so you can see what is left.

Usage
-----
    python3 tools/record_frames.py --seconds 60 --out /tmp/frames
    python3 tools/measure_hole_layout.py --frames_dir /tmp/frames
"""

import argparse
import glob
import os
import sys

import cv2
import numpy as np

from cyberrunner_state_estimation.core.hole_mask import (
    HOLES_CENTERED_M,
    HOLE_RADII_M,
    MARKERS_CENTERED_M,
)
from cyberrunner_state_estimation.core.ai_map_state import MarkerQuadGuard
from cyberrunner_state_estimation.core.detection import Detector
from cyberrunner_state_estimation.core.maze_layout import (
    BOARD_HEIGHT_M,
    BOARD_WIDTH_M,
)


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PACKAGE_DIR = os.path.join(REPO_ROOT, "cyberrunner_state_estimation")


def detect_hole_blobs(frame, px_per_m, radius_m, dark_max=70):
    """Every dark, roughly round blob of about the right size, in pixels."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    dark = (gray < dark_max).astype(np.uint8)
    count, _, stats, centroids = cv2.connectedComponentsWithStats(dark, 8)
    expected = np.pi * (radius_m * px_per_m) ** 2
    found = []
    for index in range(1, count):
        area = float(stats[index, cv2.CC_STAT_AREA])
        width = float(stats[index, cv2.CC_STAT_WIDTH])
        height = float(stats[index, cv2.CC_STAT_HEIGHT])
        if not 0.45 * expected < area < 2.2 * expected:
            continue
        if not 0.65 < width / max(height, 1.0) < 1.55:
            continue
        found.append(centroids[index])
    return np.asarray(found, dtype=np.float32).reshape(-1, 2)


def make_tracker(markers):
    """Tight search window plus the estimator's guard.

    Without both, corner 2 locks onto a fixed frame marker and every measured
    hole position is skewed -- see the module docstring.
    """
    tracker = Detector(markers[4:], ai_mode="off", corner_subimage_half_size=12)
    guard = MarkerQuadGuard(
        np.asarray(markers[4:], dtype=np.float64)[:, ::-1], mode="moving"
    )
    return tracker, guard, [0]


def frame_measurements(tracked, frame, radius_m):
    """Board-frame hole positions for one frame, plus a levelness score."""
    tracker, guard, counter = tracked
    raw = tracker.detect_corners(frame)
    corners_rc, valid, _ = guard.update(
        raw, tracker.corner_found, counter[0] / 50.0
    )
    counter[0] += 1
    tracker.corners = corners_rc.astype(np.float32)
    tracker.corners_missing = False
    if not valid:
        return None, None
    corners_xy = np.asarray(corners_rc, dtype=np.float32)[:, ::-1]
    board_to_image = cv2.getPerspectiveTransform(
        MARKERS_CENTERED_M.astype(np.float32), corners_xy
    )
    px_per_m = float(np.linalg.norm(corners_xy[1] - corners_xy[0])) / float(
        MARKERS_CENTERED_M[1, 0] - MARKERS_CENTERED_M[0, 0]
    )
    blobs = detect_hole_blobs(frame, px_per_m, radius_m)
    if len(blobs) < 8:
        return None, None
    board = cv2.perspectiveTransform(
        blobs.reshape(-1, 1, 2), np.linalg.inv(board_to_image)
    ).reshape(-1, 2)
    # The homography's last row carries the perspective terms; they vanish for a
    # fronto-parallel plane, so their magnitude ranks how level this view is.
    normalised = board_to_image / board_to_image[2, 2]
    levelness = float(np.hypot(normalised[2, 0], normalised[2, 1]))
    return board, levelness


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames_dir", required=True)
    parser.add_argument(
        "--markers", default=os.path.join(PACKAGE_DIR, "markers.csv")
    )
    parser.add_argument(
        "--use_frames", type=int, default=15,
        help="How many of the most fronto-parallel frames to average."
    )
    parser.add_argument("--radius_m", type=float, default=float(HOLE_RADII_M[0]))
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    paths = sorted(
        glob.glob(os.path.join(args.frames_dir, "*.png"))
        + glob.glob(os.path.join(args.frames_dir, "*.jpg"))
    )
    if not paths:
        print(f"no frames in {args.frames_dir}", file=sys.stderr)
        return 1
    markers = np.loadtxt(args.markers, delimiter=",")
    tracked = make_tracker(markers)

    views = []
    for path in paths:
        frame = cv2.imread(path)
        if frame is None:
            continue
        board, levelness = frame_measurements(tracked, frame, args.radius_m)
        if board is not None:
            views.append((levelness, board))
    if not views:
        print("no frame yielded a usable dot quad plus blobs", file=sys.stderr)
        return 1
    views.sort(key=lambda item: item[0])
    views = views[: max(1, args.use_frames)]
    print("frames read %d, usable %d, averaging the %d most level"
          % (len(paths), len(views), len(views)))

    # Reference = most fronto-parallel view; match the rest to it by nearest
    # neighbour, so a frame that misses a hole simply contributes nothing there.
    reference = views[0][1]
    sums = reference.copy()
    counts = np.ones(len(reference))
    for _, board in views[1:]:
        distance = np.linalg.norm(
            reference[:, None, :] - board[None, :, :], axis=2
        )
        for index in range(len(reference)):
            nearest = int(distance[index].argmin())
            if distance[index, nearest] < 0.008:
                sums[index] += board[nearest]
                counts[index] += 1
    measured = sums / counts[:, None]

    spread = []
    for _, board in views:
        distance = np.linalg.norm(
            measured[:, None, :] - board[None, :, :], axis=2
        ).min(axis=1)
        spread.append(distance)
    spread = np.asarray(spread)
    print("cross-frame spread    : median %.2f mm  p95 %.2f mm"
          % (np.median(spread) * 1000, np.percentile(spread, 95) * 1000))
    print("holes measured        : %d (layout claims %d)"
          % (len(measured), len(HOLES_CENTERED_M)))

    layout = np.asarray(HOLES_CENTERED_M, dtype=np.float64)
    mismatch = np.linalg.norm(
        measured[:, None, :] - layout[None, :, :], axis=2
    ).min(axis=1)
    print("vs the layout file    : median %.1f mm  max %.1f mm  (<3 mm: %d/%d)"
          % (np.median(mismatch) * 1000, mismatch.max() * 1000,
             int((mismatch < 0.003).sum()), len(measured)))

    lower_left = measured + np.asarray([BOARD_WIDTH_M / 2.0, BOARD_HEIGHT_M / 2.0])
    order = np.lexsort((lower_left[:, 0], lower_left[:, 1]))
    lower_left = lower_left[order]
    print("\nmeasured HOLES_LOWER_LEFT_M (metres, lower-left origin):")
    print("HOLES_LOWER_LEFT_M = [")
    for x, y in lower_left:
        print("    [%.4f, %.4f]," % (x, y))
    print("]")
    if args.output:
        np.savetxt(args.output, lower_left, delimiter=",", fmt="%.6f")
        print(f"\nwrote {args.output}")
    print(
        "\nReview these against the physical board before replacing "
        "maze_layout.HOLES_LOWER_LEFT_M -- they are measured, not CAD."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
