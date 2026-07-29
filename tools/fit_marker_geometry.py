#!/usr/bin/env python3
"""Jointly fit the moving-dot spacing and the dot-plane height from tilted views.

The problem
-----------
Marble position comes from a homography off the four moving dots, so its whole
scale is set by MOVING_MARKER_SPACING_X_M / Y_M (0.269 / 0.237). Checking the 21
detected hole centres against their known DXF positions leaves a 2-2.5% scale
residual -- about 3 mm at the board edge, near zero at the centre, which is
exactly the signature of your reach overshoot.

That residual has two candidate causes and a single near-level frame cannot
separate them:

  * the assumed dot spacing is wrong, or
  * marker_plane_height_m is wrong -- the holes sit on the floor, the dots some
    height h above it, so mapping floor features through a dot-plane homography
    needs a parallax correction of (1 + h / camera_height).

With the camera nearly overhead (measured offset 8 x 13 mm at ~290 mm) both act
as an almost pure radial scale about the board centre. Algebraically degenerate.

Why tilt breaks it
------------------
The parallax correction is centred on the CAMERA's position in the board frame,
and that point sweeps across the board as the plate tilts. A spacing error, by
contrast, always scales about the board origin. So across a range of tilts the
two produce different residual patterns, and h becomes identifiable.

The fit
-------
Five parameters: spacing sx, sy, dot-plane height h, and a 2D offset (ox, oy)
between the dot-quad centroid and the DXF board origin. The offset must be free
or it would be absorbed into the other three. Ground truth is the DXF hole
positions, which tools/measure_hole_layout.py confirmed the board matches.

    python3 tools/record_frames.py --seconds 90 --out /tmp/calib_frames
    python3 tools/fit_marker_geometry.py --frames_dir /tmp/calib_frames
"""

import argparse
import glob
import os
import sys

import cv2
import numpy as np
from scipy.optimize import least_squares

from cyberrunner_state_estimation.core.ai_map_state import (
    MarkerQuadGuard,
    camera_center_in_board,
)
from cyberrunner_state_estimation.core.detection import Detector
from cyberrunner_state_estimation.core.hole_mask import HOLES_CENTERED_M, HOLE_RADII_M
from cyberrunner_state_estimation.core.plate_pose import PlatePoseEstimator


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PACKAGE_DIR = os.path.join(REPO_ROOT, "cyberrunner_state_estimation")
NOMINAL_X, NOMINAL_Y = 0.269, 0.237


def model_quad(sx, sy):
    return np.asarray(
        [[-sx / 2, -sy / 2], [sx / 2, -sy / 2], [sx / 2, sy / 2], [-sx / 2, sy / 2]],
        dtype=np.float32,
    )


def collect(paths, markers, min_holes, radius_m):
    """Per-frame dot quad and hole blob pixels. Done once; the fit reuses it."""
    tracker = Detector(markers[4:], ai_mode="off", corner_subimage_half_size=12)
    guard = MarkerQuadGuard(
        np.asarray(markers[4:], dtype=np.float64)[:, ::-1], mode="moving"
    )
    frames = []
    for index, path in enumerate(paths):
        frame = cv2.imread(path)
        if frame is None:
            continue
        raw = tracker.detect_corners(frame)
        corners_rc, valid, _ = guard.update(
            raw, tracker.corner_found, index / 50.0
        )
        tracker.corners = corners_rc.astype(np.float32)
        tracker.corners_missing = False
        if not valid:
            continue
        quad = np.asarray(corners_rc, dtype=np.float32)[:, ::-1]
        # Nominal scale only, to size the blob filter. The fit does not use it.
        px_per_m = float(np.linalg.norm(quad[1] - quad[0])) / NOMINAL_X
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        dark = (gray < 70).astype(np.uint8)
        count, _, stats, centroids = cv2.connectedComponentsWithStats(dark, 8)
        expected = np.pi * (radius_m * px_per_m) ** 2
        blobs = [
            centroids[i]
            for i in range(1, count)
            if 0.45 * expected < stats[i, cv2.CC_STAT_AREA] < 2.2 * expected
            and 0.65
            < stats[i, cv2.CC_STAT_WIDTH] / max(stats[i, cv2.CC_STAT_HEIGHT], 1)
            < 1.55
        ]
        if len(blobs) >= min_holes:
            frames.append((quad, np.asarray(blobs, dtype=np.float32)))
    return frames


def spread_subset(frames, count):
    """Farthest-point sample on quad geometry, to maximise tilt diversity."""
    if len(frames) <= count:
        return frames
    feats = np.asarray([f[0].reshape(-1) for f in frames])
    chosen = [0]
    while len(chosen) < count:
        distance = np.min(
            np.linalg.norm(feats[:, None, :] - feats[None, chosen, :], axis=2), axis=1
        )
        distance[chosen] = -1.0
        chosen.append(int(distance.argmax()))
    return [frames[i] for i in chosen]


def residuals(params, frames, pose, layout, trim_m):
    sx, sy, h, ox, oy = params
    model = model_quad(sx, sy)
    model3 = np.column_stack((model, np.zeros(4))).astype(np.float32)
    offset = np.asarray([ox, oy])
    out = []
    for quad, blobs in frames:
        transform = cv2.getPerspectiveTransform(quad, model)
        try:
            T, _, _ = pose.get_pose_T__C_P(model3, pose.undistort_points(quad[:, ::-1]))
        except cv2.error:
            continue
        centre = camera_center_in_board(T)
        camera_xy, camera_h = centre[:2], abs(float(centre[2]))
        if not np.isfinite(camera_h) or camera_h < 0.05:
            continue
        planar = cv2.perspectiveTransform(
            blobs.reshape(-1, 1, 2), transform
        ).reshape(-1, 2)
        # Hole floor sits h BELOW the dot plane, so the ray continues outward.
        scale = 1.0 + h / camera_h
        corrected = camera_xy + (planar - camera_xy) * scale + offset
        distance = np.linalg.norm(
            corrected[:, None, :] - layout[None, :, :], axis=2
        ).min(axis=1)
        # Clip rather than drop: least_squares needs a fixed-length residual, and
        # a clipped outlier contributes a constant with zero gradient, which is
        # the trimming behaviour without changing the vector size.
        out.append(np.minimum(distance, trim_m))
    if not out:
        return np.full(1, 1.0e3)
    return np.concatenate(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames_dir", required=True)
    parser.add_argument("--markers", default=os.path.join(PACKAGE_DIR, "markers.csv"))
    parser.add_argument("--min_holes", type=int, default=16)
    parser.add_argument("--use_frames", type=int, default=40)
    parser.add_argument("--trim_mm", type=float, default=15.0)
    parser.add_argument("--radius_m", type=float, default=float(HOLE_RADII_M[0]))
    args = parser.parse_args()

    paths = sorted(
        glob.glob(os.path.join(args.frames_dir, "*.png"))
        + glob.glob(os.path.join(args.frames_dir, "*.jpg"))
    )
    if not paths:
        print(f"no frames in {args.frames_dir}", file=sys.stderr)
        return 1
    markers = np.loadtxt(args.markers, delimiter=",")
    frames = collect(paths, markers, args.min_holes, args.radius_m)
    print("frames read %d, usable %d" % (len(paths), len(frames)))
    if len(frames) < 8:
        print("need at least 8 usable frames", file=sys.stderr)
        return 1
    frames = spread_subset(frames, args.use_frames)

    pose = PlatePoseEstimator()
    print("intrinsics: fx %.2f  k1 %s"
          % (pose.K[0, 0], "none" if pose.dist is None else round(pose.dist.ravel()[0], 4)))
    layout = np.asarray(HOLES_CENTERED_M, dtype=np.float64)
    trim = args.trim_mm / 1000.0

    # How far the camera's board-frame position moves across these views. This
    # IS the leverage that separates h from the spacing -- report it, because a
    # small spread means the answer is not identifiable no matter what comes out.
    sweep = []
    model3 = np.column_stack((model_quad(NOMINAL_X, NOMINAL_Y), np.zeros(4))).astype(np.float32)
    for quad, _ in frames:
        T, _, _ = pose.get_pose_T__C_P(model3, pose.undistort_points(quad[:, ::-1]))
        sweep.append(camera_center_in_board(T)[:2])
    sweep = np.asarray(sweep)
    span = (sweep.max(axis=0) - sweep.min(axis=0)) * 1000
    print("views used %d, camera board-frame sweep %.1f x %.1f mm"
          % (len(frames), span[0], span[1]))

    guess = np.asarray([NOMINAL_X, NOMINAL_Y, 0.005, 0.0, 0.0])
    result = least_squares(
        residuals, guess, args=(frames, pose, layout, trim),
        bounds=([0.24, 0.21, -0.005, -0.02, -0.02], [0.30, 0.26, 0.030, 0.02, 0.02]),
        x_scale=[0.01, 0.01, 0.005, 0.005, 0.005], diff_step=0.02,
        loss="soft_l1", f_scale=0.003,
    )
    sx, sy, h, ox, oy = result.x
    before = residuals(guess, frames, pose, layout, trim)
    after = residuals(result.x, frames, pose, layout, trim)
    print()
    print("=== fitted marker geometry ===")
    print("dot spacing x : %.1f mm  (nominal %.1f, %+.2f%%)"
          % (sx * 1000, NOMINAL_X * 1000, (sx / NOMINAL_X - 1) * 100))
    print("dot spacing y : %.1f mm  (nominal %.1f, %+.2f%%)"
          % (sy * 1000, NOMINAL_Y * 1000, (sy / NOMINAL_Y - 1) * 100))
    print("dot plane h   : %.1f mm above the hole floor" % (h * 1000))
    print("origin offset : %+.1f, %+.1f mm" % (ox * 1000, oy * 1000))
    print()
    keep = after < trim * 0.99
    print("hole residual : %.2f mm median -> %.2f mm  (over %d inlier points)"
          % (np.median(before[before < trim * 0.99]) * 1000,
             np.median(after[keep]) * 1000, int(keep.sum())))
    print("clipped as spurious: %d of %d" % (int((~keep).sum()), len(after)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
