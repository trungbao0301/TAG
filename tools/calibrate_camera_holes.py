#!/usr/bin/env python3
"""Calibrate the camera intrinsics using the maze holes as a planar target.

Why this exists
---------------
calib_results_cyberrunner.txt is an ocam model for a 1920x1200 capture, but the
pipeline runs 1280x720 downscaled to 640x360 and padded to 640x400. The
polynomial does not transfer: measured against markers.csv it changes the dot
spacing by 1.93x while leaving the aspect ratios identical to four decimals, so
it applies no shape correction at all, only a wrong magnification. Because
PlatePoseEstimator.get_pose_T__C_P solvePnPs with the same self.f that
K_ocam reprojected with, self.f cancels and the effective focal length becomes
the one the polynomial implies on-axis (673.2 px after scale(3)) instead of the
true 298.1 px. That put the camera at 0.556 m instead of a ruler-measured
0.290 m and scaled every angle derived from it.

No rescale of that file can fix this. OcamModel.scale is angle-preserving by
construction, and multiplying its coefficients scales every point's tan(theta)
by one factor, which is algebraically identical to changing f in a pinhole and
adds no distortion correction. The only fix is new calibration data.

The target
----------
The 21 maze holes are coplanar, at known metric positions
(maze_layout.HOLES_LOWER_LEFT_M) and all radius 7.5 mm. That is a valid planar
calibration target -- the same thing a checkerboard provides. The four blue
moving dots give a per-frame homography that predicts where each hole should
be, so the correspondence problem is solved before any blob detection runs.
Tilting the plate supplies the pose variety Zhang's method needs.

Note the holes lie on the board floor while the dots sit slightly above it. The
dots are used ONLY to predict search windows; the calibration points are the
refined hole centres, which are genuinely coplanar. So the small dot-to-floor
offset cannot bias the result.

Usage
-----
    ros2 run cyberrunner_state_estimation ai_map_estimator   # optional
    python3 tools/calibrate_camera_holes.py

Tilt the plate slowly through its full range while it captures. It accepts a
frame only when the board pose differs enough from every frame already kept, so
redundant near-identical views cannot dominate the fit. Press ``q`` to stop
early, ``s`` to force-accept the current frame.
"""

import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image

from cyberrunner_state_estimation.core.ai_map_state import MarkerQuadGuard
from cyberrunner_state_estimation.core.detection import Detector
from cyberrunner_state_estimation.core.hole_mask import (
    HOLES_CENTERED_M,
    HOLE_RADII_M,
    MARKERS_CENTERED_M,
)


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PACKAGE_DIR = os.path.join(REPO_ROOT, "cyberrunner_state_estimation")


def warn_if_shadowed(logger):
    """Warn when another workspace shadows this repo on AMENT_PREFIX_PATH.

    ~/.bashrc sources ~/cyberrunner_ws, and install/setup.bash APPENDS, so this
    repo lands behind it and get_package_share_directory resolves elsewhere.
    That is what silently sent a marker re-selection into the wrong workspace.
    This tool reads and writes repo-relative paths so it cannot repeat that, but
    the ESTIMATOR still resolves via ament, so the mismatch is worth saying.
    """
    try:
        resolved = get_package_share_directory("cyberrunner_state_estimation")
    except Exception:  # noqa: BLE001 - absent package is not this tool's problem
        return
    # Compare the resolved markers.csv, not the directory: a correctly built
    # workspace has install/.../share as a real directory holding symlinks back
    # to these sources, so the DIRECTORY paths always differ even when they
    # serve identical files. Only a genuine cross-workspace mismatch matters.

    def target(directory):
        return os.path.realpath(os.path.join(directory, "markers.csv"))

    if target(resolved) != target(PACKAGE_DIR):
        logger.warn(
            "ament resolves cyberrunner_state_estimation to %s, not %s. This "
            "tool uses the repo paths regardless, but the estimator will read "
            "the other workspace -- source this repo's install last, or run "
            "colcon build here, before trusting the estimator." % (resolved, PACKAGE_DIR)
        )


def message_time(message):
    stamp = message.header.stamp
    return float(stamp.sec) + float(stamp.nanosec) * 1.0e-9


def refine_hole(frame_gray, predicted_xy, radius_px, pad_px=6.0):
    """Locate one hole centre near ``predicted_xy`` to sub-pixel accuracy.

    Returns ``(centre_xy, reason)`` with ``centre_xy`` None when the window
    holds nothing that looks like a hole. Rejecting is always preferable to
    contributing a wrong correspondence to the fit.
    """
    height, width = frame_gray.shape[:2]
    half = float(radius_px) + float(pad_px)
    x0, y0 = int(np.floor(predicted_xy[0] - half)), int(np.floor(predicted_xy[1] - half))
    x1, y1 = int(np.ceil(predicted_xy[0] + half)), int(np.ceil(predicted_xy[1] + half))
    if x0 < 0 or y0 < 0 or x1 > width or y1 > height:
        return None, "outside_frame"
    window = frame_gray[y0:y1, x0:x1].astype(np.float32)
    if window.size == 0:
        return None, "empty_window"

    low, high = float(window.min()), float(window.max())
    if high - low < 25.0:
        # Flat window: no hole/board contrast, so any centroid would be noise.
        return None, "no_contrast"
    dark = (window < low + 0.5 * (high - low)).astype(np.uint8)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(dark, 8)
    if count < 2:
        return None, "no_dark_blob"

    expected_area = np.pi * float(radius_px) ** 2
    centre = np.asarray([window.shape[1] / 2.0, window.shape[0] / 2.0])
    best, best_distance = None, None
    for index in range(1, count):
        area = float(stats[index, cv2.CC_STAT_AREA])
        if not 0.35 * expected_area <= area <= 2.5 * expected_area:
            continue
        distance = float(np.linalg.norm(np.asarray(centroids[index]) - centre))
        if best_distance is None or distance < best_distance:
            best, best_distance = index, distance
    if best is None:
        return None, "no_blob_of_hole_size"
    if best_distance > radius_px:
        return None, "blob_too_far"

    # Intensity-weighted centroid of the blob: darker pixels sit nearer the
    # hole centre, which recovers sub-pixel accuracy a binary centroid loses.
    mask = labels == best
    weight = np.clip(high - window, 0.0, None) * mask
    total = float(weight.sum())
    if total <= 0.0:
        return None, "zero_weight"
    rows, cols = np.nonzero(mask)
    cx = float((weight[rows, cols] * cols).sum() / total)
    cy = float((weight[rows, cols] * rows).sum() / total)
    return np.asarray([x0 + cx, y0 + cy], dtype=np.float64), "ok"


class DotTracker:
    """Moving-dot tracking with the same guard the estimator uses.

    A bare Detector is NOT safe here. With its default corner_subimage_half_size
    of 25 the first-frame search window is +-25 px, and on this rig that made
    corner 2 lock onto a FIXED frame marker 26 px away: the dot x span went from
    277 to 301 px (the fixed dots span 319), which silently corrupted every board
    coordinate derived from the homography. A tighter window plus
    MarkerQuadGuard, which rejects exactly this substitution, restores 277.7 px.
    """

    def __init__(self, markers, fps=50.0):
        self.tracker = Detector(
            markers[4:], ai_mode="off", corner_subimage_half_size=12
        )
        self.guard = MarkerQuadGuard(
            np.asarray(markers[4:], dtype=np.float64)[:, ::-1], mode="moving"
        )
        self.timestep = 1.0 / max(1.0, float(fps))
        self.index = 0

    def corners(self, frame):
        raw = self.tracker.detect_corners(frame)
        accepted, valid, status = self.guard.update(
            raw, self.tracker.corner_found, self.index * self.timestep
        )
        self.index += 1
        # Keep the next local search attached to the accepted quad, never to a
        # rejected blob -- the same feedback the estimator applies.
        self.tracker.corners = accepted.astype(np.float32)
        self.tracker.corners_missing = False
        return (accepted if valid else None), status


def find_holes(dots, frame):
    """Refine every visible hole centre in one frame.

    Returns (object_points, image_points, corners_rc, reasons). Shared by the
    live node and the offline path so both apply identical acceptance rules.
    """
    reasons = {}
    corners_rc, status = dots.corners(frame)
    if corners_rc is None:
        return None, None, None, {status: 1}
    corners_xy = np.asarray(corners_rc, dtype=np.float32)[:, ::-1]
    transform = cv2.getPerspectiveTransform(
        MARKERS_CENTERED_M.astype(np.float32), corners_xy
    )
    holes_xy = cv2.perspectiveTransform(
        HOLES_CENTERED_M.reshape(-1, 1, 2).astype(np.float32), transform
    ).reshape(-1, 2)
    px_per_m = float(np.linalg.norm(corners_xy[1] - corners_xy[0])) / float(
        MARKERS_CENTERED_M[1, 0] - MARKERS_CENTERED_M[0, 0]
    )
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    object_view, image_view = [], []
    for index, predicted in enumerate(holes_xy):
        centre, reason = refine_hole(
            gray, predicted, float(HOLE_RADII_M[index]) * px_per_m
        )
        if centre is None:
            reasons[reason] = reasons.get(reason, 0) + 1
            continue
        object_view.append(
            [HOLES_CENTERED_M[index, 0], HOLES_CENTERED_M[index, 1], 0.0]
        )
        image_view.append(centre)
    return object_view, image_view, corners_rc, reasons


def select_spread(candidates, target, min_delta_px):
    """Farthest-point sample the poses, rather than taking them in order.

    Offline we can see every candidate at once, so pick the most mutually
    distinct views instead of greedily accepting whatever arrives first.
    """
    if not candidates:
        return []
    chosen = [0]
    while len(chosen) < target:
        best, best_distance = None, -1.0
        for index in range(len(candidates)):
            if index in chosen:
                continue
            distance = min(
                float(
                    np.mean(
                        np.linalg.norm(
                            candidates[index][2] - candidates[picked][2], axis=1
                        )
                    )
                )
                for picked in chosen
            )
            if distance > best_distance:
                best, best_distance = index, distance
        if best is None or best_distance < min_delta_px:
            break
        chosen.append(best)
    return [candidates[i] for i in chosen]


def run_offline(args):
    """Calibrate from recorded frames -- no ROS, no display, full speed."""
    paths = sorted(
        glob.glob(os.path.join(args.frames_dir, "*.png"))
        + glob.glob(os.path.join(args.frames_dir, "*.jpg"))
    )
    if not paths:
        print(f"no frames in {args.frames_dir}", file=sys.stderr)
        return 1
    markers = np.loadtxt(args.markers, delimiter=",")
    dots = DotTracker(markers)

    candidates, reasons, counts = [], {}, []
    for path in paths:
        frame = cv2.imread(path)
        if frame is None:
            continue
        object_view, image_view, corners_rc, why = find_holes(dots, frame)
        for key, value in why.items():
            reasons[key] = reasons.get(key, 0) + value
        if image_view is None:
            continue
        counts.append(len(image_view))
        if len(image_view) >= args.min_holes:
            candidates.append(
                (
                    np.asarray(object_view, dtype=np.float32),
                    np.asarray(image_view, dtype=np.float32),
                    np.asarray(corners_rc, dtype=np.float64),
                )
            )
    print("frames read           : %d" % len(paths))
    if counts:
        print(
            "holes found per frame : mean %.1f  min %d  max %d  (of %d)"
            % (np.mean(counts), min(counts), max(counts), len(HOLES_CENTERED_M))
        )
    print("frames with >= %d     : %d" % (args.min_holes, len(candidates)))
    if reasons:
        print("rejection reasons     :")
        for key, value in sorted(reasons.items(), key=lambda kv: -kv[1]):
            print("    %-24s %d" % (key, value))

    views = select_spread(candidates, args.target_frames, args.min_pose_delta_px)
    print("views selected        : %d" % len(views))
    if len(views) < 4:
        print(
            "\nToo few usable views to calibrate (need 4+, want 12+). "
            "Nothing written.",
            file=sys.stderr,
        )
        return 1
    frame_shape = cv2.imread(paths[0]).shape[:2]
    result = solve([v[0] for v in views], [v[1] for v in views], frame_shape, args)
    report(result, args)
    with open(args.output, "w") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    print(f"\nwrote {args.output}")
    return 0


class HoleCalibrator(Node):
    def __init__(self, args):
        super().__init__("cyberrunner_hole_calibrator")
        self.args = args
        self.bridge = CvBridge()

        markers = np.loadtxt(args.markers, delimiter=",")
        if markers.shape != (8, 2):
            raise ValueError(f"{args.markers} must hold 8 [x,y] markers")
        # Detector takes (x, y) and returns (row, col). Wrapped in DotTracker
        # so the guard rejects fixed-marker substitution -- see DotTracker.
        self.dots = DotTracker(markers)
        self.get_logger().info(f"moving-dot seeds from {args.markers}")
        warn_if_shadowed(self.get_logger())

        self.object_points = []
        self.image_points = []
        self.accepted_corners = []
        self.frame_shape = None
        self.rejections = {}
        self.last_capture_sec = -1.0e9

        # RELIABLE, matching the publisher: see the note in record_frames.py.
        # BEST_EFFORT drops over half of these 768 KB frames on this rig.
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.create_subscription(Image, args.camera_topic, self.on_image, qos)
        self.get_logger().info(
            f"tilt the plate slowly; need {args.target_frames} varied views "
            f"with >= {args.min_holes} of {len(HOLES_CENTERED_M)} holes each"
        )

    def _pose_is_new(self, corners_rc):
        """Require every kept view to be geometrically distinct."""
        for previous in self.accepted_corners:
            shift = float(np.mean(np.linalg.norm(corners_rc - previous, axis=1)))
            if shift < self.args.min_pose_delta_px:
                return False
        return True

    def _note(self, reason):
        self.rejections[reason] = self.rejections.get(reason, 0) + 1

    def on_image(self, message):
        frame = self.bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
        self.frame_shape = frame.shape[:2]
        display = frame.copy()

        corners_rc, dot_status = self.dots.corners(frame)
        status = "dots %s" % dot_status

        if corners_rc is not None:
            corners_xy = np.asarray(corners_rc, dtype=np.float32)[:, ::-1]
            transform = cv2.getPerspectiveTransform(
                MARKERS_CENTERED_M.astype(np.float32), corners_xy
            )
            holes_xy = cv2.perspectiveTransform(
                HOLES_CENTERED_M.reshape(-1, 1, 2).astype(np.float32), transform
            ).reshape(-1, 2)
            # Metres-per-pixel from the dot quad, so the search window tracks
            # the board as it tilts instead of assuming a fixed scale.
            span_px = float(np.linalg.norm(corners_xy[1] - corners_xy[0]))
            px_per_m = span_px / float(
                MARKERS_CENTERED_M[1, 0] - MARKERS_CENTERED_M[0, 0]
            )

            # Gate BEFORE refining. Dot detection is four small windows, but
            # refinement is 21 connected-component passes -- running it on every
            # frame is what made the window lag, and on frames we would discard
            # anyway it buys nothing.
            forced = self.args._force
            self.args._force = False
            elapsed = message_time(message) - self.last_capture_sec
            candidate = forced or (
                elapsed >= self.args.min_interval_sec
                and self._pose_is_new(corners_rc)
            )

            if not candidate:
                for predicted in holes_xy:
                    cv2.drawMarker(
                        display, tuple(np.round(predicted).astype(int)),
                        (0, 200, 200), cv2.MARKER_CROSS, 4, 1
                    )
                status += "  waiting for new pose"
            else:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                object_view, image_view = [], []
                for index, predicted in enumerate(holes_xy):
                    radius_px = float(HOLE_RADII_M[index]) * px_per_m
                    centre, reason = refine_hole(gray, predicted, radius_px)
                    if centre is None:
                        self._note(reason)
                        cv2.drawMarker(
                            display, tuple(np.round(predicted).astype(int)),
                            (0, 0, 255), cv2.MARKER_TILTED_CROSS, 6, 1
                        )
                        continue
                    object_view.append(
                        [HOLES_CENTERED_M[index, 0], HOLES_CENTERED_M[index, 1], 0.0]
                    )
                    image_view.append(centre)
                    cv2.circle(
                        display, tuple(np.round(centre).astype(int)),
                        int(round(radius_px)), (0, 255, 0), 1
                    )
                status += "  holes %d/%d" % (len(image_view), len(HOLES_CENTERED_M))
                if len(image_view) >= self.args.min_holes:
                    self.object_points.append(
                        np.asarray(object_view, dtype=np.float32)
                    )
                    self.image_points.append(np.asarray(image_view, dtype=np.float32))
                    self.accepted_corners.append(
                        np.asarray(corners_rc, dtype=np.float64)
                    )
                    self.last_capture_sec = message_time(message)
                    self.get_logger().info(
                        "kept view %d/%d (%d holes)"
                        % (
                            len(self.object_points),
                            self.args.target_frames,
                            len(image_view),
                        )
                    )
        else:
            self._note("moving_dots_not_found")

        status += "  kept %d/%d" % (len(self.object_points), self.args.target_frames)
        cv2.putText(
            display, status, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            (0, 255, 255), 1, cv2.LINE_AA
        )
        cv2.putText(
            display, "tilt the plate | s = force keep | q = done", (8, 392),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1, cv2.LINE_AA
        )
        cv2.imshow("hole calibration", display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("s"):
            self.args._force = True
        if key in (ord("q"), 27) or len(self.object_points) >= self.args.target_frames:
            rclpy.shutdown()


def solve(object_points, image_points, frame_shape, args):
    """Run Zhang's method and report enough to judge whether to trust it."""
    height, width = frame_shape
    # Principal point: the calibration centre 598.84, 958.88 scaled by 3 gives
    # row 199.6 / col 319.6, and the real capture chain agrees exactly -- the
    # 16:10 -> 16:9 crop drops 60 rows per side, which is 20 rows after /3,
    # precisely the border fast_camera_publisher_v2.py adds back. So it is
    # known to 0.4 px and holding it fixed keeps the fit well conditioned when
    # the plate can only tilt a modest amount.
    guess = np.array(
        [[300.0, 0.0, width / 2.0], [0.0, 300.0, height / 2.0], [0.0, 0.0, 1.0]]
    )
    flags = cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_ZERO_TANGENT_DIST
    flags |= cv2.CALIB_FIX_ASPECT_RATIO  # square pixels
    if args.fix_principal_point:
        flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
    for name, coefficient in (("K3", cv2.CALIB_FIX_K3), ("K2", cv2.CALIB_FIX_K2)):
        limit = {"K3": 3, "K2": 2}[name]
        if args.num_dist_coeffs < limit:
            flags |= coefficient

    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        object_points, image_points, (width, height), guess, None, flags=flags
    )
    heights = [abs(float(t[2])) * 1000.0 for t in tvecs]
    return {
        "rms_reprojection_px": float(rms),
        "fx": float(K[0, 0]),
        "fy": float(K[1, 1]),
        "cx": float(K[0, 2]),
        "cy": float(K[1, 2]),
        "dist": [float(v) for v in np.asarray(dist).reshape(-1)],
        "image_width": int(width),
        "image_height": int(height),
        "num_views": len(object_points),
        "num_points": int(sum(len(p) for p in image_points)),
        "camera_height_mm_median": float(np.median(heights)),
        "camera_height_mm_min": float(np.min(heights)),
        "camera_height_mm_max": float(np.max(heights)),
    }


def report(result, args):
    result["ruler_height_mm"] = args.ruler_height_m * 1000.0
    print("\n=== calibration ===")
    print("views %d, points %d, RMS reprojection %.3f px"
          % (result["num_views"], result["num_points"],
             result["rms_reprojection_px"]))
    print("fx %.2f  fy %.2f  cx %.2f  cy %.2f"
          % (result["fx"], result["fy"], result["cx"], result["cy"]))
    print("dist %s" % np.round(result["dist"], 6).tolist())
    print("camera height: median %.1f mm (range %.1f - %.1f)"
          % (result["camera_height_mm_median"], result["camera_height_mm_min"],
             result["camera_height_mm_max"]))
    print("ruler:         %.1f mm" % result["ruler_height_mm"])
    error = abs(result["camera_height_mm_median"] - result["ruler_height_mm"])
    print("agreement:     %.1f mm (%.1f%%)"
          % (error, 100.0 * error / result["ruler_height_mm"]))
    if result["rms_reprojection_px"] > 1.0:
        print("\nWARNING: RMS above 1 px. Check the overlay -- hole centres are "
              "probably being mis-detected. Do not install this.")
    if error > 0.05 * result["ruler_height_mm"]:
        print("\nWARNING: more than 5% from the ruler. Re-measure the lens-to-dot "
              "distance before trusting either number.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera_topic", default="/cyberrunner_camera/image")
    parser.add_argument(
        "--frames_dir", default="",
        help="Calibrate offline from recorded frames (see tools/record_frames.py) "
             "instead of subscribing live. No ROS, no display."
    )
    parser.add_argument("--target_frames", type=int, default=18)
    parser.add_argument("--min_holes", type=int, default=14)
    # Zhang's method needs genuinely different plate orientations. At 1.5 px a
    # first run captured views 33 ms apart -- near-duplicates that add no
    # conditioning. These two gates together force real spread.
    parser.add_argument(
        "--min_pose_delta_px", type=float, default=6.0,
        help="Minimum mean dot movement before a view counts as a new pose."
    )
    parser.add_argument(
        "--min_interval_sec", type=float, default=0.7,
        help="Minimum time between kept views, so one tilt cannot burst-fill."
    )
    parser.add_argument(
        "--num_dist_coeffs", type=int, default=1, choices=(0, 1, 2, 3),
        help="Radial terms to fit. 1 (k1) is the safest with limited tilt."
    )
    parser.add_argument("--fix_principal_point", action="store_true", default=True)
    parser.add_argument(
        "--free_principal_point", dest="fix_principal_point", action="store_false"
    )
    parser.add_argument(
        "--ruler_height_m", type=float, default=0.290,
        help="Measured lens-to-dot-plane distance, for the sanity check only."
    )
    # Repo-relative by default: resolving these through ament would pick up
    # whichever workspace happens to win AMENT_PREFIX_PATH. See warn_if_shadowed.
    parser.add_argument(
        "--markers", default=os.path.join(PACKAGE_DIR, "markers.csv")
    )
    parser.add_argument(
        "--output", default=os.path.join(PACKAGE_DIR, "pinhole_calib.json")
    )
    args, ros_args = parser.parse_known_args()
    args._force = False

    if args.frames_dir:
        return run_offline(args)

    rclpy.init(args=ros_args)
    node = HoleCalibrator(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    cv2.destroyAllWindows()

    if node.rejections:
        print("\nhole rejections by reason:")
        for reason, count in sorted(
            node.rejections.items(), key=lambda kv: -kv[1]
        ):
            print("  %-24s %d" % (reason, count))

    if len(node.object_points) < 4:
        print(
            f"\nOnly {len(node.object_points)} usable views -- need at least 4, "
            "and 12+ for a trustworthy fit. Nothing written.",
            file=sys.stderr,
        )
        if rclpy.ok():
            rclpy.shutdown()
        return 1

    result = solve(node.object_points, node.image_points, node.frame_shape, args)
    report(result, args)

    with open(args.output, "w") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    print(f"\nwrote {args.output}")
    print(
        "Now install it so PlatePoseEstimator sees it:\n"
        "  colcon build --symlink-install --packages-select "
        "cyberrunner_state_estimation"
    )

    if rclpy.ok():
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
