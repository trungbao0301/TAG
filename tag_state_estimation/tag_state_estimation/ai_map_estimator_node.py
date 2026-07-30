#!/usr/bin/env python3
"""AI-only TAG state estimator with map-native board registration."""

import math
import os
import time

import cv2
import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32, String

from tag_interfaces.msg import StateEstimate, StateEstimateSub
from tag_state_estimation.ai_marble_common import OnnxMarbleDetector
from tag_state_estimation.core.ai_map_state import (
    AlphaBetaKinematics,
    MarkerQuadGuard,
    find_marker_quad_global,
    MOVING_MARKERS_CENTERED_M,
    map_ai_pixel,
    marker_quad_valid,
)
from tag_state_estimation.core.detection import Detector
from tag_state_estimation.core.hole_mask import (
    TimedHoleRejector,
    candidate_hole_index,
)
from tag_state_estimation.core.plate_pose import PlatePoseEstimator


MOVING_MODEL_POINTS = np.column_stack(
    (MOVING_MARKERS_CENTERED_M, np.zeros(4, dtype=np.float32))
).astype(np.float32)
FIXED_HSV = ((43, 140), (125, 255), (40, 255))


def ros_time_seconds(stamp):
    return float(stamp.sec) + float(stamp.nanosec) * 1.0e-9


def default_model_path():
    """Locate models/marble_detector.onnx without assuming the install layout.

    The previous form used os.path.dirname(__file__), which under
    colcon --symlink-install is build/<pkg>/<pkg>/, so "../.." resolved to
    build/ and the default pointed at build/models/marble_detector.onnx --
    a path that never exists. Every launch therefore had to pass ai_model_path
    explicitly or fail with FileNotFoundError.
    """
    here = os.path.dirname(os.path.realpath(__file__))  # follows the symlink
    candidates = [
        os.path.abspath(os.path.join(here, "..", "..", "models")),
        os.path.abspath(os.path.join(here, "..", "..", "..", "models")),
        os.path.join(os.getcwd(), "models"),
    ]
    for directory in candidates:
        path = os.path.join(directory, "marble_detector.onnx")
        if os.path.isfile(path):
            return path
    return os.path.join(candidates[0], "marble_detector.onnx")


def crop_ball_image(frame, xy, size=64):
    result = np.zeros((size, size, 3), dtype=np.uint8)
    if xy is None or not np.all(np.isfinite(xy)):
        return result
    x, y = map(int, np.round(xy))
    half = size // 2
    x0, x1 = max(0, x - half), min(frame.shape[1], x + half)
    y0, y1 = max(0, y - half), min(frame.shape[0], y + half)
    if x0 >= x1 or y0 >= y1:
        return result
    dst_x0, dst_y0 = x0 - (x - half), y0 - (y - half)
    result[dst_y0 : dst_y0 + (y1 - y0), dst_x0 : dst_x0 + (x1 - x0)] = (
        frame[y0:y1, x0:x1]
    )
    return result


class AiMapEstimatorNode(Node):
    def __init__(self):
        super().__init__("tag_ai_map_estimator")
        self.declare_parameter("ai_model_path", default_model_path())
        self.declare_parameter("ai_confidence_threshold", 0.90)
        # Fallback only: map_ai_pixel prefers the height recovered by PnP from
        # the moving dots, and takes it whenever |z| > 0.05 m, which is always
        # true in practice. 0.29 m is the ruler distance from the dot plane to
        # the lens (0.31 m to the board floor); the old 0.20 m was a guess.
        #
        # NOTE: PnP recovers 0.556 m instead, a factor 1.9 too far. That is the
        # same 1.93x by which PlatePoseEstimator.undistort_points shrinks the
        # dot spacing (276.5 px raw -> 143.2 px undistorted) before solvePnP
        # sees it, so the ocam polynomial / o.scale(3) does not match the
        # camera's current 1280x720 capture mode. Ball XY is largely immune --
        # map_ai_pixel homographies the RAW dot pixels -- but alpha/beta come
        # from that PnP and may carry a scale-induced bias.
        self.declare_parameter("camera_height_m", 0.29)
        # Height of the moving-dot plane above the play surface. 10 mm was
        # MEASURED on this board: the dots sit on the moving rim, which stands
        # above the maze floor. Keep that number, because it is the measurement
        # and this is not.
        #
        # Set to 0 to take the outward push out of the loop. At 10 mm the marble
        # centre (6 mm up) is 4 mm BELOW the dot plane, so
        #   parallax_scale = 1 - (0.006 - 0.010)/0.29 = 1.0138
        # and reported positions are pushed away from the camera axis. At 0 it is
        # 1 - 0.006/0.29 = 0.9793, pulled inward -- a 3.5% swing in absolute
        # scale, the largest of any single parameter here. That matters because
        # reported y currently lands outside the board's 229 mm edge on 43% of
        # frames and on 47.6% of episode-start frames, which is what makes
        # episodes begin off-path and die in a few steps.
        #
        # It looked like it could not be the cause -- the push is 1.4% and
        # isotropic, while the problem was y-specific -- but the measurement says
        # otherwise. Over a training run at 0, against the same measurement at
        # 0.010:
        #
        #     y reported off the board     43.0%  ->  0.00%
        #     frames inside the corridor   79.9%  ->  93.6%
        #     episode starts inside it     52.0%  ->  92.9%
        #     episodes ended by OFFPATH    49/57  ->  0
        #     median episode length       10 steps -> 68 steps
        #
        # The 1.4% was enough because the marble spends its time on the outer
        # corridors, a few mm from the rim, where 1.4% of 115 mm is the whole
        # margin. Keep at 0 unless a direct measurement of the dot plane says
        # otherwise.
        #
        # Do not re-fit this jointly with the dot spacing: that fit is discredited
        # (see MOVING_MARKER_SPACING_X_M) and the two parameters are correlated, so
        # its estimate of this one is worthless too.
        self.declare_parameter("marker_plane_height_m", 0.0)
        self.declare_parameter("marble_radius_m", 0.006)
        self.declare_parameter("corner_mask_radius_px", 12.0)
        # Off by default: the detector, not the geometry, decides whether the
        # marble is still there. The premise here was that overlapping a hole
        # means the marble has left the surface, but a hole is 7.5 mm in radius
        # and the margin adds 2.5 mm, so a marble whose centre came within
        # 10 mm of a hole centre was dropped on that same frame -- and with
        # delay_sec 0.0, rolling across a hole was indistinguishable from
        # falling into it. Measured over 259048 recorded positions the marble is
        # inside that 10 mm ring on 3.11% of frames, and of 1963 entries into
        # it 47.7% lasted 1-3 frames, i.e. it rolled straight over and kept
        # going. Once the estimator marked those invalid the episode ended
        # 0.10 s later on TAG_BALL_LOSS_GRACE_SEC.
        #
        # A marble that really drops is not visible any more, so the detector
        # reports it missing and the normal loss path handles it. Sampling
        # /tag_state_estimation/status for 60 s of live play returned 2389
        # 'valid' and zero 'ai_hole_rejected_*', so this gate was contributing
        # nothing except that risk.
        #
        # The failure it was guarding against is the detector mistaking a hole
        # for the marble and reporting it parked there forever. That now shows
        # up as an episode running to TAG_TIMEOUT_STEPS instead of ending early,
        # which is visible in the log rather than silent.
        self.declare_parameter("hole_rejection_enabled", False)
        self.declare_parameter("hole_rejection_margin_m", 0.0025)
        self.declare_parameter("hole_rejection_delay_sec", 0.0)
        self.declare_parameter("velocity_alpha", 0.65)
        self.declare_parameter("velocity_beta", 0.12)
        self.declare_parameter("max_marble_speed_mps", 2.0)
        self.declare_parameter("marker_occlusion_grace_sec", 0.20)
        self.declare_parameter("fixed_marker_max_speed_px_s", 100.0)
        self.declare_parameter("moving_marker_max_speed_px_s", 300.0)
        self.declare_parameter("marker_acquire_radius_px", 14.0)
        self.declare_parameter("moving_recover_after_frames", 20)
        # Publish on /tag_state_estimation/* by default: this node is now
        # THE estimator, and every consumer already listens there --
        # overlay_map_view_simple, tag_dreamer's env.py (estimate_subimg),
        # scripts/arduino_ball_loss_bridge.py and the hardware recorder. Set false
        # to publish under /tag_ai_map/* instead, e.g. to A/B two
        # estimators side by side.
        self.declare_parameter("publish_legacy_topics", True)
        self.declare_parameter("show_image", False)
        self.declare_parameter("camera_topic", "/tag_camera/image")

        model_path = str(self.get_parameter("ai_model_path").value)
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"AI model not found: {model_path}")
        self.ai = OnnxMarbleDetector(
            model_path,
            confidence_threshold=float(
                self.get_parameter("ai_confidence_threshold").value
            ),
        )
        self.camera_height_m = float(self.get_parameter("camera_height_m").value)
        self.marker_plane_height_m = float(
            self.get_parameter("marker_plane_height_m").value
        )
        self.marble_radius_m = float(self.get_parameter("marble_radius_m").value)
        self.corner_mask_radius_px = float(
            self.get_parameter("corner_mask_radius_px").value
        )
        self.hole_rejection_enabled = bool(
            self.get_parameter("hole_rejection_enabled").value
        )
        self.hole_rejection_margin_m = float(
            self.get_parameter("hole_rejection_margin_m").value
        )
        self.hole_rejector = TimedHoleRejector(
            float(self.get_parameter("hole_rejection_delay_sec").value)
        )
        self.show_image = bool(self.get_parameter("show_image").value)
        self.bridge = CvBridge()

        share = get_package_share_directory("tag_state_estimation")
        marker_path = os.path.join(share, "markers.csv")
        markers = np.loadtxt(marker_path, delimiter=",")
        if markers.shape != (8, 2):
            raise ValueError(
                f"{marker_path} must contain 8 [x,y] image markers; got {markers.shape}"
            )
        self.fixed_tracker = Detector(
            markers[:4],
            hsv_params_corners=FIXED_HSV,
            corner_subimage_half_size=12,
            ai_mode="off",
        )
        # half_size 12, not the Detector default of 25. Measured on live frames:
        # with a +-25 px window the raw moving tracker wanders up to 84.7 px from
        # its calibrated seeds (median 28.4), because a wider window admits
        # neighbouring blue blobs. At +-12 px -- what the fixed tracker already
        # uses -- it stays within 6.5 px while still finding all four dots 100% of
        # the time.
        self.moving_tracker = Detector(
            markers[4:], ai_mode="off", corner_subimage_half_size=12
        )
        initial_fixed_rc = np.asarray(markers[:4], dtype=np.float64)[:, ::-1]
        initial_moving_rc = np.asarray(markers[4:], dtype=np.float64)[:, ::-1]
        grace = float(self.get_parameter("marker_occlusion_grace_sec").value)
        acquire_radius = float(self.get_parameter("marker_acquire_radius_px").value)
        self.moving_recover_after_frames = max(
            1, int(self.get_parameter("moving_recover_after_frames").value)
        )
        self.fixed_guard = MarkerQuadGuard(
            initial_fixed_rc,
            mode="fixed",
            occlusion_grace_sec=grace,
            max_speed_px_s=float(
                self.get_parameter("fixed_marker_max_speed_px_s").value
            ),
            acquire_radius_px=acquire_radius,
        )
        self.moving_guard = MarkerQuadGuard(
            initial_moving_rc,
            mode="moving",
            occlusion_grace_sec=grace,
            max_speed_px_s=float(
                self.get_parameter("moving_marker_max_speed_px_s").value
            ),
            acquire_radius_px=acquire_radius,
        )
        # Frames of continuous moving-quad failure before falling back to a global
        # shape-matched search. See _recover_moving_quad.
        self.moving_fail_frames = 0
        self.pose = PlatePoseEstimator()
        self.T_world_camera = None
        self.kinematics = AlphaBetaKinematics(
            alpha=float(self.get_parameter("velocity_alpha").value),
            beta=float(self.get_parameter("velocity_beta").value),
            max_speed_mps=float(
                self.get_parameter("max_marble_speed_mps").value
            ),
        )

        legacy = bool(self.get_parameter("publish_legacy_topics").value)
        prefix = "/tag_state_estimation" if legacy else "/tag_ai_map"
        self.estimate_pub = self.create_publisher(
            StateEstimate, f"{prefix}/estimate", 1
        )
        self.subimage_pub = self.create_publisher(
            StateEstimateSub, f"{prefix}/estimate_subimg", 1
        )
        self.map_point_pub = self.create_publisher(
            PointStamped, f"{prefix}/position_map", 10
        )
        self.valid_pub = self.create_publisher(Bool, f"{prefix}/valid", 10)
        self.status_pub = self.create_publisher(String, f"{prefix}/status", 10)
        self.confidence_pub = self.create_publisher(
            Float32, f"{prefix}/ai_confidence", 10
        )
        self.camera_height_pub = self.create_publisher(
            Float32, f"{prefix}/camera_height_m", 10
        )

        # RELIABLE, matching fast_camera_publisher_v2. These frames are 640x400x3
        # = 768 KB, which DDS fragments across many UDP datagrams; under
        # BEST_EFFORT a single lost fragment discards the whole sample and there
        # is no retransmission, and the loss arrives in bursts.
        #
        # Measured on this rig with the camera publishing 40.6 Hz: a BEST_EFFORT
        # depth=1 subscriber received 13.1 Hz with stalls up to 1669 ms, leaving
        # 45.5% of elapsed time inside a gap longer than the overlay's 250 ms
        # staleness threshold -- so the marble appeared to drop out roughly half
        # the time even though every frame the estimator DID see was valid at
        # 0.994 confidence. A RELIABLE subscriber received 44.3 Hz from the same
        # publisher. Depth stays 1: newest-frame-wins is still what we want, and
        # reliability here buys fragment retransmission, not queueing.
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.create_subscription(
            Image,
            str(self.get_parameter("camera_topic").value),
            self.on_image,
            qos,
        )
        # Log the resolved path: select_markers writes to whichever workspace is
        # sourced, so a stale overlay silently feeds old clicks to this node.
        self.get_logger().info(
            f"AI-map estimator ready; model={model_path}; output={prefix}; "
            f"marble=AI only; markers=fixed 4 + moving 4 from {marker_path} "
            f"(mtime {time.ctime(os.path.getmtime(marker_path))})"
        )

    def _recover_moving_quad(self, frame):
        """Break a moving-quad lock-up with a global, shape-matched search.

        The tracker only looks in a small window around its last accepted position,
        and the estimator feeds the guard's output back into it -- so once that
        position is stale, the search is aimed at the wrong place and the quad can
        never be recovered. Observed live as moving_markers_timeout on 1550 of 1556
        frames while all four dots were detectable and the marble detector sat at
        0.995 confidence; only restarting the node cleared it.

        Runs only after sustained failure, not per frame, because it thresholds the
        whole image and searches blob combinations.
        """
        if self.moving_fail_frames < self.moving_recover_after_frames:
            return None
        self.moving_fail_frames = 0
        found = find_marker_quad_global(frame, self.moving_guard.anchor)
        if found is None:
            return None
        self.moving_guard.reseed(found)
        self.get_logger().warn(
            "moving marker quad re-acquired by global shape match "
            "after %d failed frames" % self.moving_recover_after_frames
        )
        return found

    def _update_camera_reference(self, fixed_rc):
        valid, _ = marker_quad_valid(fixed_rc, min_area_px2=5_000.0)
        if not valid:
            return False
        fixed_undistorted = self.pose.undistort_points(fixed_rc)
        T_camera_world, _, _ = self.pose.get_pose_T__C_P(
            PlatePoseEstimator.MODEL_POINTS_FIXED_CORNERS,
            fixed_undistorted,
        )
        self.T_world_camera = self.pose.invert_pose(T_camera_world)
        return True

    def _moving_pose(self, moving_rc):
        valid, reason = marker_quad_valid(moving_rc)
        if not valid:
            return None, None, reason
        moving_undistorted = self.pose.undistort_points(moving_rc)
        T_camera_board, _, _ = self.pose.get_pose_T__C_P(
            MOVING_MODEL_POINTS, moving_undistorted
        )
        if self.T_world_camera is None:
            return T_camera_board, None, "fixed_reference_missing"
        T_world_board = self.T_world_camera @ T_camera_board
        alpha, beta = self.pose.getXYAnglesFrom_R__W_M(
            T_world_board[:3, :3], deg=False
        )
        # Preserve the StateEstimate convention used by the existing node.
        return T_camera_board, (-float(beta), float(alpha)), "valid"

    @staticmethod
    def _roi_from_corners(corners_rc, frame_shape):
        corners = np.asarray(corners_rc, dtype=np.float32)
        if corners.shape != (4, 2) or not np.all(np.isfinite(corners)):
            return None
        height, width = frame_shape[:2]
        rows, cols = corners[:, 0], corners[:, 1]
        return (
            max(0.0, float(cols.min()) / width),
            max(0.0, float(rows.min()) / height),
            min(1.0, float(cols.max()) / width),
            min(1.0, float(rows.max()) / height),
        )

    def _publish(self, message, frame, confidence, map_xy, status, valid):
        state = StateEstimate()
        state.x_b, state.y_b = map(float, message[0])
        state.x_b_dot, state.y_b_dot = map(float, message[1])
        state.alpha, state.beta = map(float, message[2])
        self.estimate_pub.publish(state)

        sub = StateEstimateSub()
        sub.state = state
        sub.subimg = self.bridge.cv2_to_imgmsg(message[3], encoding="bgr8")
        sub.subimg.header = frame.header
        self.subimage_pub.publish(sub)

        point = PointStamped()
        point.header = frame.header
        point.header.frame_id = "tag_map_lower_left"
        point.point.x, point.point.y = map(float, map_xy)
        point.point.z = float(confidence)
        self.map_point_pub.publish(point)
        self.valid_pub.publish(Bool(data=bool(valid)))
        self.status_pub.publish(String(data=str(status)))
        self.confidence_pub.publish(Float32(data=float(confidence)))

    def on_image(self, image_message):
        frame = self.bridge.imgmsg_to_cv2(image_message, desired_encoding="bgr8")
        timestamp = ros_time_seconds(image_message.header.stamp)
        fixed_raw_rc = self.fixed_tracker.detect_corners(frame)
        fixed_rc, fixed_valid, fixed_status = self.fixed_guard.update(
            fixed_raw_rc, self.fixed_tracker.corner_found, timestamp
        )
        # Keep the detector's next local search window attached to the accepted
        # marker positions, never to a rejected outside blob.
        self.fixed_tracker.corners = fixed_rc.astype(np.float32)
        self.fixed_tracker.corners_missing = False
        if fixed_valid:
            fixed_valid = self._update_camera_reference(fixed_rc)
            if not fixed_valid:
                fixed_status = "fixed_marker_geometry_invalid"

        moving_raw_rc = self.moving_tracker.detect_corners(frame)
        moving_rc, moving_valid, moving_status = self.moving_guard.update(
            moving_raw_rc, self.moving_tracker.corner_found, timestamp
        )
        if moving_valid:
            self.moving_fail_frames = 0
        else:
            self.moving_fail_frames += 1
            recovered = self._recover_moving_quad(frame)
            if recovered is not None:
                moving_rc, moving_valid, moving_status = self.moving_guard.update(
                    recovered, [True] * 4, timestamp
                )
        self.moving_tracker.corners = moving_rc.astype(np.float32)
        self.moving_tracker.corners_missing = False
        if moving_valid:
            T_camera_board, angles, pose_status = self._moving_pose(moving_rc)
        else:
            T_camera_board, angles, pose_status = None, None, moving_status
        if angles is None:
            angles = (math.nan, math.nan)

        roi = self._roi_from_corners(moving_rc, frame.shape) if moving_valid else None
        excluded = (
            np.vstack((fixed_rc[:, ::-1], moving_rc[:, ::-1]))
            if moving_valid and fixed_valid
            else None
        )
        detection = self.ai.detect(
            frame,
            valid_roi=roi,
            exclude_centers_px=excluded,
            exclude_radius_px=self.corner_mask_radius_px if excluded is not None else 0.0,
        )
        ball_xy = (
            np.asarray([detection.x_px, detection.y_px], dtype=np.float64)
            if detection.visible
            else None
        )

        status = pose_status
        measurement = map_ai_pixel(
            ball_xy,
            moving_rc if moving_valid else None,
            T_camera_board=T_camera_board,
            camera_height_fallback_m=self.camera_height_m,
            marker_plane_height_m=self.marker_plane_height_m,
            marble_radius_m=self.marble_radius_m,
        )
        angles_valid = bool(np.all(np.isfinite(angles)))
        valid = bool(measurement.valid and fixed_valid and angles_valid)
        marker_status = "valid"
        if not fixed_valid:
            marker_status = fixed_status
            valid = False
        elif not moving_valid:
            marker_status = moving_status
            valid = False
        elif fixed_status != "valid":
            marker_status = fixed_status
        elif moving_status != "valid":
            marker_status = moving_status
        if measurement.valid and self.hole_rejection_enabled:
            hole = candidate_hole_index(
                np.asarray([detection.y_px, detection.x_px], dtype=np.float32),
                moving_rc,
                margin_m=self.hole_rejection_margin_m,
            )
            rejected, _ = self.hole_rejector.update(hole)
            if rejected:
                valid = False
                status = f"ai_hole_rejected_{int(hole) + 1}"
        else:
            self.hole_rejector.update(None)

        nan2 = np.full(2, np.nan, dtype=np.float64)
        map_xy = measurement.lower_left_xy if measurement.valid else nan2
        position = velocity = nan2
        if valid:
            position, velocity, filter_status = self.kinematics.update(
                measurement.centered_xy, timestamp
            )
            valid = bool(np.all(np.isfinite(position)))
            status = marker_status if valid else filter_status
        else:
            if marker_status != "valid":
                status = marker_status
            elif not measurement.valid:
                status = measurement.reason

        # Invalid means absent to every consumer: do not leak a rejected raw AI
        # candidate through the map point or training subimage.
        published_map_xy = map_xy if valid else nan2
        subimage = crop_ball_image(frame, ball_xy if valid else None)
        self._publish(
            (position, velocity, angles, subimage),
            image_message,
            detection.confidence,
            published_map_xy,
            status,
            valid,
        )
        height = (
            measurement.camera_height_m
            if np.isfinite(measurement.camera_height_m)
            else self.camera_height_m
        )
        self.camera_height_pub.publish(Float32(data=float(height)))

        if self.show_image:
            display = frame.copy()
            for index, corner in enumerate(fixed_rc):
                cv2.drawMarker(
                    display, tuple(np.round(corner[::-1]).astype(int)),
                    (255, 255, 255), cv2.MARKER_CROSS, 10, 1
                )
                cv2.putText(
                    display, f"F{index + 1}",
                    tuple(np.round(corner[::-1] + [5, -5]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1
                )
            for index, corner in enumerate(moving_rc):
                cv2.drawMarker(
                    display, tuple(np.round(corner[::-1]).astype(int)),
                    (255, 255, 0), cv2.MARKER_CROSS, 10, 1
                )
                cv2.putText(
                    display, f"M{index + 1}",
                    tuple(np.round(corner[::-1] + [5, -5]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1
                )
            if valid:
                cv2.circle(
                    display,
                    tuple(np.round(ball_xy).astype(int)),
                    6,
                    (0, 255, 0),
                    2,
                )
            cv2.putText(
                display, status, (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 255, 0) if valid else (0, 0, 255), 2
            )
            cv2.imshow("AI Map Estimator", display)
            if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = AiMapEstimatorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    cv2.destroyAllWindows()
    if rclpy.ok():
        node.destroy_node()
        rclpy.shutdown()
