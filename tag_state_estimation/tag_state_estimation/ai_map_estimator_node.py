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
from tag_state_estimation.core import hsv_marble
from tag_state_estimation.core.hybrid_ball import HybridBallTracker
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
        # HSV marble detection alongside the learned one. off | shadow | fuse.
        #
        # Why it is worth having: losses cluster on a fast marble. Measured over
        # 9088 steps, 0.5% exceeded 10 mm in a frame, but among the steps
        # immediately before an episode-ending loss 6.9% did -- a 14x
        # over-representation. Colour and a convnet do not fail on the same
        # frames, so either one carrying the frame is better than neither.
        #
        # Why it was not here already: the marble is blue and so are the eight
        # reference dots, so an unrestricted HSV candidate locks onto a dot.
        # core/hsv_marble.py clips the search to the maze interior, which the dots
        # sit 4-5 mm outside, and that removes the ambiguity geometrically rather
        # than by guessing at blob sizes.
        #
        # shadow computes the candidate and logs whether it had the marble without
        # letting it decide anything, which is how to measure the hit rate before
        # handing it authority. fuse lets either detector carry the frame.
        self.declare_parameter("hsv_marble_mode", "fuse")
        # Gap length, in frames, that still counts as the same marble track. A
        # candidate arriving within it is accepted straight away if it is no
        # further from the last position than the marble could have rolled;
        # anything longer or further still serves the full confirmation window.
        # 0 restores the old behaviour of confirming after every single miss.
        self.declare_parameter("fast_reacquire_frames", 3)
        # Radius of the disc the colour search is confined to when the marble was
        # seen recently, for the first frame after that sighting; it grows by the
        # same amount per frame the marble stays unseen, since the marble keeps
        # rolling. 32 px is one frame of travel at max_reacquire_jump_px plus the
        # marble's own 6 px radius.
        #
        # The point is not speed, it is being able to lower the area floor. Across
        # 120 live frames nothing but the marble passed the blue filter anywhere
        # in the maze, and nothing at all appeared within 56 px of it, so inside
        # this disc the size gate rejects nothing real and only costs margin --
        # margin the marble needs when a wall occludes or shades half of it.
        # 0 disables the local pass and searches the whole maze every frame.
        self.declare_parameter("hsv_search_radius_px", 32.0)
        # Frames the local disc may keep growing before the search gives up on the
        # last sighting and goes back to the whole maze at the strict floor.
        self.declare_parameter("hsv_search_max_age_frames", 3)
        # Disc erased from every corner search window around the marble's previous
        # position. 10 px is about 5 mm here, so it covers a frame of travel with
        # margin while staying well inside the 10 mm that separates a dot centre
        # from the closest a marble centre can get to it.
        self.declare_parameter("marble_exclude_radius_px", 10.0)
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
        # A narrow descendant of the gate above, and the reason that one can stay
        # off. It drops an AI candidate ONLY when all of these hold at once:
        # colour also has a candidate, the two disagree by more than
        # hole_tiebreak_disagreement_px, the AI is sitting on a hole, and colour
        # is not. Measured live, the detector calls a black hole the marble on
        # about 3% of the frames it fires, at confidences near 0.87.
        #
        # What made the old blanket gate harmful was dropping the marble as it
        # merely rolled over a hole -- 47.7% of hole entries lasted 1-3 frames.
        # This cannot do that: a marble on a hole is a marble colour ALSO sees on
        # that hole, the two agree, and nothing is dropped. It only fires when
        # the two sources point at different places and one of them is a hole,
        # which colour cannot mistake, since a hole is black and the filter only
        # passes blue.
        #
        # The bound matters more than it looks. Hole #11 sits 22.7 mm from
        # waypoint 100, which is 23 px -- inside max_reacquire_jump_px -- so the
        # tracker's own distance test cannot catch that one and hands over a
        # position 23 mm off, in the middle of the run the marble is trying to
        # complete.
        self.declare_parameter("hole_tiebreak_enabled", True)
        self.declare_parameter("hole_tiebreak_disagreement_px", 12.0)
        # Who settles it when the two detectors point at different places.
        # always | open | off.
        #
        # always: colour wins, wherever the marble is. That is the setting here,
        # because colour has no failure mode for this scene and the learned
        # detector has a well measured one -- it reports a black hole as the
        # marble on about 3% of the frames it fires, at confidences near 0.87,
        # sometimes most of the board away. Colour cannot make that mistake: the
        # filter passes blue and a hole is black.
        #
        # This costs less than it looks. It does NOT sideline the AI: a
        # disagreement needs both sources to have a candidate, so the frames the
        # AI carries alone -- about 1% of them, where colour has nothing -- are
        # untouched, and when the two agree within agreement_radius_px the
        # published point is still the 50/50 blend.
        #
        # The one real cost is at the rim, where the maze mask clips part of the
        # marble: 16% of the blob against the right wall. Rasterised against the
        # real homography that biases colour's centroid by 0.6 mm at a wall and
        # 0.9 mm in a corner, against a median AI-to-colour gap of 2.3 mm. Small
        # enough to prefer over handing those frames to a source that can put the
        # marble on a hole.
        #
        # coverage: colour settles it as long as the maze mask has not eaten too
        # much of the marble -- the condition that actually matters, rather than
        # distance from the rim standing in for it. Measured on the real
        # homography, the mask keeps 100% of the marble in the middle of the
        # board, 92% against the top wall and 84% against the right wall and in
        # the corners; the resulting centroid bias is 0.6 mm at a wall and 0.9 mm
        # in a corner. So with the threshold at 0.70 this board never actually
        # hands a disagreement to the AI -- which is the intent -- but the rule
        # still does the right thing if the inset, the lens or the board change.
        #
        # open: colour only settles it while at least hsv_priority_margin_m
        # inside the maze edge.
        self.declare_parameter("hsv_priority_mode", "coverage")
        self.declare_parameter("hsv_min_coverage", 0.70)
        self.declare_parameter("hsv_priority_margin_m", 0.010)
        self.declare_parameter("hsv_priority_disagreement_px", 12.0)
        self.declare_parameter("hole_rejection_margin_m", 0.0025)
        self.declare_parameter("hole_rejection_delay_sec", 0.0)
        self.declare_parameter("velocity_alpha", 0.65)
        self.declare_parameter("velocity_beta", 0.12)
        # Off (0 disables it). It fired on 18 of 6000 measured frames, and every
        # one of those was a real marble the detectors had located -- it rejected
        # them purely for arriving faster than 2 m/s, which on a steep run they
        # legitimately do. The estimate then went invalid on a frame where the
        # marble was in plain sight.
        #
        # The failure it guarded against, a wildly displaced detection entering
        # the filter, is already covered upstream and more cheaply:
        # HybridBallTracker will not accept any candidate further from the last
        # position than max_reacquire_jump_px, which is the same bound expressed
        # in pixels per frame. Two gates for one job, and this was the one that
        # could only report loss after the fact.
        self.declare_parameter("max_marble_speed_mps", 0.0)
        # The remaining speed bound, and now the only one. Derived rather than
        # guessed: the moving dots span 277 px for 269 mm, so 1.03 px/mm, and the
        # camera runs ~45 fps, which puts 2 m/s at 46 px per frame. The old 25 px
        # was 1.05 m/s -- BELOW the 2 m/s the node itself claimed to support, so
        # a marble running faster than that with only one detector on it was
        # dropped every single frame, and _confirm_reacquisition could not
        # recover it either since it tests against the same bound.
        self.declare_parameter("max_reacquire_jump_px", 48.0)
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
        self.hsv_marble_mode = str(
            self.get_parameter("hsv_marble_mode").value
        ).strip().lower()
        self.hybrid_ball = HybridBallTracker(
            trust_hsv_alone=self.hsv_marble_mode == "fuse",
            fast_reacquire_frames=int(
                self.get_parameter("fast_reacquire_frames").value
            ),
            max_reacquire_jump_px=float(
                self.get_parameter("max_reacquire_jump_px").value
            ),
        )
        self.hsv_search_radius_px = float(
            self.get_parameter("hsv_search_radius_px").value
        )
        self.hsv_search_max_age = int(
            self.get_parameter("hsv_search_max_age_frames").value
        )
        # Where colour last had the marble, and how many frames ago, so the local
        # disc can follow a marble that goes missing for more than one frame.
        self.hsv_search_px = None
        self.hsv_search_age = 0
        self.hsv_local_hits = 0
        self.hole_tiebreak_enabled = bool(
            self.get_parameter("hole_tiebreak_enabled").value
        )
        self.hole_tiebreak_disagreement_px = float(
            self.get_parameter("hole_tiebreak_disagreement_px").value
        )
        self.hole_tiebreak_drops = 0
        self.hsv_priority_mode = str(
            self.get_parameter("hsv_priority_mode").value
        ).strip().lower()
        if self.hsv_priority_mode not in ("coverage", "always", "open", "off"):
            raise ValueError(
                f"hsv_priority_mode must be coverage|always|open|off, got "
                f"{self.hsv_priority_mode}"
            )
        self.hsv_min_coverage = float(
            self.get_parameter("hsv_min_coverage").value
        )
        self.last_hsv_coverage = float("nan")
        self.hsv_priority_margin_m = float(
            self.get_parameter("hsv_priority_margin_m").value
        )
        self.hsv_priority_disagreement_px = float(
            self.get_parameter("hsv_priority_disagreement_px").value
        )
        self.hsv_priority_wins = 0
        self.ai_rejected_reason = None
        self.hsv_marble_stats = {"ai": 0, "hsv": 0, "both": 0, "neither": 0}
        self.last_ball_source = "ai"
        self.last_hsv_xy = None
        # Where the marble was on the previous frame, fed back into the corner
        # search so it cannot be mistaken for a dot.
        self.last_ball_px = None
        self.marble_exclude_radius_px = float(
            self.get_parameter("marble_exclude_radius_px").value
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

    def colour_coverage(self, hsv_xy, moving_rc):
        """Fraction of the marble still inside the colour search polygon.

        1.0 anywhere the mask is not cutting the marble; it falls off only in the
        last few millimetres before a wall, where the marble's outline crosses
        the polygon that DEFAULT_INSET_M pulls in. This is the quantity that
        biases colour's centroid, so it is the one worth testing -- rather than
        distance from the rim, which is only a proxy for it.

        Rasterised on a small local canvas: the whole thing is a disc a dozen
        pixels across.
        """
        polygon = hsv_marble.maze_polygon_px(moving_rc)
        if polygon is None or hsv_xy is None:
            return float("nan")
        span = float(np.linalg.norm(np.asarray(moving_rc[1]) - np.asarray(moving_rc[0])))
        if not np.isfinite(span) or span <= 1.0:
            return float("nan")
        radius = int(round(self.marble_radius_m * 1000.0 * span / 269.0))
        if radius < 1:
            return float("nan")
        pad = radius + 2
        cx, cy = float(hsv_xy[0]), float(hsv_xy[1])
        side = 2 * pad + 1
        origin = np.array([cx - pad, cy - pad], dtype=np.float32)
        disc = np.zeros((side, side), np.uint8)
        cv2.circle(disc, (pad, pad), radius, 255, -1)
        inside = np.zeros((side, side), np.uint8)
        cv2.fillConvexPoly(inside, (polygon - origin).astype(np.int32), 255)
        total = int(np.count_nonzero(disc))
        if total == 0:
            return float("nan")
        return float(np.count_nonzero(cv2.bitwise_and(disc, inside))) / total

    def colour_should_settle(self, ai_xy, hsv_xy, moving_rc):
        """True when a disagreement should be settled by colour, not the AI.

        A disagreement needs both sources to have produced something, so this
        never touches a frame the AI is carrying on its own.
        """
        if self.hsv_priority_mode == "off":
            return False
        if ai_xy is None or hsv_xy is None:
            return False
        if (
            float(np.linalg.norm(np.asarray(ai_xy) - np.asarray(hsv_xy)))
            <= self.hsv_priority_disagreement_px
        ):
            return False
        if self.hsv_priority_mode == "always":
            return True
        if self.hsv_priority_mode == "coverage":
            coverage = self.colour_coverage(hsv_xy, moving_rc)
            self.last_hsv_coverage = coverage
            # An uncomputable coverage means the quad or the scale is unusable,
            # and colour is only trustworthy while the mask it depends on is.
            return bool(np.isfinite(coverage)) and coverage >= self.hsv_min_coverage
        # "open": only where the maze mask is not clipping the marble.
        interior = hsv_marble.maze_polygon_px(
            moving_rc, inset_m=self.hsv_priority_margin_m
        )
        if interior is None:
            return False
        point = (float(hsv_xy[0]), float(hsv_xy[1]))
        return cv2.pointPolygonTest(interior, point, False) >= 0

    def ai_landed_on_a_hole(self, ai_xy, hsv_xy, moving_rc):
        """Index of the hole the AI mistook for the marble, or None.

        Only answers when colour has a candidate of its own and puts the marble
        somewhere else. A marble genuinely crossing a hole is a marble colour
        sees on that same hole, the two agree, and this stays quiet -- which is
        what keeps it from repeating the mistake of the blanket hole gate.
        """
        if not self.hole_tiebreak_enabled:
            return None
        if ai_xy is None or hsv_xy is None:
            return None
        if (
            float(np.linalg.norm(np.asarray(ai_xy) - np.asarray(hsv_xy)))
            <= self.hole_tiebreak_disagreement_px
        ):
            return None
        # candidate_hole_index works in the trackers' [row, column].
        ai_hole = candidate_hole_index(
            np.asarray(ai_xy, dtype=np.float32)[::-1],
            moving_rc,
            margin_m=self.hole_rejection_margin_m,
        )
        if ai_hole is None:
            return None
        hsv_hole = candidate_hole_index(
            np.asarray(hsv_xy, dtype=np.float32)[::-1],
            moving_rc,
            margin_m=self.hole_rejection_margin_m,
        )
        return None if hsv_hole == ai_hole else int(ai_hole)

    def detect_marble_by_colour(self, frame, moving_rc):
        """Colour search: a disc around the last sighting first, then the maze.

        The local pass exists to let the area floor drop where it is safe to. It
        is strictly additive -- if it finds nothing the full-maze search runs
        unchanged at the strict floor, so this can only add detections, never
        remove one that the previous code would have made.
        """
        local = None
        if (
            self.hsv_search_radius_px > 0.0
            and self.hsv_search_px is not None
            and self.hsv_search_age <= self.hsv_search_max_age
        ):
            local = hsv_marble.detect_marble(
                frame,
                moving_rc,
                area_px2=hsv_marble.DEFAULT_LOCAL_AREA_PX2,
                search_center_px=self.hsv_search_px,
                search_radius_px=self.hsv_search_radius_px
                * float(self.hsv_search_age + 1),
            )
        if local is not None:
            self.hsv_local_hits += 1
            self.hsv_search_px = np.asarray(local, dtype=np.float64)
            self.hsv_search_age = 0
            return local

        found = hsv_marble.detect_marble(frame, moving_rc)
        if found is not None:
            self.hsv_search_px = np.asarray(found, dtype=np.float64)
            self.hsv_search_age = 0
        else:
            self.hsv_search_age += 1
            if self.hsv_search_age > self.hsv_search_max_age:
                self.hsv_search_px = None
        return found

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

        # Deny the corner search the marble. It was located on the previous frame,
        # and both marker sets are blue, so this is the only guard that holds when
        # the marble sits at the edge of a corner window -- there the size gate
        # sees ~64 px2 of a clipped marble, inside the 42-65 px2 a dot occupies.
        # Radius covers a frame of travel: p90 is 3.7 mm, about 7 px here.
        for tracker in (self.fixed_tracker, self.moving_tracker):
            tracker.exclude_px = self.last_ball_px
            tracker.exclude_radius_px = self.marble_exclude_radius_px
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
        ai_xy = (
            np.asarray([detection.x_px, detection.y_px], dtype=np.float64)
            if detection.visible
            else None
        )
        ball_xy = ai_xy
        self.last_hsv_xy = None
        if self.hsv_marble_mode in ("shadow", "fuse") and moving_valid:
            hsv_xy = self.detect_marble_by_colour(frame, moving_rc)
            key = (
                "both" if (ai_xy is not None and hsv_xy is not None)
                else "ai" if ai_xy is not None
                else "hsv" if hsv_xy is not None
                else "neither"
            )
            self.hsv_marble_stats[key] += 1
            total = sum(self.hsv_marble_stats.values())
            if total % 500 == 0:
                counts = self.hsv_marble_stats
                self.get_logger().info(
                    "marble sources over %d frames: both %.1f%%, ai only %.1f%%, "
                    "hsv only %.1f%%, neither %.1f%% -- 'hsv only' is what the "
                    "pairing buys; %.1f%% of colour hits came from the local "
                    "disc; colour settled %d disagreements in the open, %d on "
                    "a hole"
                    % (
                        total,
                        100.0 * counts["both"] / total,
                        100.0 * counts["ai"] / total,
                        100.0 * counts["hsv"] / total,
                        100.0 * counts["neither"] / total,
                        100.0 * self.hsv_local_hits / total,
                        self.hsv_priority_wins,
                        self.hole_tiebreak_drops,
                    )
                )
            self.last_hsv_xy = hsv_xy
            if self.hsv_marble_mode == "fuse":
                ai_for_fusion = ai_xy
                self.ai_rejected_reason = None
                # Order matters only for the bookkeeping: both rules drop the
                # same candidate, and a frame that trips both should be counted
                # once, under the more specific diagnosis.
                hole = self.ai_landed_on_a_hole(ai_xy, hsv_xy, moving_rc)
                if hole is None and self.colour_should_settle(
                    ai_xy, hsv_xy, moving_rc
                ):
                    self.hsv_priority_wins += 1
                    self.ai_rejected_reason = (
                        "disagreed, colour %.0f%% covered"
                        % (100.0 * self.last_hsv_coverage)
                        if np.isfinite(self.last_hsv_coverage)
                        else "disagreed"
                    )
                    ai_for_fusion = None
                if hole is not None:
                    self.ai_rejected_reason = f"hole {hole + 1}"
                    self.hole_tiebreak_drops += 1
                    self.get_logger().warn(
                        "AI put the marble on hole %d, %.0f px from where "
                        "colour had it; keeping colour (%d so far)"
                        % (
                            hole + 1,
                            float(np.linalg.norm(ai_xy - hsv_xy)),
                            self.hole_tiebreak_drops,
                        )
                    )
                    ai_for_fusion = None
                fused = self.hybrid_ball.update(
                    hsv_position=hsv_xy, ai_position=ai_for_fusion
                )
                self.last_ball_source = fused.source
                if np.all(np.isfinite(fused.measurement)):
                    ball_xy = np.asarray(fused.measurement, dtype=np.float64)
                else:
                    ball_xy = None

        self.last_ball_px = (
            np.asarray(ball_xy, dtype=np.float64) if ball_xy is not None else None
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
            # The maze mask the HSV search is clipped to. Everything outside it
            # is where the reference dots live, so seeing it is how you check the
            # two are actually separated on this board rather than in principle.
            if self.hsv_marble_mode != "off" and moving_valid:
                polygon = hsv_marble.maze_polygon_px(moving_rc)
                if polygon is not None:
                    cv2.polylines(
                        display, [polygon.astype(np.int32)], True, (0, 200, 255), 1
                    )
            # Each detector drawn separately, so a disagreement is visible rather
            # than averaged away: magenta square is colour, yellow diamond is the
            # learned detector, green circle is what was published.
            if self.last_hsv_xy is not None:
                cv2.drawMarker(
                    display, tuple(np.round(self.last_hsv_xy).astype(int)),
                    (255, 0, 255), cv2.MARKER_SQUARE, 16, 2
                )
            if ai_xy is not None:
                point = tuple(np.round(ai_xy).astype(int))
                cv2.drawMarker(
                    display, point, (0, 255, 255), cv2.MARKER_DIAMOND, 16, 2
                )
                # Struck through in red when this frame's AI candidate was
                # thrown away, so "the yellow diamond jumped onto a hole" and
                # "the yellow diamond dragged the estimate onto a hole" are
                # different pictures rather than the same one.
                if self.ai_rejected_reason is not None:
                    cv2.drawMarker(
                        display, point, (0, 0, 255), cv2.MARKER_TILTED_CROSS, 22, 2
                    )
                    cv2.putText(
                        display, f"AI dropped: {self.ai_rejected_reason}",
                        (point[0] + 14, point[1] + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 255), 1,
                        cv2.LINE_AA,
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
            if self.hsv_marble_mode != "off":
                counts = self.hsv_marble_stats
                total = max(1, sum(counts.values()))
                cv2.putText(
                    display,
                    "hsv=%s ai=%s  src=%s" % (
                        "yes" if self.last_hsv_xy is not None else "no",
                        "yes" if ai_xy is not None else "no",
                        self.last_ball_source,
                    ),
                    (10, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
                )
                cv2.putText(
                    display,
                    "both %.0f%%  ai only %.0f%%  HSV ONLY %.0f%%  neither %.0f%%"
                    % (
                        100.0 * counts["both"] / total,
                        100.0 * counts["ai"] / total,
                        100.0 * counts["hsv"] / total,
                        100.0 * counts["neither"] / total,
                    ),
                    (10, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 255, 200), 1
                )
            cv2.imshow("AI + HSV Map Estimator", display)
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
