"""ROS2 ONNX marble detector with isolated diagnostic outputs."""

import math
import os
from pathlib import Path

import cv2
import numpy as np
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Float32

from tag_state_estimation.ai_marble_common import OnnxMarbleDetector
from tag_state_estimation.core.detection import Detector
from tag_state_estimation.core.hole_mask import (
    TimedHoleRejector,
    candidate_hole_index,
)

MARBLE_DIAMETER_M = 0.012
# Fitted from 40 tilted views / 745 hole observations by
# tools/fit_marker_geometry.py, NOT the ETH original's 0.269 x 0.237. Those were
# inherited nominal values for different hardware; this board runs a custom maze.
# Profiling the dot-plane height h against the known DXF hole positions gives a
# clear minimum at h = 10 mm (median residual 2.13 mm at h=0, 1.49 mm at h=10,
# 1.92 mm at h=20), independently reproducing a 1 cm ruler measurement, and at
# that optimum the spacing is 249.2 x 222.3 mm. Median hole residual over the
# whole set improves 5.67 -> 1.49 mm versus the old constants.
CORNER_SPAN_X_M = 0.2492
CORNER_SPAN_Y_M = 0.2223


def projected_marble_radius_px(moving_corners_rc, fallback=6):
    """Estimate the visible 6 mm radius from the live board-marker scale."""
    corners = np.asarray(moving_corners_rc, dtype=np.float32)
    if corners.shape != (4, 2) or not np.all(np.isfinite(corners)):
        return int(fallback)

    # Marker order: lower-left, lower-right, upper-right, upper-left.
    horizontal_scales = (
        np.linalg.norm(corners[1] - corners[0]) / CORNER_SPAN_X_M,
        np.linalg.norm(corners[2] - corners[3]) / CORNER_SPAN_X_M,
    )
    vertical_scales = (
        np.linalg.norm(corners[3] - corners[0]) / CORNER_SPAN_Y_M,
        np.linalg.norm(corners[2] - corners[1]) / CORNER_SPAN_Y_M,
    )
    pixels_per_meter = float(np.median(horizontal_scales + vertical_scales))
    radius = 0.5 * MARBLE_DIAMETER_M * pixels_per_meter
    return int(np.clip(round(radius), 2, 30))


class AiMarbleDetectorNode(Node):
    def __init__(self):
        super().__init__("tag_ai_marble_detector")
        self.declare_parameter("model_path", "")
        self.declare_parameter("camera_topic", "/tag_camera/image")
        self.declare_parameter("confidence_threshold", 0.90)
        self.declare_parameter("input_width", 320)
        self.declare_parameter("input_height", 200)
        self.declare_parameter("backend", "cpu")
        self.declare_parameter("show_image", True)
        self.declare_parameter("publish_diagnostics", True)
        self.declare_parameter("miss_grace_frames", 90)
        self.declare_parameter("smoothing", 0.35)
        self.declare_parameter("roi_x_min", 0.25)
        self.declare_parameter("roi_y_min", 0.15)
        self.declare_parameter("roi_x_max", 0.72)
        self.declare_parameter("roi_y_max", 0.80)
        self.declare_parameter("hole_rejection_enabled", True)
        self.declare_parameter("hole_rejection_margin_m", 0.0025)
        self.declare_parameter("hole_rejection_delay_sec", 2.0)
        model_path = Path(str(self.get_parameter("model_path").value)).expanduser()
        if not str(model_path) or not model_path.is_file():
            raise FileNotFoundError("Set model_path to a trained marble_detector.onnx file")

        self.detector = OnnxMarbleDetector(
            model_path,
            input_width=int(self.get_parameter("input_width").value),
            input_height=int(self.get_parameter("input_height").value),
            confidence_threshold=float(self.get_parameter("confidence_threshold").value),
            backend=str(self.get_parameter("backend").value),
            valid_roi=(
                float(self.get_parameter("roi_x_min").value),
                float(self.get_parameter("roi_y_min").value),
                float(self.get_parameter("roi_x_max").value),
                float(self.get_parameter("roi_y_max").value),
            ),
        )
        self.show_image = bool(self.get_parameter("show_image").value)
        self.publish_diagnostics = bool(self.get_parameter("publish_diagnostics").value)
        self.hole_rejection_enabled = bool(
            self.get_parameter("hole_rejection_enabled").value
        )
        self.hole_rejection_margin_m = max(
            0.0, float(self.get_parameter("hole_rejection_margin_m").value)
        )
        self.hole_rejection_delay_sec = max(
            0.0, float(self.get_parameter("hole_rejection_delay_sec").value)
        )
        self.hole_rejector = TimedHoleRejector(self.hole_rejection_delay_sec)
        self.corner_tracker = None
        if self.hole_rejection_enabled:
            share = get_package_share_directory("tag_state_estimation")
            markers = np.loadtxt(os.path.join(share, "markers.csv"), delimiter=",")
            self.corner_tracker = Detector(markers[4:], ai_mode="off")
        self.miss_grace = int(self.get_parameter("miss_grace_frames").value)
        self.smoothing = float(self.get_parameter("smoothing").value)
        self.bridge = CvBridge()
        self.filtered = None
        self.misses = 0
        if self.publish_diagnostics:
            self.point_publisher = self.create_publisher(
                PointStamped, "/tag_ai_marble/pixel", 10
            )
            self.confidence_publisher = self.create_publisher(
                Float32, "/tag_ai_marble/confidence", 10
            )
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        camera_topic = str(self.get_parameter("camera_topic").value)
        self.create_subscription(Image, camera_topic, self._on_image, qos)
        self.get_logger().info(
            f"AI marble detector loaded {model_path}; camera={camera_topic}; "
            "motor/control topics are not used."
        )

    def _on_image(self, message):
        frame = self.bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
        detection = self.detector.detect(frame)
        rejected_hole = None
        hole_candidate = None
        hole_elapsed_sec = 0.0
        moving_corners = None
        if detection.visible and self.corner_tracker is not None:
            moving_corners = self.corner_tracker.detect_corners(frame)
            if not self.corner_tracker.corners_missing:
                hole_candidate = candidate_hole_index(
                    np.asarray([detection.y_px, detection.x_px], dtype=np.float32),
                    moving_corners,
                    margin_m=self.hole_rejection_margin_m,
                )
            reject_hole, hole_elapsed_sec = self.hole_rejector.update(hole_candidate)
            if reject_hole:
                rejected_hole = hole_candidate
                # Do not retain a previously smoothed point at the rejected
                # location: diagnostics must immediately say "not visible".
                self.filtered = None
                self.misses = self.miss_grace + 1
        else:
            self.hole_rejector.update(None)
        if detection.visible:
            if rejected_hole is None:
                point = (detection.x_px, detection.y_px)
                if self.filtered is None:
                    self.filtered = point
                else:
                    keep = 1.0 - self.smoothing
                    self.filtered = (
                        keep * self.filtered[0] + self.smoothing * point[0],
                        keep * self.filtered[1] + self.smoothing * point[1],
                    )
                self.misses = 0
        else:
            self.misses += 1
        visible = self.filtered is not None and self.misses <= self.miss_grace

        if self.publish_diagnostics:
            point_message = PointStamped()
            point_message.header = message.header
            if visible:
                point_message.point.x, point_message.point.y = self.filtered
            else:
                point_message.point.x = math.nan
                point_message.point.y = math.nan
            point_message.point.z = detection.confidence
            self.point_publisher.publish(point_message)
            confidence = Float32()
            confidence.data = detection.confidence
            self.confidence_publisher.publish(confidence)

        if self.show_image:
            color = (0, 255, 0) if visible else (0, 0, 255)
            if visible:
                center = (round(self.filtered[0]), round(self.filtered[1]))
                radius_px = projected_marble_radius_px(moving_corners)
                cv2.circle(frame, center, radius_px, color, 2)
                cv2.drawMarker(
                    frame,
                    center,
                    color,
                    cv2.MARKER_CROSS,
                    max(7, 2 * radius_px + 1),
                    1,
                )
            if rejected_hole is not None:
                status = (
                    f"HOLE {rejected_hole + 1} REJECTED after "
                    f"{self.hole_rejection_delay_sec:.1f}s"
                )
            elif hole_candidate is not None:
                status = (
                    f"HOLE {hole_candidate + 1} PENDING "
                    f"{hole_elapsed_sec:.1f}/{self.hole_rejection_delay_sec:.1f}s "
                    "- MARBLE ACCEPTED"
                )
            else:
                status = f"AI confidence={detection.confidence:.3f} misses={self.misses}"
            cv2.putText(
                frame,
                status,
                (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("AI Marble Detector", frame)
            if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = AiMarbleDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    cv2.destroyAllWindows()
    if rclpy.ok():
        node.destroy_node()
        rclpy.shutdown()
