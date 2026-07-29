"""Passive live view of AI detections and board-relative hole masks."""

import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
import cv2
from cv_bridge import CvBridge
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image

from tag_state_estimation.ai_marble_common import OnnxMarbleDetector
from tag_state_estimation.core.detection import Detector
from tag_state_estimation.core.hole_mask import (
    TimedHoleRejector,
    candidate_hole_index,
    project_holes_to_image,
)


class AiHoleMaskViewer(Node):
    """Subscribe-only diagnostic viewer; it contains no ROS publishers."""

    def __init__(self):
        super().__init__("tag_ai_hole_mask_viewer")
        self.declare_parameter("model_path", "")
        self.declare_parameter("camera_topic", "/tag_camera/image")
        self.declare_parameter("confidence_threshold", 0.90)
        self.declare_parameter("hole_margin_m", 0.0025)
        self.declare_parameter("hole_rejection_delay_sec", 2.0)
        self.declare_parameter("process_every_n", 3)
        self.declare_parameter("roi_x_min", 0.25)
        self.declare_parameter("roi_y_min", 0.15)
        self.declare_parameter("roi_x_max", 0.72)
        self.declare_parameter("roi_y_max", 0.80)

        model_path = Path(str(self.get_parameter("model_path").value)).expanduser()
        if not model_path.is_file():
            raise FileNotFoundError("Set model_path to marble_detector.onnx")
        self.ai = OnnxMarbleDetector(
            model_path,
            confidence_threshold=float(
                self.get_parameter("confidence_threshold").value
            ),
            valid_roi=(
                float(self.get_parameter("roi_x_min").value),
                float(self.get_parameter("roi_y_min").value),
                float(self.get_parameter("roi_x_max").value),
                float(self.get_parameter("roi_y_max").value),
            ),
        )
        share = get_package_share_directory("tag_state_estimation")
        markers = np.loadtxt(os.path.join(share, "markers.csv"), delimiter=",")
        self.corner_tracker = Detector(markers[4:], ai_mode="off")
        self.hole_margin_m = max(
            0.0, float(self.get_parameter("hole_margin_m").value)
        )
        self.hole_rejection_delay_sec = max(
            0.0, float(self.get_parameter("hole_rejection_delay_sec").value)
        )
        self.hole_rejector = TimedHoleRejector(self.hole_rejection_delay_sec)
        self.bridge = CvBridge()
        self.process_every_n = max(1, int(self.get_parameter("process_every_n").value))
        self.frame_count = 0
        self.last_centers_xy = None
        self.last_radii_px = None

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        topic = str(self.get_parameter("camera_topic").value)
        self.create_subscription(Image, topic, self._on_image, qos)
        self.get_logger().info(
            f"Passive AI hole-mask viewer started; camera={topic}; "
            "no publishers or motor/control interfaces."
        )

    def _on_image(self, message):
        self.frame_count += 1
        if self.frame_count % self.process_every_n:
            return
        frame = self.bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
        detection = self.ai.detect(frame)
        corners_rc = self.corner_tracker.detect_corners(frame)
        markers_current = not self.corner_tracker.corners_missing
        if markers_current:
            centers_xy, radii_px = project_holes_to_image(
                corners_rc, margin_m=self.hole_margin_m
            )
            if centers_xy is not None:
                self.last_centers_xy = centers_xy
                self.last_radii_px = radii_px

        display = frame.copy()
        if self.last_centers_xy is not None:
            red_mask = display.copy()
            for index, (center_xy, radius_px) in enumerate(
                zip(self.last_centers_xy, self.last_radii_px), start=1
            ):
                center = tuple(np.round(center_xy).astype(int))
                radius = max(3, int(round(float(radius_px))))
                cv2.circle(red_mask, center, radius, (0, 0, 255), -1)
                cv2.circle(display, center, radius, (0, 0, 255), 2)
                cv2.putText(
                    display,
                    str(index),
                    (center[0] - 5, center[1] + 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
            cv2.addWeighted(red_mask, 0.35, display, 0.65, 0.0, display)

        rejected_hole = None
        hole_candidate = None
        hole_elapsed_sec = 0.0
        if detection.visible:
            hole_candidate = candidate_hole_index(
                np.asarray([detection.y_px, detection.x_px], dtype=np.float32),
                corners_rc if markers_current else np.full((4, 2), np.nan),
                margin_m=self.hole_margin_m,
            )
            reject_hole, hole_elapsed_sec = self.hole_rejector.update(hole_candidate)
            rejected_hole = hole_candidate if reject_hole else None
            center = (round(detection.x_px), round(detection.y_px))
            if rejected_hole is not None:
                color = (0, 0, 255)
            elif hole_candidate is not None:
                color = (0, 255, 255)
            else:
                color = (0, 255, 0)
            cv2.drawMarker(display, center, color, cv2.MARKER_CROSS, 24, 2)
            cv2.circle(display, center, 11, color, 2)
        else:
            self.hole_rejector.update(None)

        if rejected_hole is not None:
            status = (
                f"RAW AI conf={detection.confidence:.3f} -> "
                f"HOLE {rejected_hole + 1} REJECTED"
            )
            color = (0, 0, 255)
        elif hole_candidate is not None:
            status = (
                f"RAW AI OVER HOLE {hole_candidate + 1}: "
                f"{hole_elapsed_sec:.1f}/{self.hole_rejection_delay_sec:.1f}s "
                "PENDING - ACCEPTED"
            )
            color = (0, 255, 255)
        elif detection.visible:
            status = f"RAW AI conf={detection.confidence:.3f} -> MARBLE CANDIDATE"
            color = (0, 255, 0)
        else:
            status = f"NO AI CANDIDATE conf={detection.confidence:.3f}"
            color = (0, 0, 255)
        marker_status = "markers=current" if markers_current else "markers=LAST VALID"
        cv2.putText(
            display,
            status,
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            color,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            display,
            f"red = forbidden hole mask | {marker_status} | Q = quit",
            (10, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (0, 0, 255) if not markers_current else (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.imshow("AI Hole Mask Validator - Passive", display)
        if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
            rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = AiHoleMaskViewer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    cv2.destroyAllWindows()
    if rclpy.ok():
        node.destroy_node()
        rclpy.shutdown()
