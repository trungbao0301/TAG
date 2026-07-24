"""ROS2 ONNX marble detector with isolated diagnostic outputs."""

import math
from pathlib import Path

import cv2
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Float32

from tag_state_estimation.ai_marble_common import OnnxMarbleDetector


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
        if detection.visible:
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
                cv2.circle(frame, center, 10, color, 2)
                cv2.drawMarker(frame, center, color, cv2.MARKER_CROSS, 20, 2)
            cv2.putText(
                frame,
                f"AI confidence={detection.confidence:.3f} misses={self.misses}",
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
