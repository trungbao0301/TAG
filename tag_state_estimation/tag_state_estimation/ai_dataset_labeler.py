"""Passive ROS2 camera labeler for the TAG marble dataset."""

import csv
from datetime import datetime, timezone
from pathlib import Path

import cv2
from cv_bridge import CvBridge
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image


WINDOW = "AI Marble Dataset Labeler"


class MarbleDatasetLabeler(Node):
    def __init__(self):
        super().__init__("ai_marble_dataset_labeler")
        self.declare_parameter("camera_topic", "/tag_camera/image")
        self.declare_parameter("output_dir", "ai_marble_dataset")
        self.declare_parameter("jpeg_quality", 95)
        self.camera_topic = str(self.get_parameter("camera_topic").value)
        self.output_dir = Path(str(self.get_parameter("output_dir").value)).expanduser()
        self.jpeg_quality = int(self.get_parameter("jpeg_quality").value)
        self.images_dir = self.output_dir / "images"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.labels_path = self.output_dir / "labels.csv"
        if not self.labels_path.exists():
            with self.labels_path.open("w", newline="") as handle:
                csv.writer(handle).writerow(
                    ["filename", "visible", "x_px", "y_px", "utc_timestamp"]
                )

        self.bridge = CvBridge()
        self.frame = None
        self.frozen = None
        self.counter = sum(1 for _ in self.images_dir.glob("frame_*.jpg"))
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.create_subscription(Image, self.camera_topic, self._on_image, qos)
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(WINDOW, self._on_mouse)
        self.get_logger().info(
            "Passive labeler ready. SPACE freezes/unfreezes; left-click labels the "
            "marble; N saves 'not visible'; Q quits. It creates no application "
            "publishers."
        )

    def _on_image(self, message):
        if self.frozen is not None:
            return
        self.frame = self.bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")

    def _active_frame(self):
        return self.frozen if self.frozen is not None else self.frame

    def _save(self, visible, x_px="", y_px=""):
        frame = self._active_frame()
        if frame is None:
            return
        filename = f"frame_{self.counter:06d}.jpg"
        path = self.images_dir / filename
        cv2.imwrite(
            str(path),
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality],
        )
        timestamp = datetime.now(timezone.utc).isoformat()
        with self.labels_path.open("a", newline="") as handle:
            csv.writer(handle).writerow([filename, int(visible), x_px, y_px, timestamp])
        self.counter += 1
        self.frozen = None
        label = f"({x_px:.1f}, {y_px:.1f})" if visible else "not visible"
        self.get_logger().info(f"Saved {filename}: {label}")

    def _on_mouse(self, event, x, y, _flags, _data):
        if event == cv2.EVENT_LBUTTONDOWN:
            self._save(True, float(x), float(y))

    def render(self):
        frame = self._active_frame()
        if frame is None:
            return
        display = frame.copy()
        status = "FROZEN" if self.frozen is not None else "LIVE"
        cv2.putText(
            display,
            f"{status} | labels={self.counter} | click marble | N=not visible",
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow(WINDOW, display)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            rclpy.shutdown()
        elif key == ord(" "):
            self.frozen = None if self.frozen is not None else frame.copy()
        elif key == ord("n"):
            self._save(False)


def main(args=None):
    rclpy.init(args=args)
    node = MarbleDatasetLabeler()
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.01)
            node.render()
    except KeyboardInterrupt:
        pass
    cv2.destroyAllWindows()
    if rclpy.ok():
        node.destroy_node()
        rclpy.shutdown()
