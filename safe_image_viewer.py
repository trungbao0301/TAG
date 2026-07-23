#!/usr/bin/env python3

import argparse
import time

import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy


class SafeImageViewer(Node):
    def __init__(self, topic, display_hz, scale, reliability):
        super().__init__("safe_tag_image_viewer")

        self.bridge = CvBridge()
        self.topic = topic
        self.display_hz = display_hz
        self.scale = scale

        self.last_show_time = 0.0
        self.min_dt = 1.0 / display_hz if display_hz > 0 else 0.0
        self.frame_count = 0

        reliability_policy = (
            ReliabilityPolicy.RELIABLE
            if reliability == "reliable"
            else ReliabilityPolicy.BEST_EFFORT
        )
        qos = QoSProfile(
            reliability=reliability_policy,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.sub = self.create_subscription(
            Image,
            self.topic,
            self.image_callback,
            qos,
        )

        self.get_logger().info(f"Safe viewer subscribed to: {self.topic}")
        self.get_logger().info("This viewer only subscribes. It does NOT open the camera device.")
        self.get_logger().info("Press q in the image window to quit.")

    def image_callback(self, msg):
        now = time.time()

        # Throttle display so viewer does not waste CPU during training
        if now - self.last_show_time < self.min_dt:
            return

        self.last_show_time = now
        self.frame_count += 1

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"cv_bridge error: {e}")
            return

        if self.scale != 1.5:
            frame = cv2.resize(
                frame,
                None,
                fx=self.scale,
                fy=self.scale,
                interpolation=cv2.INTER_AREA,
            )


        cv2.imshow("Safe Tag Camera Viewer", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            self.get_logger().info("Quit requested.")
            rclpy.shutdown()


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--topic",
        default="/tag_camera/image",
        help="ROS image topic to view",
    )
    parser.add_argument(
        "--display_hz",
        type=float,
        default=60.0,
        help="Max display FPS. Lower value uses less CPU.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.75,
        help="Display scale. Example: 1.0 full size, 0.5 half size.",
    )
    parser.add_argument(
        "--reliability",
        choices=("best_effort", "reliable"),
        default="best_effort",
        help="ROS QoS reliability for the image subscription.",
    )

    parsed_args, ros_args = parser.parse_known_args()

    rclpy.init(args=ros_args)
    node = SafeImageViewer(
        topic=parsed_args.topic,
        display_hz=parsed_args.display_hz,
        scale=parsed_args.scale,
        reliability=parsed_args.reliability,
    )

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    cv2.destroyAllWindows()

    if rclpy.ok():
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
