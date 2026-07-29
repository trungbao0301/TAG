#!/usr/bin/env python3
"""Grab a few live camera frames (headless) and save them for analysis."""
import time
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

OUT = "/tmp/claude-1000/-home-trungbao-CYBER-cyberruner-main/9dd1b22e-5fbf-444c-90d8-6462684b9bed/scratchpad"


class Grab(Node):
    def __init__(self):
        super().__init__("marble_frame_grabber")
        self.br = CvBridge()
        self.frames = []
        self.last_save = 0.0
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST, depth=1)
        self.create_subscription(Image, "/cyberrunner_camera/image", self.cb, qos)

    def cb(self, msg):
        now = time.time()
        if now - self.last_save < 0.4:
            return
        self.last_save = now
        try:
            f = self.br.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(str(e)); return
        i = len(self.frames)
        cv2.imwrite(f"{OUT}/marble_frame_{i}.png", f)
        self.frames.append(f)
        self.get_logger().info(f"saved frame {i}  shape={f.shape}")
        if len(self.frames) >= 6:
            rclpy.shutdown()


def main():
    rclpy.init()
    n = Grab()
    try:
        rclpy.spin(n)
    except Exception:
        pass


if __name__ == "__main__":
    main()
