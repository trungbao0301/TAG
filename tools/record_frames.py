#!/usr/bin/env python3
"""Record camera frames to disk for offline calibration.

Deliberately dumb: subscribe, write PNG, nothing else. No detection, no
imshow. The live calibrator was unusable on this rig because the camera
publishes ~18 Hz with sub-second stalls, so anything interactive feels laggy
regardless of how cheap it is. Recording decouples the two -- tilt the plate
while this writes files, then calibrate offline at full speed.

PNG, not JPEG: hole centres are refined to ~0.1 px and JPEG ringing around the
high-contrast hole edges would bias exactly that measurement.

    python3 tools/record_frames.py --seconds 90 --out /tmp/calib_frames
"""

import argparse
import os
import sys

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image


class Recorder(Node):
    def __init__(self, args):
        super().__init__("cyberrunner_frame_recorder")
        self.args = args
        self.bridge = CvBridge()
        self.count = 0
        self.skipped = 0
        self.started = None
        os.makedirs(args.out, exist_ok=True)
        self.done = False
        # RELIABLE, matching the publisher. These frames are 768 KB, which DDS
        # fragments over many UDP datagrams; with BEST_EFFORT one lost fragment
        # discards the whole sample and there is no retransmission. Measured on
        # this rig: BEST_EFFORT depth=1 receives 21.0 Hz, RELIABLE depth=1
        # receives 44.3 Hz from the same publisher.
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.create_subscription(Image, args.topic, self.on_image, qos)
        self.get_logger().info(
            f"recording every {args.every}th frame to {args.out} "
            f"for {args.seconds:.0f}s -- tilt the plate through its full range"
        )

    def on_image(self, message):
        stamp = float(message.header.stamp.sec) + message.header.stamp.nanosec * 1e-9
        if self.started is None:
            self.started = stamp
        self.skipped += 1
        if self.skipped % max(1, self.args.every) == 0:
            frame = self.bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
            path = os.path.join(self.args.out, "frame_%05d.png" % self.count)
            cv2.imwrite(path, frame)
            self.count += 1
            if self.count % 10 == 0:
                self.get_logger().info(
                    "%d frames (%.0fs)" % (self.count, stamp - self.started)
                )
        if stamp - self.started >= self.args.seconds:
            # Flag rather than rclpy.shutdown() here: shutting down from inside
            # a callback while spin() is running can hang the process.
            self.done = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topic", default="/cyberrunner_camera/image")
    parser.add_argument("--out", default="/tmp/calib_frames")
    parser.add_argument("--seconds", type=float, default=90.0)
    parser.add_argument(
        "--every", type=int, default=3,
        help="Keep every Nth frame. At ~18 Hz, 3 gives ~6 Hz, plenty of poses."
    )
    args, ros_args = parser.parse_known_args()
    rclpy.init(args=ros_args)
    node = Recorder(args)
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.2)
    except KeyboardInterrupt:
        pass
    print("recorded %d frames to %s" % (node.count, args.out))
    if rclpy.ok():
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
