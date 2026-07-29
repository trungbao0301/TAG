#!/usr/bin/env python3
"""Experimental GRAYSCALE + size marble detector (subscribe-only).

Tests the idea: "the marble is smaller than the holes, so separate them by size
in grayscale." Runs alongside the real pipeline without disturbing it -- it only
subscribes to the camera topic and shows what a grayscale detector would find.

Method per frame:
  1. gray = BGR2GRAY
  2. candidate mask = (black_cut < gray < dark_max)
        - excludes the near-black HOLES (darker than black_cut)
        - excludes the bright board (brighter than dark_max)
        - keeps the medium-gray marble
  3. morphological open/close to clean speckle
  4. contours -> for each blob compute area, circularity, equivalent radius
  5. ACCEPT if  min_area < area < max_area  and  circularity > min_circ
        - max_area rejects hole-sized / merged blobs (the size idea)
  6. pick the most circular accepted blob as the marble

Overlay colors:
  green  = accepted marble candidate (+area)
  red    = rejected: too big  (hole / marble-merged-with-hole)
  gray   = rejected: too small (noise) or not circular enough

Trackbars let you tune live. Keys: q/ESC quit, m toggle mask, SPACE pause.

Run:  python3 grayscale_marble_detector.py
      (add --topic / --reliability if needed)
"""

import argparse

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

WINDOW = "Grayscale Marble Detector"
MASK_WINDOW = "Candidate Mask"
CTRL = "Controls"


def _noop(_):
    pass


class GrayMarbleDetector(Node):
    def __init__(self, topic, reliability, scale):
        super().__init__("grayscale_marble_detector")
        self.bridge = CvBridge()
        self.scale = scale
        self.frame = None
        self.show_mask = True
        self.paused = False

        qos = QoSProfile(
            reliability=(ReliabilityPolicy.RELIABLE
                         if reliability == "reliable"
                         else ReliabilityPolicy.BEST_EFFORT),
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.sub = self.create_subscription(Image, topic, self._on_image, qos)

        cv2.namedWindow(CTRL)
        # defaults tuned for 640x400 view (~2.4 px/mm): marble d~12mm~29px.
        cv2.createTrackbar("black_cut", CTRL, 25, 255, _noop)   # below = hole
        cv2.createTrackbar("dark_max", CTRL, 95, 255, _noop)    # above = board
        cv2.createTrackbar("min_area", CTRL, 150, 3000, _noop)
        cv2.createTrackbar("max_area", CTRL, 850, 5000, _noop)  # above = hole
        cv2.createTrackbar("min_circ_x100", CTRL, 45, 100, _noop)
        cv2.createTrackbar("blur", CTRL, 1, 5, _noop)

        self.get_logger().info(f"Subscribed to {topic} (subscribe-only, grayscale test).")

    def _on_image(self, msg):
        if self.paused:
            return
        try:
            self.frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"cv_bridge error: {e}")

    def _detect(self, gray):
        black_cut = cv2.getTrackbarPos("black_cut", CTRL)
        dark_max = max(black_cut + 1, cv2.getTrackbarPos("dark_max", CTRL))
        min_area = cv2.getTrackbarPos("min_area", CTRL)
        max_area = max(min_area + 1, cv2.getTrackbarPos("max_area", CTRL))
        min_circ = cv2.getTrackbarPos("min_circ_x100", CTRL) / 100.0
        blur = cv2.getTrackbarPos("blur", CTRL)

        g = cv2.GaussianBlur(gray, (0, 0), blur) if blur > 0 else gray
        mask = cv2.inRange(g, black_cut, dark_max)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)

        contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        accepted, rejected = [], []
        for c in contours:
            area = cv2.contourArea(c)
            if area < 10:
                continue
            perim = cv2.arcLength(c, True)
            circ = (4 * np.pi * area) / (perim ** 2) if perim else 0.0
            (x, y), rad = cv2.minEnclosingCircle(c)
            info = (int(x), int(y), int(rad), area, circ)
            if min_area < area < max_area and circ > min_circ:
                accepted.append(info)
            else:
                too_big = area >= max_area
                rejected.append((info, too_big))

        # marble = most circular accepted blob
        marble = max(accepted, key=lambda i: i[4]) if accepted else None
        return mask, accepted, rejected, marble

    def render(self):
        if self.frame is None:
            return
        gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        mask, accepted, rejected, marble = self._detect(gray)

        disp = self.frame.copy()
        for (x, y, r, area, circ), too_big in rejected:
            color = (0, 0, 255) if too_big else (140, 140, 140)
            cv2.circle(disp, (x, y), r, color, 1)
        for (x, y, r, area, circ) in accepted:
            cv2.circle(disp, (x, y), r, (0, 200, 0), 2)
            cv2.putText(disp, f"{int(area)}", (x + r, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 0), 1, cv2.LINE_AA)
        if marble is not None:
            x, y, r, area, circ = marble
            cv2.drawMarker(disp, (x, y), (0, 255, 255), cv2.MARKER_CROSS, 18, 2)
            cv2.putText(disp,
                        f"MARBLE a={int(area)} circ={circ:.2f}",
                        (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 255), 2, cv2.LINE_AA)
        else:
            cv2.putText(disp, "no marble", (8, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(disp,
                    f"accepted={len(accepted)}  rejected={len(rejected)} "
                    f"(red=too-big/hole)",
                    (8, disp.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 0), 1, cv2.LINE_AA)

        if self.scale != 1.0:
            disp = cv2.resize(disp, None, fx=self.scale, fy=self.scale,
                              interpolation=cv2.INTER_NEAREST)
        cv2.imshow(WINDOW, disp)
        if self.show_mask:
            cv2.imshow(MASK_WINDOW, mask)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            rclpy.shutdown()
        elif key == ord("m"):
            self.show_mask = not self.show_mask
            if not self.show_mask:
                cv2.destroyWindow(MASK_WINDOW)
        elif key == ord(" "):
            self.paused = not self.paused


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", default="/cyberrunner_camera/image")
    parser.add_argument("--reliability", choices=("best_effort", "reliable"),
                        default="reliable")
    parser.add_argument("--scale", type=float, default=1.75)
    parsed, ros_args = parser.parse_known_args()

    rclpy.init(args=ros_args)
    node = GrayMarbleDetector(parsed.topic, parsed.reliability, parsed.scale)
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


if __name__ == "__main__":
    main()
