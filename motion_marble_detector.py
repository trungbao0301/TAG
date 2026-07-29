#!/usr/bin/env python3
"""Experimental MOTION-based marble detector (subscribe-only, live tuning).

The marble is the only thing that moves inside the board, so we difference the
current frame against a slowly-adapting background: the moving marble lights up
regardless of its (reflective, unstable) color. Robust to the blue arm/cables
and corner dots, which are static.

Pipeline per frame:
  gray -> abs-diff vs adaptive background -> threshold -> morphology
  -> contours -> pick blob of marble size nearest the last position
  -> update background everywhere EXCEPT at the marble (so a slow/stopped
     marble isn't absorbed) -> hysteresis holds last pos briefly when it stops.

Overlay: yellow cross = tracked marble; green circles = other motion blobs.
Windows: detector view, motion mask, Controls (trackbars).
Keys: q/ESC quit, r reset background, m toggle mask, SPACE pause.

Run:  python3 motion_marble_detector.py
"""
import argparse
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

VIEW = "Motion Marble Detector"
MASK = "Motion Mask"
CTRL = "Controls"


def _noop(_):
    pass


class MotionDetector(Node):
    def __init__(self, topic, reliability, scale):
        super().__init__("motion_marble_detector")
        self.bridge = CvBridge()
        self.scale = scale
        self.frame = None
        self.bg = None                 # float32 adaptive background (gray)
        self.last_pos = None           # last accepted marble (x,y) px
        self.miss = 0
        self.show_mask = True
        self.paused = False

        qos = QoSProfile(
            reliability=(ReliabilityPolicy.RELIABLE if reliability == "reliable"
                         else ReliabilityPolicy.BEST_EFFORT),
            history=HistoryPolicy.KEEP_LAST, depth=1)
        self.create_subscription(Image, topic, self._on_image, qos)

        cv2.namedWindow(CTRL)
        cv2.createTrackbar("bg_adapt_x1000", CTRL, 20, 200, _noop)   # 0.020 default
        cv2.createTrackbar("motion_thresh", CTRL, 18, 100, _noop)
        cv2.createTrackbar("min_area", CTRL, 40, 2000, _noop)
        cv2.createTrackbar("max_area", CTRL, 600, 5000, _noop)
        cv2.createTrackbar("min_circ_x100", CTRL, 30, 100, _noop)
        cv2.createTrackbar("miss_hold", CTRL, 6, 30, _noop)
        cv2.createTrackbar("gate_px", CTRL, 60, 300, _noop)          # max jump from last pos
        self.get_logger().info(f"Subscribed {topic} (motion detector, subscribe-only).")

    def _on_image(self, msg):
        if self.paused:
            return
        try:
            self.frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(str(e))

    def _detect(self, gray):
        a = cv2.getTrackbarPos("bg_adapt_x1000", CTRL) / 1000.0
        thr = cv2.getTrackbarPos("motion_thresh", CTRL)
        amin = cv2.getTrackbarPos("min_area", CTRL)
        amax = max(amin + 1, cv2.getTrackbarPos("max_area", CTRL))
        cmin = cv2.getTrackbarPos("min_circ_x100", CTRL) / 100.0
        hold = cv2.getTrackbarPos("miss_hold", CTRL)
        gate = cv2.getTrackbarPos("gate_px", CTRL)

        g = cv2.GaussianBlur(gray, (0, 0), 1.2).astype(np.float32)
        if self.bg is None:
            self.bg = g.copy()
            return None, np.zeros_like(gray), []

        diff = cv2.absdiff(g, self.bg)
        mask = (diff > thr).astype(np.uint8) * 255
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)

        cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        cands = []
        for c in cnts:
            area = cv2.contourArea(c)
            if not (amin < area < amax):
                continue
            per = cv2.arcLength(c, True)
            circ = (4 * np.pi * area) / (per * per) if per else 0
            if circ < cmin:
                continue
            (x, y), r = cv2.minEnclosingCircle(c)
            cands.append((int(x), int(y), int(r), area, circ))

        # choose: nearest to last position (if tracking), else most circular
        marble = None
        if cands:
            if self.last_pos is not None:
                lx, ly = self.last_pos
                near = [c for c in cands
                        if (c[0]-lx)**2 + (c[1]-ly)**2 <= gate*gate]
                pool = near if near else (cands if self.miss >= hold else [])
                if pool:
                    marble = min(pool, key=lambda c: (c[0]-lx)**2 + (c[1]-ly)**2)
            else:
                marble = max(cands, key=lambda c: c[4])  # most circular

        # update tracking + hysteresis
        if marble is not None:
            self.last_pos = (marble[0], marble[1])
            self.miss = 0
        else:
            self.miss += 1

        # adapt background everywhere; freeze a disc around the marble so a
        # slow/stopped marble is not absorbed into the background.
        upd = np.full(gray.shape, a, np.float32)
        if self.last_pos is not None and self.miss <= hold:
            cv2.circle(upd, self.last_pos, 14, 0.0, -1)
        self.bg = (1 - upd) * self.bg + upd * g
        return marble, mask, cands

    def render(self):
        if self.frame is None:
            return
        gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        marble, mask, cands = self._detect(gray)
        disp = self.frame.copy()
        for (x, y, r, area, circ) in cands:
            cv2.circle(disp, (x, y), max(r, 5), (0, 180, 0), 1)
        hold = cv2.getTrackbarPos("miss_hold", CTRL)
        if marble is not None:
            x, y, r, area, circ = marble
            cv2.drawMarker(disp, (x, y), (0, 255, 255), cv2.MARKER_CROSS, 20, 2)
            cv2.circle(disp, (x, y), max(r, 6), (0, 255, 255), 2)
            cv2.putText(disp, f"MARBLE a={int(area)} circ={circ:.2f}", (8, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
        elif self.last_pos is not None and self.miss <= hold:
            cv2.drawMarker(disp, self.last_pos, (0, 165, 255), cv2.MARKER_TILTED_CROSS, 18, 2)
            cv2.putText(disp, f"holding ({self.miss})", (8, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2, cv2.LINE_AA)
        else:
            cv2.putText(disp, "no marble", (8, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

        if self.scale != 1.0:
            disp = cv2.resize(disp, None, fx=self.scale, fy=self.scale)
        cv2.imshow(VIEW, disp)
        if self.show_mask:
            cv2.imshow(MASK, mask)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            rclpy.shutdown()
        elif key == ord("r"):
            self.bg = None; self.last_pos = None; self.miss = 0
            self.get_logger().info("background reset")
        elif key == ord("m"):
            self.show_mask = not self.show_mask
            if not self.show_mask:
                cv2.destroyWindow(MASK)
        elif key == ord(" "):
            self.paused = not self.paused


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--topic", default="/cyberrunner_camera/image")
    p.add_argument("--reliability", choices=("best_effort", "reliable"), default="best_effort")
    p.add_argument("--scale", type=float, default=1.75)
    a, ros = p.parse_known_args()
    rclpy.init(args=ros)
    n = MotionDetector(a.topic, a.reliability, a.scale)
    try:
        while rclpy.ok():
            rclpy.spin_once(n, timeout_sec=0.01)
            n.render()
    except KeyboardInterrupt:
        pass
    cv2.destroyAllWindows()
    if rclpy.ok():
        n.destroy_node(); rclpy.shutdown()


if __name__ == "__main__":
    main()
