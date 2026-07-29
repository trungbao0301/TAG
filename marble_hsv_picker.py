#!/usr/bin/env python3
"""Live HSV picker for tuning marble (ball) detection.

Subscribe-only, exactly like safe_image_viewer.py -- this never opens the
camera device. Click the marble in the video window to sample its HSV. Clicks
accumulate into a min/max range, a live mask preview shows what that range
selects, and the range is printed ready to paste into
cyberrunner_state_estimation/core/detection.py (DEFAULT_HSV_BALL).

Keys (focus the "Marble HSV Picker" window):
    left-click   sample an NxN patch under the cursor
    r            reset the accumulated range
    m            toggle the mask preview window
    p            print the range as a DEFAULT_HSV_BALL snippet
    q / ESC      quit
"""

import argparse

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

WINDOW = "Marble HSV Picker"
MASK_WINDOW = "Mask Preview"


class MarbleHsvPicker(Node):
    def __init__(self, topic, display_hz, scale, reliability, patch, margin):
        super().__init__("marble_hsv_picker")

        self.bridge = CvBridge()
        self.topic = topic
        self.scale = scale
        self.patch = max(1, patch)          # sample an NxN neighborhood per click
        self.margin = max(0, margin)        # padding added around sampled min/max
        self.min_dt = 1.0 / display_hz if display_hz > 0 else 0.0

        self.frame = None                   # latest original-resolution BGR frame
        self.hsv = None                     # its HSV conversion
        self.show_mask = True
        self.last_click = None              # (ox, oy) for the diagnose key

        # must match Detector.DEFAULT_SIZE_CROP_BALL / gaussian_robust ball gates
        self.crop_size = 80
        self.min_area = 30
        self.min_circularity = 0.08

        # accumulated HSV range across all sampled clicks
        self.h_lo = self.s_lo = self.v_lo = None
        self.h_hi = self.s_hi = self.v_hi = None

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
        self.sub = self.create_subscription(Image, self.topic, self._on_image, qos)

        cv2.namedWindow(WINDOW)
        cv2.setMouseCallback(WINDOW, self._on_mouse)

        self.get_logger().info(f"Subscribed to {self.topic} (subscribe-only).")
        self.get_logger().info(
            "Click the marble to sample. Keys: r reset, m mask, "
            "d diagnose, p print, q quit."
        )

    # ------------------------------------------------------------------ sampling
    def _on_mouse(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN or self.hsv is None:
            return

        # map click from the (scaled) display back to original pixel coords
        ox = int(round(x / self.scale))
        oy = int(round(y / self.scale))
        h, w = self.hsv.shape[:2]
        ox = min(w - 1, max(0, ox))
        oy = min(h - 1, max(0, oy))
        self.last_click = (ox, oy)

        half = self.patch // 2
        x0, x1 = max(0, ox - half), min(w, ox + half + 1)
        y0, y1 = max(0, oy - half), min(h, oy + half + 1)
        patch = self.hsv[y0:y1, x0:x1].reshape(-1, 3).astype(int)

        p_lo = patch.min(axis=0)
        p_hi = patch.max(axis=0)

        if self.h_lo is None:
            self.h_lo, self.s_lo, self.v_lo = p_lo
            self.h_hi, self.s_hi, self.v_hi = p_hi
        else:
            self.h_lo = min(self.h_lo, p_lo[0])
            self.s_lo = min(self.s_lo, p_lo[1])
            self.v_lo = min(self.v_lo, p_lo[2])
            self.h_hi = max(self.h_hi, p_hi[0])
            self.s_hi = max(self.s_hi, p_hi[1])
            self.v_hi = max(self.v_hi, p_hi[2])

        center = self.hsv[oy, ox]
        self.get_logger().info(
            f"click ({ox},{oy}) HSV={tuple(int(c) for c in center)}  "
            f"range now {self._range_str()}"
        )

    def _range_bounds(self):
        """Return padded (lo, hi) HSV arrays, clamped to valid OpenCV ranges."""
        m = self.margin
        lo = np.array(
            [
                max(0, self.h_lo - m),
                max(0, self.s_lo - m),
                max(0, self.v_lo - m),
            ]
        )
        hi = np.array(
            [
                min(179, self.h_hi + m),
                min(255, self.s_hi + m),
                min(255, self.v_hi + m),
            ]
        )
        return lo, hi

    def _range_str(self):
        if self.h_lo is None:
            return "(none)"
        lo, hi = self._range_bounds()
        return f"H[{lo[0]},{hi[0]}] S[{lo[1]},{hi[1]}] V[{lo[2]},{hi[2]}]"

    def _print_snippet(self):
        if self.h_lo is None:
            self.get_logger().warn("No samples yet -- click the marble first.")
            return
        lo, hi = self._range_bounds()
        print(
            "\n# paste into Detector.DEFAULT_HSV_BALL (detection.py)\n"
            "DEFAULT_HSV_BALL = (\n"
            f"    ({lo[0]}, {hi[0]}),  # (minHue, maxHue)\n"
            f"    ({lo[1]}, {hi[1]}),  # (minSat, maxSat)\n"
            f"    ({lo[2]}, {hi[2]}),  # (minVal, maxVal)\n"
            ")\n"
        )

    def _diagnose(self):
        """Replicate the detector's ball-blob logic on an 80x80 crop around the
        last click and report measured area + circularity vs the accept gates."""
        if self.h_lo is None or self.last_click is None or self.hsv is None:
            self.get_logger().warn(
                "Click the marble first, then press d to diagnose."
            )
            return

        ox, oy = self.last_click
        h, w = self.hsv.shape[:2]
        half = self.crop_size // 2
        x0, x1 = max(0, ox - half), min(w, ox + half)
        y0, y1 = max(0, oy - half), min(h, oy + half)
        crop = self.hsv[y0:y1, x0:x1]

        lo, hi = self._range_bounds()
        mask = cv2.inRange(crop, lo.astype(np.uint8), hi.astype(np.uint8))

        # identical morphology to gaussian_robust.detect_gaussian (ball branch)
        mask = cv2.erode(mask, np.ones((2, 2), np.uint8), iterations=1)
        mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=1)

        contours = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )[0]
        if not contours:
            self.get_logger().warn(
                "DIAGNOSE: no contour in crop -> marble would NOT be detected."
            )
            return

        contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter else 0.0
        passes = area > self.min_area and circularity > self.min_circularity

        print(
            "\n=== DIAGNOSE (largest blob in 80x80 crop) ===\n"
            f"  area        = {area:.1f}   (gate: > {self.min_area})\n"
            f"  circularity = {circularity:.3f}  (gate: > {self.min_circularity})\n"
            f"  RESULT      = {'PASS -> detected' if passes else 'FAIL -> NOT detected'}\n"
        )

    # -------------------------------------------------------------------- render
    def _on_image(self, msg):
        try:
            self.frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"cv_bridge error: {e}")
            return
        self.hsv = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)

    def render(self):
        if self.frame is None:
            return

        disp = self.frame
        if self.scale != 1.0:
            disp = cv2.resize(
                disp, None, fx=self.scale, fy=self.scale,
                interpolation=cv2.INTER_AREA,
            )

        cv2.putText(
            disp, self._range_str(), (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA,
        )
        cv2.imshow(WINDOW, disp)

        if self.show_mask and self.h_lo is not None:
            lo, hi = self._range_bounds()
            mask = cv2.inRange(self.hsv, lo.astype(np.uint8), hi.astype(np.uint8))
            preview = cv2.bitwise_and(self.frame, self.frame, mask=mask)
            if self.scale != 1.0:
                preview = cv2.resize(
                    preview, None, fx=self.scale, fy=self.scale,
                    interpolation=cv2.INTER_AREA,
                )
            cv2.imshow(MASK_WINDOW, preview)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            rclpy.shutdown()
        elif key == ord("r"):
            self.h_lo = self.s_lo = self.v_lo = None
            self.h_hi = self.s_hi = self.v_hi = None
            self.get_logger().info("Range reset.")
        elif key == ord("m"):
            self.show_mask = not self.show_mask
            if not self.show_mask:
                cv2.destroyWindow(MASK_WINDOW)
        elif key == ord("d"):
            self._diagnose()
        elif key == ord("p"):
            self._print_snippet()


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", default="/cyberrunner_camera/image")
    parser.add_argument("--display_hz", type=float, default=30.0)
    parser.add_argument("--scale", type=float, default=1.75)
    parser.add_argument(
        "--reliability", choices=("best_effort", "reliable"), default="best_effort"
    )
    parser.add_argument("--patch", type=int, default=5, help="NxN patch per click")
    parser.add_argument(
        "--margin", type=int, default=8, help="padding added around sampled HSV min/max"
    )
    parsed, ros_args = parser.parse_known_args()

    rclpy.init(args=ros_args)
    node = MarbleHsvPicker(
        topic=parsed.topic,
        display_hz=parsed.display_hz,
        scale=parsed.scale,
        reliability=parsed.reliability,
        patch=parsed.patch,
        margin=parsed.margin,
    )

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=node.min_dt or 0.01)
            node.render()
    except KeyboardInterrupt:
        pass

    cv2.destroyAllWindows()
    if rclpy.ok():
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
