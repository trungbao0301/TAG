#!/usr/bin/env python3
"""Live camera color tuner for the See3CAM (subscribe-only view + v4l2 sliders).

Shows the ROS camera feed (what the state-estimation pipeline actually sees) and
lets you adjust the camera's V4L2 color controls live with trackbars. Each slider
change is pushed to the device with `v4l2-ctl`, so you see the effect immediately.

This does NOT open the camera device (the publisher owns it) -- it only sets
controls on it, which is allowed while the device is streaming.

Goal: lock white balance + exposure and tune saturation so the marble's blue is
STABLE across every maze position. Then re-run marble_hsv_picker.py to re-tune the
HSV range for the new, stable color.

Run:  python3 camera_tuner_live.py            (device defaults to /dev/video2)
Keys: q/ESC quit,  s print current settings,  r reset camera to defaults
"""

import argparse
import subprocess

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

VIEW = "Camera Tuner (live ROS feed)"
CTRL = "Camera Controls"


def _noop(_):
    pass


class CameraTuner(Node):
    # name -> (v4l2 ctrl, tb_min, tb_max, tb_default, offset)
    # trackbar value used = slider + offset  (OpenCV sliders start at 0)
    CONTROLS = [
        ("auto_wb",     "white_balance_automatic",   0, 1, 0, 0),
        ("wb_temp",     "white_balance_temperature", 1000, 10000, 4500, 0),
        ("auto_exp",    "auto_exposure",             0, 3, 0, 0),   # 0=Auto,1=Manual
        ("exposure",    "exposure_time_absolute",    1, 2000, 312, 0),
        ("saturation",  "saturation",                0, 40, 32, 0),
        ("gamma",       "gamma",                     40, 500, 220, 0),
        ("contrast",    "contrast",                  0, 30, 9, 0),
        ("brightness",  "brightness",                0, 30, 15, -15),  # real -15..15
        ("powerline",   "power_line_frequency",      0, 2, 2, 0),   # 2 = 60 Hz
    ]

    def __init__(self, device, topic, reliability, scale):
        super().__init__("camera_tuner_live")
        self.device = device
        self.scale = scale
        self.bridge = CvBridge()
        self.frame = None
        self._last = {}

        qos = QoSProfile(
            reliability=(ReliabilityPolicy.RELIABLE if reliability == "reliable"
                         else ReliabilityPolicy.BEST_EFFORT),
            history=HistoryPolicy.KEEP_LAST, depth=1,
        )
        self.create_subscription(Image, topic, self._on_image, qos)

        cv2.namedWindow(CTRL)
        for name, ctrl, lo, hi, dflt, off in self.CONTROLS:
            # Initialise each slider to the camera's CURRENT value so launching
            # the tuner changes nothing until you actually drag something.
            cur = self._get(ctrl)
            start = (cur if cur is not None else dflt)
            start = max(lo, min(hi, start))
            cv2.createTrackbar(name, CTRL, start - off, hi - off, _noop)
            self._last[name] = start   # seed so nothing is pushed until moved
        self.get_logger().info(f"Tuning {self.device}. Drag sliders; watch the feed.")

    def _get(self, ctrl):
        try:
            out = subprocess.run(
                ["v4l2-ctl", "-d", self.device, f"--get-ctrl={ctrl}"],
                check=False, capture_output=True, text=True,
            ).stdout.strip()
            # format: "ctrl_name: 123"  or menu: "ctrl_name: 1 (Manual Mode)"
            return int(out.split(":")[-1].strip().split()[0])
        except Exception:
            return None

    def _set(self, ctrl, value):
        try:
            subprocess.run(
                ["v4l2-ctl", "-d", self.device, f"--set-ctrl={ctrl}={value}"],
                check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        except FileNotFoundError:
            self.get_logger().error("v4l2-ctl not found (apt install v4l-utils)")

    def _apply_all(self, force=False):
        vals = {}
        for name, ctrl, lo, hi, dflt, off in self.CONTROLS:
            slider = cv2.getTrackbarPos(name, CTRL)
            val = slider + off
            val = max(lo, min(hi, val))
            vals[name] = (ctrl, val)
        # Order matters: turn auto OFF before setting the dependent manual value.
        order = ["auto_wb", "auto_exp", "powerline", "wb_temp", "exposure",
                 "saturation", "gamma", "contrast", "brightness"]
        for name in order:
            ctrl, val = vals[name]
            if force or self._last.get(name) != val:
                self._set(ctrl, val)
                self._last[name] = val

    def _on_image(self, msg):
        try:
            self.frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"cv_bridge error: {e}")

    def _print_settings(self):
        print("\n=== current camera settings (paste to persist) ===")
        for name, ctrl, lo, hi, dflt, off in self.CONTROLS:
            val = cv2.getTrackbarPos(name, CTRL) + off
            print(f"v4l2-ctl -d {self.device} --set-ctrl={ctrl}={val}")
        print()

    def _reset(self):
        for name, ctrl, lo, hi, dflt, off in self.CONTROLS:
            cv2.setTrackbarPos(name, CTRL, dflt - off)
        self._apply_all(force=True)
        self.get_logger().info("Reset trackbars to defaults and pushed to camera.")

    def render(self):
        self._apply_all()
        if self.frame is None:
            # No camera frame yet -- still paint a placeholder so the windows
            # appear and waitKey runs (OpenCV needs waitKey to draw anything).
            disp = np.zeros((200, 640, 3), dtype=np.uint8)
            cv2.putText(disp, "waiting for camera frames...", (20, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        else:
            disp = self.frame
            if self.scale != 1.0:
                disp = cv2.resize(disp, None, fx=self.scale, fy=self.scale,
                                  interpolation=cv2.INTER_AREA)
            disp = disp.copy()
            cv2.putText(disp, "drag sliders -> live | s=print  r=reset  q=quit",
                        (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2,
                        cv2.LINE_AA)
        cv2.imshow(VIEW, disp)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            rclpy.shutdown()
        elif key == ord("s"):
            self._print_settings()
        elif key == ord("r"):
            self._reset()


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="/dev/v4l/by-id/usb-e-con_systems_See3CAM_24CUG_0F2D140416020900-video-index0")
    parser.add_argument("--topic", default="/cyberrunner_camera/image")
    parser.add_argument("--reliability", choices=("best_effort", "reliable"),
                        default="best_effort")
    parser.add_argument("--scale", type=float, default=1.75)
    parsed, ros_args = parser.parse_known_args()

    rclpy.init(args=ros_args)
    node = CameraTuner(parsed.device, parsed.topic, parsed.reliability, parsed.scale)
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
