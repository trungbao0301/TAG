#!/usr/bin/env python3

import time
import subprocess
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy


class FastCameraPublisher(Node):
    def __init__(self):
        super().__init__("tag_camera")

        self.declare_parameter("device", "/dev/v4l/by-id/usb-e-con_systems_See3CAM_24CUG_0F2D140416020900-video-index0")
        self.declare_parameter("fps", 60.0)
        self.declare_parameter("width", 1280)
        self.declare_parameter("height", 720)
        self.declare_parameter("output_width", 640)
        self.declare_parameter("output_height", 360)
        self.declare_parameter("border_y", 20)
        self.declare_parameter("fourcc", "MJPG")
        self.declare_parameter("exposure", -1)  # -1 = auto, >0 = manual (100µs units, e.g. 150 = 15ms)

        # Locked color controls (tuned via camera_tuner_live.py) applied with
        # v4l2-ctl after the device opens, so the marble's blue is STABLE across
        # every maze position. -999 means "leave the camera default / skip".
        self.declare_parameter("v4l2_white_balance_automatic", 0)
        self.declare_parameter("v4l2_white_balance_temperature", 5477)
        # 1 = Manual (locks exposure_time_absolute -> stable frame rate). Do NOT
        # use 3 (aperture priority): the camera then auto-picks a long exposure in
        # this lighting (~55ms), which caps the frame rate at ~18fps and starves
        # the RL training of data. Manual 8ms exposure gives ~45fps.
        self.declare_parameter("v4l2_auto_exposure", 1)
        self.declare_parameter("v4l2_exposure_time_absolute", 80)
        self.declare_parameter("v4l2_saturation", 40)
        self.declare_parameter("v4l2_gamma", 193)
        self.declare_parameter("v4l2_contrast", 30)
        self.declare_parameter("v4l2_brightness", -6)
        self.declare_parameter("v4l2_power_line_frequency", 0)

        self.device = self.get_parameter("device").value
        self.fps = float(self.get_parameter("fps").value)
        self.width = int(self.get_parameter("width").value)
        self.height = int(self.get_parameter("height").value)
        self.output_width = int(self.get_parameter("output_width").value)
        self.output_height = int(self.get_parameter("output_height").value)
        self.border_y = int(self.get_parameter("border_y").value)
        self.fourcc = str(self.get_parameter("fourcc").value)
        self.exposure = int(self.get_parameter("exposure").value)

        self.bridge = CvBridge()

        # Best QoS for camera streaming:
        # Keep only newest frame, do not block waiting for old frames.
        image_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.pub = self.create_publisher(
            Image,
            "/tag_camera/image",
            image_qos
        )

        self.cap = cv2.VideoCapture(self.device, cv2.CAP_V4L2)

        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera device: {self.device}")

        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.fourcc))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if self.exposure > 0:
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # 1 = manual in V4L2
            self.cap.set(cv2.CAP_PROP_EXPOSURE, self.exposure)
            self.get_logger().info(f"Exposure: manual ({self.exposure})")
        else:
            self.get_logger().info("Exposure: auto")

        self.get_logger().info(f"Camera opened: {self.device}")
        self.get_logger().info(f"Requested FOURCC: {self.fourcc}")
        self.get_logger().info(f"Actual width: {self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)}")
        self.get_logger().info(f"Actual height: {self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
        self.get_logger().info(f"Actual FPS setting: {self.cap.get(cv2.CAP_PROP_FPS)}")
        self.get_logger().info(
            f"Publishing output: {self.output_width}x{self.output_height + 2 * self.border_y}"
        )

        self._apply_v4l2_color_controls()

        self.frame_count = 0
        self.last_report_time = time.time()

    def _apply_v4l2_color_controls(self):
        """Push the locked color controls with v4l2-ctl (after cv2 config so
        these win). Order matters: disable the 'auto' before its manual value."""
        gp = lambda n: int(self.get_parameter(n).value)
        # (v4l2 control name, parameter name) in dependency-safe order
        controls = [
            ("white_balance_automatic", "v4l2_white_balance_automatic"),
            ("auto_exposure", "v4l2_auto_exposure"),
            ("power_line_frequency", "v4l2_power_line_frequency"),
            ("white_balance_temperature", "v4l2_white_balance_temperature"),
            ("exposure_time_absolute", "v4l2_exposure_time_absolute"),
            ("saturation", "v4l2_saturation"),
            ("gamma", "v4l2_gamma"),
            ("contrast", "v4l2_contrast"),
            ("brightness", "v4l2_brightness"),
        ]
        applied = []
        for ctrl, param in controls:
            val = gp(param)
            if val == -999:
                continue
            try:
                subprocess.run(
                    ["v4l2-ctl", "-d", self.device, f"--set-ctrl={ctrl}={val}"],
                    check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
                applied.append(f"{ctrl}={val}")
            except FileNotFoundError:
                self.get_logger().warn("v4l2-ctl not found; color controls not applied.")
                return
        self.get_logger().info("Applied locked color controls: " + ", ".join(applied))

    def run(self):
        while rclpy.ok():
            ok, frame = self.cap.read()

            if not ok or frame is None:
                self.get_logger().warn("Failed to read camera frame")
                continue

            # Resize 1280x720 -> 640x360
            frame = cv2.resize(
                frame,
                (self.output_width, self.output_height),
                interpolation=cv2.INTER_AREA
            )

            # Add top/bottom border: 640x360 -> 640x400
            if self.border_y > 0:
                frame = cv2.copyMakeBorder(
                    frame,
                    self.border_y,
                    self.border_y,
                    0,
                    0,
                    cv2.BORDER_CONSTANT,
                    value=(0, 0, 0)
                )

            msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = "camera"

            self.pub.publish(msg)

            self.frame_count += 1
            now = time.time()

            if now - self.last_report_time >= 2.0:
                fps_now = self.frame_count / (now - self.last_report_time)
                self.get_logger().info(f"Publish FPS: {fps_now:.1f}")
                self.frame_count = 0
                self.last_report_time = now

        self.cleanup()

    def cleanup(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None


def main(args=None):
    rclpy.init(args=args)
    node = None

    try:
        node = FastCameraPublisher()
        node.run()

    except KeyboardInterrupt:
        pass

    except Exception as e:
        print(f"[ERROR] {e}")

    finally:
        if node is not None:
            node.cleanup()
            node.destroy_node()

        # Prevent "rcl_shutdown already called" crash
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()