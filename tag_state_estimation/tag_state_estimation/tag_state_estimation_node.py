#!/usr/bin/env python3

import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from tag_state_estimation.core.estimation_pipeline import EstimationPipeline
from tag_state_estimation.core.opencv_acceleration import (
    configure_opencv_acceleration,
)
from tag_interfaces.msg import StateEstimate


class ImageSubscriber(Node):
    def __init__(self):
        super().__init__("tag_state_estimation")

        # Parameters
        self.declare_parameter("process_every_n", 1)   # 1 = every frame, 2 = every other frame
        self.declare_parameter("debug_timing", True)
        self.declare_parameter("pipeline_fps", 55.0)
        self.declare_parameter("show_image", False)
        self.declare_parameter("print_measurements", False)
        self.declare_parameter("use_gpu", False)
        self.declare_parameter("gpu_backend", "auto")
        self.declare_parameter("gpu_device_id", 0)
        self.declare_parameter("require_gpu", False)

        self.process_every_n = max(1, int(self.get_parameter("process_every_n").value))
        self.debug_timing = bool(self.get_parameter("debug_timing").value)
        self.pipeline_fps = float(self.get_parameter("pipeline_fps").value)
        self.show_image = bool(self.get_parameter("show_image").value)
        self.print_measurements = bool(self.get_parameter("print_measurements").value)
        self.use_gpu = bool(self.get_parameter("use_gpu").value)
        self.gpu_backend = str(self.get_parameter("gpu_backend").value)
        self.gpu_device_id = int(self.get_parameter("gpu_device_id").value)
        self.require_gpu = bool(self.get_parameter("require_gpu").value)

        self.acceleration_backend, acceleration_msg = configure_opencv_acceleration(
            self.use_gpu,
            self.gpu_backend,
            self.gpu_device_id,
            self.require_gpu,
        )
        self.get_logger().info(acceleration_msg)

        self.frame_count = 0
        self.last_log_time = time.monotonic()
        self.processed_count = 0
        self.received_count = 0

        self.br = CvBridge()

        self.subscription = self.create_subscription(
            Image,
            "tag_camera/image",
            self.listener_callback,
            1,
        )

        self.publisher_ = self.create_publisher(
            StateEstimate,
            "stateEstimation",
            1,
        )

        self.estimation_pipeline = EstimationPipeline(
            fps=self.pipeline_fps,
            estimator="FiniteDiff",
            FiniteDiff_mean_steps=4,
            print_measurements=self.print_measurements,
            show_image=self.show_image,
            do_anim_3d=False,
            viewpoint="top",
            show_subimages_detector=False,
            acceleration_backend=self.acceleration_backend,
        )

        self.get_logger().info("State estimation node initialized.")
        self.get_logger().info(f"process_every_n = {self.process_every_n}")
        self.get_logger().info(f"pipeline_fps = {self.pipeline_fps}")
        self.get_logger().info(f"print_measurements = {self.print_measurements}")
        self.get_logger().info(f"show_image = {self.show_image}")
        self.get_logger().info(f"acceleration_backend = {self.acceleration_backend}")

    def listener_callback(self, data):
        t0 = time.monotonic()

        self.received_count += 1
        self.frame_count += 1

        # Skip frames if estimation is too heavy
        if self.frame_count % self.process_every_n != 0:
            return

        try:
            frame = self.br.imgmsg_to_cv2(data, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().warn(f"cv_bridge conversion failed: {e}")
            return

        t_convert = time.monotonic()

        try:
            x_hat, P, angles, _, _ = self.estimation_pipeline.estimate(frame)
        except Exception as e:
            self.get_logger().warn(f"Estimation failed: {e}", throttle_duration_sec=1.0)
            return

        t_estimate = time.monotonic()

        msg = StateEstimate()
        msg.x_b = float(x_hat[0])
        msg.y_b = float(x_hat[1])
        msg.x_b_dot = float(x_hat[2])
        msg.y_b_dot = float(x_hat[3])
        msg.alpha = float(-angles[1])
        msg.beta = float(angles[0])

        self.publisher_.publish(msg)

        t_publish = time.monotonic()

        self.processed_count += 1

        if self.debug_timing:
            now = time.monotonic()
            if now - self.last_log_time >= 1.0:
                received_fps = self.received_count / (now - self.last_log_time)
                processed_fps = self.processed_count / (now - self.last_log_time)

                self.get_logger().info(
                    f"received_fps={received_fps:.1f}, "
                    f"processed_fps={processed_fps:.1f}, "
                    f"convert={(t_convert - t0) * 1000.0:.2f} ms, "
                    f"estimate={(t_estimate - t_convert) * 1000.0:.2f} ms, "
                    f"publish={(t_publish - t_estimate) * 1000.0:.2f} ms, "
                    f"total={(t_publish - t0) * 1000.0:.2f} ms"
                )

                self.received_count = 0
                self.processed_count = 0
                self.last_log_time = now


def main(args=None):
    rclpy.init(args=args)

    node = ImageSubscriber()

    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
