#!/usr/bin/env python3

import base64
import json
import os
import socket
import sys
import threading
import time
import numpy as np

import rclpy
from rclpy.node import Node

from tag_interfaces.msg import HiwonderVel, StateEstimateSub
from tag_interfaces.srv import HiwonderReset


def recv_line(sock):
    data = bytearray()
    while True:
        b = sock.recv(1)
        if not b:
            raise ConnectionError("Server disconnected")
        if b == b"\n":
            return data.decode("utf-8")
        data.extend(b)


def send_json(sock, obj):
    sock.sendall(json.dumps(obj, separators=(",", ":")).encode("utf-8") + b"\n")


def imgmsg_to_gray64(msg):
    data = np.frombuffer(msg.data, dtype=np.uint8)

    if msg.height == 0 or msg.width == 0:
        return np.zeros((64, 64, 1), dtype=np.uint8)

    channels = max(1, int(msg.step // msg.width))
    arr = data.reshape(msg.height, msg.step)
    arr = arr[:, : msg.width * channels]

    if channels == 1:
        img = arr.reshape(msg.height, msg.width)
    else:
        img = arr.reshape(msg.height, msg.width, channels)
        img = img.mean(axis=2).astype(np.uint8)

    if img.shape != (64, 64):
        import cv2
        img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)

    return img.reshape(64, 64, 1).astype(np.uint8)


class TcpRosBridge(Node):
    def __init__(self, host, port):
        super().__init__("tcp_ros_bridge")
 
        self.host = host
        self.port = int(port)
        self.latest = None
        self.last_state_time = 0.0
        self.latest_lock = threading.Lock()
        self.running = True
        self.reset_on_ball_lost = os.environ.get(
            # env_tcp owns the occlusion grace period and requests a reset only
            # after a loss is confirmed. Keep the bridge fallback opt-in so it
            # cannot reset the board during a temporary camera occlusion.
            "TAG_RESET_ON_BALL_LOST", "0"
        ).lower() not in ("0", "false", "no")
        self.ball_lost_threshold = int(
            os.environ.get("TAG_BALL_LOST_RESET_FRAMES", "15")
        )
        self.reset_cooldown_sec = float(
            os.environ.get("TAG_BALL_LOST_RESET_COOLDOWN", "0.0")
        )
        self.max_cmd_1 = float(os.environ.get("TAG_MAX_CMD_1", "180"))
        self.max_cmd_2 = float(os.environ.get("TAG_MAX_CMD_2", "180"))
        self.ball_lost_count = 0
        self.ball_seen_count = 0
        self.last_reset_time = 0.0
        self.reset_done_for_current_loss = False

        self.sub = self.create_subscription(
            StateEstimateSub,
            "/tag_state_estimation/estimate_subimg",
            self.on_state,
            1,
        )

        self.pub = self.create_publisher(
            HiwonderVel,
            "/tag_hiwonder/cmd",
            1,
        )
        self.reset_client = self.create_client(
            HiwonderReset,
            "/tag_hiwonder/reset",
        )

        self.get_logger().info(f"TCP ROS bridge will connect to {host}:{port}")
        self.get_logger().info(
            "Ball-lost board reset is "
            f"{'enabled' if self.reset_on_ball_lost else 'disabled'} "
            f"(frames={self.ball_lost_threshold}, "
            f"cooldown={self.reset_cooldown_sec:.1f}s)"
        )
        self.get_logger().info(
            f"TCP command clamp: vel_1=[-{self.max_cmd_1}, {self.max_cmd_1}], "
            f"vel_2=[-{self.max_cmd_2}, {self.max_cmd_2}]"
        )

        self.spin_thread = threading.Thread(target=self.spin_loop, daemon=True)
        self.spin_thread.start()

    def spin_loop(self):
        while rclpy.ok() and self.running:
            rclpy.spin_once(self, timeout_sec=0.01)

    def on_state(self, msg):
        try:
            img = imgmsg_to_gray64(msg.subimg)

            x_b = float(msg.state.x_b)
            y_b = float(msg.state.y_b)

            ball_ok = not (np.isnan(x_b) or np.isnan(y_b))
            self.maybe_reset_on_ball_lost(ball_ok)

            latest = {
                "ok": True,
                "ball": ball_ok,
                "alpha": float(msg.state.alpha),
                "beta": float(msg.state.beta),
                "x_b": x_b,
                "y_b": y_b,
                "image_b64": base64.b64encode(img.tobytes()).decode("ascii"),
            }

            with self.latest_lock:
                self.latest = latest
                self.last_state_time = time.time()

        except Exception as e:
            self.get_logger().error(f"Failed to convert state: {e}")

    def maybe_reset_on_ball_lost(self, ball_ok):

        if ball_ok:

            self.ball_seen_count += 1

            if self.ball_seen_count >= 30:

                if self.reset_done_for_current_loss:
                    self.get_logger().info(
                        "Ball detected consistently again. Reset re-enabled."
                    )

                self.ball_lost_count = 0
                self.reset_done_for_current_loss = False

            return

        self.ball_seen_count = 0

        if not self.reset_on_ball_lost:
            return

        if self.reset_done_for_current_loss:
            return

        self.ball_lost_count += 1

        now = time.time()

        if self.ball_lost_count < self.ball_lost_threshold:
            return

        if now - self.last_reset_time < self.reset_cooldown_sec:
            return

        self.get_logger().warn(
            "Ball missing; resetting board to balance position."
        )

        self.reset_done_for_current_loss = True

        self.reset_board()

    def reset_board(self):

        now = time.time()

        # Prevent multiple resets from any source
        if now - self.last_reset_time < self.reset_cooldown_sec:
            self.get_logger().info(
                f"Ignoring reset request ({now - self.last_reset_time:.2f}s since last reset)"
            )
            return True

        self.last_reset_time = now

        self.publish_action(0.0, 0.0)

        if not self.reset_client.service_is_ready():
            self.get_logger().warn(
                "/tag_hiwonder/reset is not ready; sent zero velocity only."
            )
            return False

        req = HiwonderReset.Request()
        req.max_temp = 256

        future = self.reset_client.call_async(req)
        future.add_done_callback(self.on_reset_done)

        return True

    def on_reset_done(self, future):
        try:
            result = future.result()
            if result is None or not result.success:
                self.get_logger().warn("Board reset service returned failure.")
            else:
                self.get_logger().info("Board reset service completed.")
        except Exception as exc:
            self.get_logger().warn(f"Board reset service failed: {exc}")

    def get_latest(self):
        with self.latest_lock:
            if self.latest is None:
                return None
            return dict(self.latest)

    def get_latest_time(self):
        with self.latest_lock:
            return self.last_state_time

    def wait_for_latest_after(self, previous_time, timeout_sec=0.018):
        deadline = time.time() + max(0.0, float(timeout_sec))
        latest = self.get_latest()

        while time.time() < deadline:
            with self.latest_lock:
                if self.latest is not None:
                    latest = dict(self.latest)
                    if self.last_state_time > previous_time:
                        return latest
            time.sleep(0.001)

        return latest

    def publish_action(self, vel_1, vel_2):
        # Safety guard: Dreamer or TCP can sometimes send NaN/inf.
        # Hiwonder node cannot convert NaN to servo position, so replace with zero.
        try:
            vel_1 = float(vel_1)
            vel_2 = float(vel_2)
        except Exception:
            vel_1, vel_2 = 0.0, 0.0

        if not np.isfinite(vel_1):
            self.get_logger().warn(f"vel_1 was non-finite: {vel_1}; replacing with 0.0")
            vel_1 = 0.0

        if not np.isfinite(vel_2):
            self.get_logger().warn(f"vel_2 was non-finite: {vel_2}; replacing with 0.0")
            vel_2 = 0.0

        vel_1 = float(np.clip(vel_1, -self.max_cmd_1, self.max_cmd_1))
        vel_2 = float(np.clip(vel_2, -self.max_cmd_2, self.max_cmd_2))

        msg = HiwonderVel()
        msg.vel_1 = vel_1
        msg.vel_2 = vel_2
        self.pub.publish(msg)

    def connect_loop(self):
        while rclpy.ok():
            try:
                self.get_logger().info(f"Connecting to server {self.host}:{self.port} ...")
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                sock.connect((self.host, self.port))
                self.get_logger().info("Connected to TCP training server.")
                return sock
            except Exception as e:
                self.get_logger().warn(f"Connection failed: {e}. Retrying in 1 sec...")
                time.sleep(1.0)

    def serve_forever(self):
        sock = self.connect_loop()

        while rclpy.ok():
            try:
                line = recv_line(sock)
                req = json.loads(line)
                cmd = req.get("cmd", "")

                if cmd == "obs":
                    latest = self.get_latest()
                    if latest is None:
                        send_json(sock, {"ok": False, "error": "no state yet"})
                    else:
                        send_json(sock, latest)

                elif cmd == "action":
                    self.publish_action(req.get("vel_1", 0.0), req.get("vel_2", 0.0))
                    send_json(sock, {"ok": True})

                elif cmd == "step":
                    previous_time = self.get_latest_time()
                    self.publish_action(req.get("vel_1", 0.0), req.get("vel_2", 0.0))
                    latest = self.wait_for_latest_after(
                        previous_time,
                        timeout_sec=req.get("timeout", 0.018),
                    )
                    if latest is None:
                        send_json(sock, {"ok": False, "error": "no state yet"})
                    else:
                        send_json(sock, latest)

                elif cmd == "reset":
                    latest = self.get_latest()
                    ball_missing = latest is not None and not latest.get("ball", False)
                    if ball_missing and self.reset_done_for_current_loss:
                        self.publish_action(0.0, 0.0)
                        self.get_logger().info(
                            "Ignoring duplicate reset while marble is still missing."
                        )
                        send_json(sock, {"ok": True})
                    else:
                        if ball_missing:
                            self.reset_done_for_current_loss = True
                        ok = self.reset_board()
                        send_json(sock, {"ok": ok})

                else:
                    send_json(sock, {"ok": False, "error": f"unknown cmd {cmd}"})

            except Exception as e:
                self.get_logger().error(f"TCP error: {e}")
                try:
                    sock.close()
                except Exception:
                    pass
                sock = self.connect_loop()

    def destroy_node(self):
        self.running = False
        if hasattr(self, "spin_thread"):
            self.spin_thread.join(timeout=1.0)
        super().destroy_node()


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 tcp_ros_bridge.py SERVER_IP [PORT]")
        sys.exit(1)

    host = sys.argv[1]
    port = int(sys.argv[2]) if len(sys.argv) >= 3 else 5555

    rclpy.init()
    node = TcpRosBridge(host, port)

    try:
        node.serve_forever()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
