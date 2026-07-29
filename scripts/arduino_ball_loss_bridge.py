#!/usr/bin/env python3

import argparse
import glob
import math
import os
import sys
import threading
import time

import rclpy
from rclpy.node import Node

from tag_interfaces.msg import StateEstimate
from tag_interfaces.srv import HiwonderReset

try:
    import serial
except ImportError as exc:
    raise SystemExit(
        "Missing pyserial. Install with: python3 -m pip install pyserial"
    ) from exc


class ArduinoBallLossBridge(Node):
    def __init__(self, args):
        super().__init__("arduino_ball_loss_bridge")
        self.args = args
        self.ser = None
        self.connect_serial()

        self.detected_count = 0
        self.lost_since = None
        self.last_run_time = 0.0
        self.last_stop_time = 0.0
        self.auto_enabled = True

        self.create_subscription(StateEstimate, args.topic, self.cb, 10)
        self.reset_client = self.create_client(HiwonderReset, args.reset_service)
        self.send(args.stop_cmd)
        self.get_logger().info(
            f"Arduino bridge on {args.port} watching {args.topic}. "
            "finite x_b/y_b = ball detected; continuously missing for "
            f"{args.lost_seconds:.1f}s sends '{args.run_cmd}'."
        )

    def connect_serial(self):
        if self.ser is not None:
            try:
                self.ser.close()
            except serial.SerialException:
                pass
        port = resolve_port(self.args.port)
        self.ser = serial.Serial(port, self.args.baud, timeout=0.05)
        time.sleep(self.args.serial_warmup)
        self.get_logger().info(f"Connected Arduino serial: {port}")

    def send(self, cmd):
        data = (cmd.strip() + "\n").encode("utf-8")
        for attempt in range(2):
            try:
                self.ser.write(data)
                self.ser.flush()
                self.get_logger().info(f"Arduino <- {cmd}")
                # Log Arduino response for diagnostics
                try:
                    reply = self.ser.readline().decode("utf-8", errors="ignore").strip()
                    if reply:
                        self.get_logger().info(f"Arduino -> {reply}")
                except (OSError, serial.SerialException):
                    pass
                return True
            except (OSError, serial.SerialException) as exc:
                self.get_logger().warn(
                    f"Serial write failed on attempt {attempt + 1}: {exc}"
                )
                time.sleep(0.5)
                try:
                    self.connect_serial()
                except (OSError, serial.SerialException) as reconnect_exc:
                    self.get_logger().warn(f"Serial reconnect failed: {reconnect_exc}")
                    time.sleep(1.0)
        return False

    def reset_board(self):
        if not self.args.reset_with_test:
            return
        if not self.reset_client.service_is_ready():
            self.get_logger().warn(
                f"Reset service not ready: {self.args.reset_service}"
            )
            return
        req = HiwonderReset.Request()
        req.max_temp = self.args.reset_max_temp
        self.reset_client.call_async(req)
        self.get_logger().info(
            f"Reset board -> {self.args.reset_service} max_temp={req.max_temp}"
        )

    def start_test_mode(self):
        if not self.test_active:
            self.reset_board()
        self.send(self.args.test_cmd)
        self.test_active = True

    def clear_stuck_state(self):
        self.stuck_anchor = None
        self.stuck_since = None
        self.stuck_test_active = False

    def update_stuck_state(self, x_b, y_b, now):
        pos = (float(x_b), float(y_b))
        if self.stuck_anchor is None:
            self.stuck_anchor = pos
            self.stuck_since = now
            return False

        dist = math.hypot(pos[0] - self.stuck_anchor[0], pos[1] - self.stuck_anchor[1])
        if dist > self.args.stuck_radius_m:
            if self.stuck_test_active:
                self.send(self.args.stop_cmd)
                self.last_stop_time = now
                self.test_active = False
            self.stuck_anchor = pos
            self.stuck_since = now
            self.stuck_test_active = False
            return False

        if self.stuck_since is None:
            self.stuck_since = now
            return False

        if now - self.stuck_since < self.args.stuck_seconds:
            return False

        if now - self.last_test_time >= self.args.repeat_seconds:
            self.get_logger().warn(
                "Marble appears stuck; resetting board and starting test "
                f"(still within {dist:.4f} m for {now - self.stuck_since:.1f}s)."
            )
            self.start_test_mode()
            self.last_test_time = now
        self.stuck_test_active = True
        return True

    def cb(self, msg):
        if not self.auto_enabled:
            return

        detected = math.isfinite(msg.x_b) and math.isfinite(msg.y_b)
        now = time.monotonic()

        if detected:
            self.detected_count += 1
            if self.detected_count < self.args.detected_frames:
                return
            self.lost_since = None
            self.last_run_time = 0.0
            if now - self.last_stop_time >= self.args.stop_every:
                self.send(self.args.stop_cmd)
                self.last_stop_time = now
            return

        self.detected_count = 0
        if self.lost_since is None:
            self.lost_since = now
            return

        if now - self.lost_since < self.args.lost_seconds:
            return

        if self.last_run_time and now - self.last_run_time < self.args.repeat_seconds:
            return

        self.send(self.args.run_cmd)
        self.last_run_time = now

    def stdin_loop(self):
        print("Manual commands: fire, test, run, stop, status, auto on, auto off, quit")
        for line in sys.stdin:
            cmd = line.strip().lower()
            if not cmd:
                continue
            if cmd == "fire":
                self.send(self.args.fire_cmd)
            elif cmd == "test":
                self.start_test_mode()
            elif cmd == "run":
                self.send(self.args.run_cmd)
            elif cmd == "stop":
                self.send(self.args.stop_cmd)
                self.last_run_time = 0.0
                self.last_test_time = 0.0
                self.test_active = False
                self.clear_stuck_state()
                self.lost_since = None
            elif cmd == "status":
                self.send(self.args.status_cmd)
            elif cmd == "auto on":
                self.auto_enabled = True
                print("AUTO ENABLED")
            elif cmd == "auto off":
                self.auto_enabled = False
                print("AUTO DISABLED")
            elif cmd in ("quit", "exit"):
                rclpy.shutdown()
                return
            else:
                print("Unknown command. Use: fire, test, run, stop, status, auto on, auto off, quit")


def resolve_port(port):
    if port and port != "auto":
        return port

    by_id = sorted(glob.glob("/dev/serial/by-id/*"))
    preferred_tokens = ("1a86_USB_Serial", "CH340", "CH341", "Arduino")
    excluded_tokens = ("FTDI", "Dynamixel")

    for path in by_id:
        name = os.path.basename(path)
        if any(token in name for token in excluded_tokens):
            continue
        if any(token in name for token in preferred_tokens):
            return path

    candidates = []
    for path in sorted(glob.glob("/dev/ttyUSB*") + glob.glob("/dev/ttyACM*")):
        real = os.path.realpath(path)
        is_dynamixel = False
        for stable in by_id:
            name = os.path.basename(stable)
            if any(token in name for token in excluded_tokens) and os.path.realpath(stable) == real:
                is_dynamixel = True
                break
        if not is_dynamixel:
            candidates.append(path)

    if candidates:
        return candidates[0]

    raise SystemExit("Could not find Arduino serial port. Plug it in or pass --port manually.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--port",
        default="auto",
        help="Arduino serial port, or 'auto' to avoid the FTDI/Dynamixel port.",
    )
    parser.add_argument("--baud", type=int, default=9600)
    parser.add_argument("--topic", default="/tag_state_estimation/estimate")
    parser.add_argument("--lost_seconds", type=float, default=3.0)
    parser.add_argument("--repeat_seconds", type=float, default=5.0)
    parser.add_argument("--detected_frames", type=int, default=5)
    parser.add_argument("--stop_every", type=float, default=2.0)
    parser.add_argument("--serial_warmup", type=float, default=2.0)
    parser.add_argument("--run_cmd", default="run")
    parser.add_argument("--stop_cmd", default="stop")
    parser.add_argument("--fire_cmd", default="fire")
    parser.add_argument("--test_cmd", default="test")
    parser.add_argument("--status_cmd", default="status")
    parser.add_argument("--reset_service", default="tag_hiwonder/reset")
    parser.add_argument("--reset_max_temp", type=int, default=256)
    parser.add_argument("--reset_with_test", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    rclpy.init()
    node = ArduinoBallLossBridge(args)
    threading.Thread(target=node.stdin_loop, daemon=True).start()
    try:
        rclpy.spin(node)
    finally:
        try:
            node.send(args.stop_cmd)
        except (OSError, serial.SerialException):
            pass
        if node.ser is not None:
            node.ser.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
