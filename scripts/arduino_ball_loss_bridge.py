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
from rclpy.executors import ExternalShutdownException

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
        self.missing_reply_count = 0
        self.connect_serial()

        self.detected_count = 0
        self.lost_since = None
        self.last_run_time = 0.0
        self.last_stop_time = 0.0
        self.auto_enabled = True

        self.subscription = None
        if not args.monitor_only:
            self.subscription = self.create_subscription(
                StateEstimate, args.topic, self.cb, 10
            )
        self.reset_client = self.create_client(HiwonderReset, args.reset_service)
        if not args.monitor_only:
            self.send(args.stop_cmd)
        self.status_timer = None
        if args.status_every > 0:
            self.status_timer = self.create_timer(
                args.status_every, self.poll_status
            )
        if args.monitor_only:
            self.get_logger().info(
                f"Arduino monitor on {args.port}; autonomous firmware owns "
                f"solenoid control. Lux status interval: {args.status_every:.1f}s."
            )
        else:
            self.get_logger().info(
                f"Arduino bridge on {args.port} watching {args.topic}. "
                "finite x_b/y_b = ball detected; continuously missing for "
                f"{args.lost_seconds:.1f}s sends '{args.run_cmd}'. "
                f"Lux status interval: {args.status_every:.1f}s."
            )

    def connect_serial(self):
        if self.ser is not None:
            try:
                self.ser.close()
            except Exception:
                pass
        port = resolve_port(self.args.port)
        self.ser = serial.Serial(
            port, self.args.baud, timeout=self.args.serial_reply_timeout
        )
        time.sleep(self.args.serial_warmup)
        self.ser.reset_input_buffer()
        self.get_logger().info(f"Connected Arduino serial: {port}")

    @staticmethod
    def expected_reply(cmd):
        cmd = cmd.strip().lower()
        if cmd in ("run", "fire", "test"):
            return "ok pulse"
        if cmd == "stop":
            return "ok stop"
        if cmd == "status":
            return "ok lux="
        return "ok"

    def recover_serial(self, reason):
        self.get_logger().warn(f"Serial watchdog reconnect: {reason}")
        try:
            self.connect_serial()
            self.ser.write(b"status\n")
            self.ser.flush()
            reply = self.ser.readline().decode(
                "utf-8", errors="ignore"
            ).strip()
            if reply.startswith("ok lux="):
                self.get_logger().info(f"Arduino watchdog status -> {reply}")
                self.missing_reply_count = 0
                return True
            self.get_logger().warn(
                "Arduino watchdog status had no valid reply"
                + (f": {reply}" if reply else "")
            )
        except (Exception, SystemExit) as exc:
            self.get_logger().warn(f"Serial watchdog reconnect failed: {exc}")
        return False

    def send(self, cmd):
        data = (cmd.strip() + "\n").encode("utf-8")
        if self.ser is None or not self.ser.is_open:
            if not self.recover_serial("port was closed"):
                return False
        try:
            # A late reply must not be mistaken for this command's ACK.
            self.ser.reset_input_buffer()
            self.ser.write(data)
            self.ser.flush()
            self.get_logger().info(f"Arduino <- {cmd}")
            reply = self.ser.readline().decode(
                "utf-8", errors="ignore"
            ).strip()
            expected = self.expected_reply(cmd)
            if reply:
                self.get_logger().info(f"Arduino -> {reply}")
            if reply.startswith(expected):
                self.missing_reply_count = 0
                return True

            self.missing_reply_count += 1
            self.get_logger().warn(
                f"Arduino ACK missing or invalid for '{cmd}' "
                f"({self.missing_reply_count}/{self.args.max_missing_replies})"
            )
            if self.missing_reply_count >= self.args.max_missing_replies:
                # Reopen and probe only. Do not resend an actuating command here:
                # the Arduino may have pulsed even if its ACK was lost.
                self.recover_serial(
                    f"{self.missing_reply_count} consecutive missing replies"
                )
            return False
        except Exception as exc:
            # Reopen and probe, but leave command retry to the normal callback
            # schedule to avoid a possible double solenoid pulse.
            self.get_logger().warn(f"Serial write failed: {exc}")
            try:
                self.recover_serial("write/read exception")
            except Exception as reconnect_exc:
                self.get_logger().warn(
                    f"Serial reconnect failed unexpectedly: {reconnect_exc}"
                )
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

    def poll_status(self):
        """Log Arduino lux and auto state without opening a second serial client."""
        self.send(self.args.status_cmd)

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
    parser.add_argument("--serial_reply_timeout", type=float, default=0.25)
    parser.add_argument("--max_missing_replies", type=int, default=3)
    parser.add_argument(
        "--monitor_only",
        action="store_true",
        help="Poll status only; do not control the solenoid from state estimation.",
    )
    parser.add_argument(
        "--status_every",
        type=float,
        default=1.0,
        help="Seconds between Arduino lux/status log lines; 0 disables polling.",
    )
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
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        try:
            if not args.monitor_only:
                node.send(args.stop_cmd)
        except Exception:
            pass
        if node.ser is not None:
            node.ser.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
