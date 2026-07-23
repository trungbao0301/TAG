#!/usr/bin/env python3

import math
import time

import rclpy
from rclpy.node import Node

from tag_interfaces.msg import HiwonderVel
from tag_interfaces.srv import HiwonderReset

try:
    from vassar_feetech_servo_sdk import ServoController
except ImportError:
    ServoController = None

try:
    from scservo_sdk import PortHandler, sms_sts
except ImportError:
    PortHandler = None
    sms_sts = None


def _clamp(value, lower, upper):
    return max(lower, min(upper, value))


class FeetechVelNode(Node):
    def __init__(self):
        super().__init__("tag_feetech")

        self.declare_parameter("port", "/dev/ttyUSB0")
        self.declare_parameter("baudrate", 1000000)
        self.declare_parameter("servo_type", "sts")
        self.declare_parameter("servo1_id", 1)
        self.declare_parameter("servo2_id", 2)
        self.declare_parameter("scale_1", 1.0)
        self.declare_parameter("scale_2", 1.0)
        self.declare_parameter("max_speed_1", 3000)
        self.declare_parameter("max_speed_2", 3000)
        self.declare_parameter("acc_1", 10)
        self.declare_parameter("acc_2", 10)
        self.declare_parameter("enforce_position_limits", True)
        self.declare_parameter("position_limit_fail_stop", True)
        self.declare_parameter("position_limit_stop_outside", False)
        self.declare_parameter("position_limit_margin", 0)
        self.declare_parameter("position_limit_check_period_sec", 0.02)
        self.declare_parameter("positive_speed_increases_position_1", True)
        self.declare_parameter("positive_speed_increases_position_2", True)
        self.declare_parameter("min_position_1", 0)
        self.declare_parameter("max_position_1", 4095)
        self.declare_parameter("min_position_2", 0)
        self.declare_parameter("max_position_2", 4095)
        self.declare_parameter("torque_limit", 1000)
        self.declare_parameter("reset_position_1", 2048)
        self.declare_parameter("reset_position_2", 2048)
        self.declare_parameter("reset_speed", 600)
        self.declare_parameter("reset_acc", 10)
        self.declare_parameter("reset_wait_sec", 1.0)
        self.declare_parameter("reset_timeout_sec", 4.0)
        self.declare_parameter("reset_tolerance", 20)
        self.declare_parameter("command_timeout_sec", 1.0)

        self.port_name = self.get_parameter("port").value
        self.baudrate = int(self.get_parameter("baudrate").value)
        self.servo_type = str(self.get_parameter("servo_type").value).lower()
        self.servo_ids = (
            int(self.get_parameter("servo1_id").value),
            int(self.get_parameter("servo2_id").value),
        )
        self.scales = (
            float(self.get_parameter("scale_1").value),
            float(self.get_parameter("scale_2").value),
        )
        self.max_speeds = (
            int(self.get_parameter("max_speed_1").value),
            int(self.get_parameter("max_speed_2").value),
        )
        self.accs = (
            int(self.get_parameter("acc_1").value),
            int(self.get_parameter("acc_2").value),
        )
        self.enforce_position_limits = bool(
            self.get_parameter("enforce_position_limits").value
        )
        self.position_limit_fail_stop = bool(
            self.get_parameter("position_limit_fail_stop").value
        )
        self.position_limit_stop_outside = bool(
            self.get_parameter("position_limit_stop_outside").value
        )
        self.position_limit_margin = int(
            self.get_parameter("position_limit_margin").value
        )
        self.position_limit_check_period_sec = float(
            self.get_parameter("position_limit_check_period_sec").value
        )
        self.positive_speed_increases_position = (
            bool(self.get_parameter("positive_speed_increases_position_1").value),
            bool(self.get_parameter("positive_speed_increases_position_2").value),
        )
        self.position_limits = (
            (
                int(self.get_parameter("min_position_1").value),
                int(self.get_parameter("max_position_1").value),
            ),
            (
                int(self.get_parameter("min_position_2").value),
                int(self.get_parameter("max_position_2").value),
            ),
        )
        self.torque_limit = int(self.get_parameter("torque_limit").value)
        self.reset_positions = (
            int(self.get_parameter("reset_position_1").value),
            int(self.get_parameter("reset_position_2").value),
        )
        self.reset_speed = int(self.get_parameter("reset_speed").value)
        self.reset_acc = int(self.get_parameter("reset_acc").value)
        self.reset_wait_sec = float(self.get_parameter("reset_wait_sec").value)
        self.reset_timeout_sec = float(self.get_parameter("reset_timeout_sec").value)
        self.reset_tolerance = int(self.get_parameter("reset_tolerance").value)
        self.command_timeout_sec = float(
            self.get_parameter("command_timeout_sec").value
        )
        self.last_cmd_time = None
        self.timed_out = False
        self.resetting = False
        self.ignore_position_limits = False
        self.current_speeds = [0, 0]

        self.controller = None
        self.port = None
        self.packet = None
        self.connected = False
        self.connect()

        self.subscription = self.create_subscription(
            HiwonderVel,
            "tag_hiwonder/cmd",
            self.on_cmd,
            1,
        )
        self.service = self.create_service(
            HiwonderReset,
            "tag_hiwonder/reset",
            self.on_reset,
        )
        self.timeout_timer = self.create_timer(
            max(0.005, self.position_limit_check_period_sec),
            self.check_timeout,
        )

    def connect(self):
        if ServoController is not None:
            self.connect_vassar()
            return

        if PortHandler is None or sms_sts is None:
            self.get_logger().error(
                "Install vassar-feetech-servo-sdk or scservo_sdk before "
                "running this node."
            )
            return

        self.port = PortHandler(self.port_name)
        self.packet = sms_sts(self.port)

        if not self.port.openPort():
            self.get_logger().error(f"Failed to open Feetech port {self.port_name}")
            return
        if not self.port.setBaudRate(self.baudrate):
            self.get_logger().error(f"Failed to set Feetech baudrate {self.baudrate}")
            return

        self.connected = True
        self.get_logger().info(
            f"Opened Feetech port {self.port_name} at {self.baudrate}"
        )
        self.log_position_limits()
        for servo_id in self.servo_ids:
            self.enable_wheel_mode(servo_id)
        self.stop_servos()

    def connect_vassar(self):
        try:
            self.controller = ServoController(
                servo_ids=list(self.servo_ids),
                servo_type=self.servo_type,
                port=self.port_name,
                baudrate=self.baudrate,
            )
            self.controller.connect()
        except Exception as exc:
            self.get_logger().error(f"Failed to connect with Vassar Feetech SDK: {exc}")
            return

        self.port = self.controller.port_handler
        self.packet = self.controller.packet_handler
        self.connected = True
        self.get_logger().info(
            f"Opened Feetech {self.servo_type.upper()} via Vassar SDK on "
            f"{self.port_name} at {self.baudrate}"
        )
        self.log_position_limits()
        for servo_id in self.servo_ids:
            if not self.controller.set_operating_mode(servo_id, 1):
                self.get_logger().warn(
                    f"Could not set Feetech servo {servo_id} to speed mode"
                )
        self.stop_servos()

    def log_position_limits(self):
        if not self.enforce_position_limits:
            self.get_logger().info("Feetech position limits are disabled.")
            return
        self.get_logger().info(
            "Feetech position limits: "
            f"id {self.servo_ids[0]}=[{self.position_limits[0][0]}, "
            f"{self.position_limits[0][1]}], "
            f"id {self.servo_ids[1]}=[{self.position_limits[1][0]}, "
            f"{self.position_limits[1][1]}]"
        )
        self.get_logger().info(
            "Feetech limit watchdog: "
            f"period={self.position_limit_check_period_sec:.3f}s, "
            f"margin={self.position_limit_margin}"
        )
        self.get_logger().info(
            "Feetech limit direction: "
            f"id {self.servo_ids[0]} positive speed "
            f"{'increases' if self.positive_speed_increases_position[0] else 'decreases'} position, "
            f"id {self.servo_ids[1]} positive speed "
            f"{'increases' if self.positive_speed_increases_position[1] else 'decreases'} position"
        )

    def enable_wheel_mode(self, servo_id):
        # SMS/STS servos use wheel mode for continuous velocity commands.
        for method in ("unLockEprom", "WheelMode", "LockEprom"):
            func = getattr(self.packet, method, None)
            if func is None:
                continue
            try:
                func(servo_id)
            except Exception as exc:
                self.get_logger().warn(
                    f"Feetech {method} failed for servo {servo_id}: {exc}"
                )

    def command_to_speed(self, value, index):
        if value is None or not math.isfinite(float(value)):
            value = 0.0
        speed = int(round(float(value) * self.scales[index]))
        return _clamp(speed, -self.max_speeds[index], self.max_speeds[index])

    def write_speed(self, servo_id, speed, acc):
        if not self.connected:
            return False
        try:
            if self.servo_type == "hls" and hasattr(self.packet, "WriteSpec"):
                result = self.packet.WriteSpec(
                    servo_id, int(speed), int(acc), self.torque_limit
                )
            elif hasattr(self.packet, "WriteSpec"):
                result = self.packet.WriteSpec(servo_id, int(speed), int(acc))
            else:
                result = self.packet.WriteSpe(servo_id, int(speed), int(acc))
        except Exception as exc:
            self.get_logger().warn(f"Feetech speed write failed: {exc}")
            return False

        if isinstance(result, tuple):
            comm_result = result[0]
            return comm_result == 0
        return result == 0

    def read_position(self, servo_id):
        if self.controller is not None:
            return int(self.controller.read_position(servo_id))
        if self.packet is None:
            raise RuntimeError("Feetech packet handler is not connected")
        position, comm_result, error = self.packet.ReadPos(servo_id)
        if comm_result != 0 or error != 0:
            raise RuntimeError(
                f"ReadPos failed for servo {servo_id}: "
                f"comm_result={comm_result} error={error}"
            )
        return int(position)

    def apply_position_limits(self, speed_1, speed_2):
        if not self.enforce_position_limits or self.ignore_position_limits:
            return speed_1, speed_2

        speeds = [int(speed_1), int(speed_2)]
        for index, servo_id in enumerate(self.servo_ids):
            try:
                position = self.read_position(servo_id)
            except Exception as exc:
                self.get_logger().warn(
                    f"Could not read Feetech servo {servo_id} position: {exc}",
                    throttle_duration_sec=1.0,
                )
                if self.position_limit_fail_stop:
                    speeds[index] = 0
                continue

            min_pos, max_pos = self.position_limits[index]
            min_stop = min_pos + self.position_limit_margin
            max_stop = max_pos - self.position_limit_margin
            position_delta = speeds[index]
            if not self.positive_speed_increases_position[index]:
                position_delta = -position_delta

            if position <= min_stop and position_delta < 0:
                self.get_logger().warn(
                    f"Servo {servo_id} near/below min position {position} <= "
                    f"{min_stop}; blocking speed that lowers position.",
                    throttle_duration_sec=1.0,
                )
                speeds[index] = 0
            elif position >= max_stop and position_delta > 0:
                self.get_logger().warn(
                    f"Servo {servo_id} near/above max position {position} >= "
                    f"{max_stop}; blocking speed that raises position.",
                    throttle_duration_sec=1.0,
                )
                speeds[index] = 0

        return speeds[0], speeds[1]

    def write_pair_raw(self, speed_1, speed_2):
        ok_1 = self.write_speed(self.servo_ids[0], speed_1, self.accs[0])
        ok_2 = self.write_speed(self.servo_ids[1], speed_2, self.accs[1])
        self.current_speeds = [int(speed_1), int(speed_2)]
        self.get_logger().info(
            f"speed=({speed_1}, {speed_2}) ids={self.servo_ids}",
            throttle_duration_sec=2.0,
        )
        return ok_1 and ok_2

    def write_pair(self, speed_1, speed_2):
        speed_1, speed_2 = self.apply_position_limits(speed_1, speed_2)
        return self.write_pair_raw(speed_1, speed_2)

    def stop_servos(self):
        return self.write_pair(0, 0)

    def set_speed_mode(self):
        ok = True
        for servo_id in self.servo_ids:
            if self.controller is not None:
                if not self.controller.set_operating_mode(servo_id, 1):
                    ok = False
                    self.get_logger().warn(
                        f"Could not set Feetech servo {servo_id} to speed mode"
                    )
            else:
                self.enable_wheel_mode(servo_id)
        return ok

    def reset_to_position(self):
        if not self.connected:
            return False
        self.resetting = True
        self.ignore_position_limits = True

        positions = {
            self.servo_ids[0]: self.reset_positions[0],
            self.servo_ids[1]: self.reset_positions[1],
        }
        self.get_logger().info(
            f"Resetting Feetech servos to positions {positions} "
            f"speed={self.reset_speed} acc={self.reset_acc} "
            f"tolerance={self.reset_tolerance}"
        )

        try:
            if self.controller is not None:
                results = self.controller.write_position(
                    positions,
                    speed=self.reset_speed,
                    acceleration=self.reset_acc,
                )
                ok = all(results.get(servo_id, False) for servo_id in self.servo_ids)
            elif self.servo_type == "hls" and hasattr(self.packet, "WritePosEx"):
                ok = all(
                    self.packet.WritePosEx(
                        servo_id,
                        position,
                        self.reset_speed,
                        self.reset_acc,
                        self.torque_limit,
                    )[0]
                    == 0
                    for servo_id, position in positions.items()
                )
            else:
                ok = all(
                    self.packet.WritePosEx(
                        servo_id,
                        position,
                        self.reset_speed,
                        self.reset_acc,
                    )[0]
                    == 0
                    for servo_id, position in positions.items()
                )
        except Exception as exc:
            self.get_logger().warn(f"Feetech reset position write failed: {exc}")
            self.resetting = False
            self.ignore_position_limits = False
            return False

        reached = self.wait_for_reset_position(positions)
        self.resetting = False
        speed_mode_ok = self.set_speed_mode()
        stop_ok = self.stop_servos()
        self.ignore_position_limits = False
        return ok and reached and speed_mode_ok and stop_ok

    def wait_for_reset_position(self, targets):
        deadline = time.time() + max(0.0, self.reset_timeout_sec)
        last_positions = {}

        if self.reset_wait_sec > 0:
            time.sleep(self.reset_wait_sec)

        while time.time() <= deadline:
            reached = True
            for servo_id, target in targets.items():
                try:
                    position = self.read_position(servo_id)
                except Exception as exc:
                    self.get_logger().warn(
                        f"Could not read reset position for servo {servo_id}: {exc}"
                    )
                    return False
                last_positions[servo_id] = position
                if abs(position - target) > self.reset_tolerance:
                    reached = False
            if reached:
                self.get_logger().info(
                    f"Reset reached target positions: {last_positions}"
                )
                return True
            time.sleep(0.05)

        self.get_logger().warn(
            f"Reset timed out before reaching target. Last positions: {last_positions}"
        )
        return False

    def on_cmd(self, msg):
        speed_1 = self.command_to_speed(msg.vel_1, 0)
        speed_2 = self.command_to_speed(msg.vel_2, 1)
        if self.write_pair(speed_1, speed_2):
            self.last_cmd_time = time.monotonic()
            self.timed_out = False

    def check_active_position_limits(self):
        if (
            not self.enforce_position_limits
            or self.ignore_position_limits
            or self.current_speeds == [0, 0]
        ):
            return

        limited_1, limited_2 = self.apply_position_limits(
            self.current_speeds[0],
            self.current_speeds[1],
        )
        if [limited_1, limited_2] == self.current_speeds:
            return

        self.get_logger().warn(
            "Position limit watchdog stopped an active Feetech command.",
            throttle_duration_sec=1.0,
        )
        self.write_pair_raw(limited_1, limited_2)

    def check_timeout(self):
        if self.resetting:
            return
        self.check_active_position_limits()
        if self.last_cmd_time is None or self.command_timeout_sec <= 0:
            return
        if time.monotonic() - self.last_cmd_time <= self.command_timeout_sec:
            return
        if not self.timed_out:
            self.get_logger().warn("Command timeout; stopping Feetech servos")
            self.timed_out = True
        self.stop_servos()

    def on_reset(self, request, response):
        response.success = self.reset_to_position()
        return response

    def destroy_node(self):
        self.stop_servos()
        if self.controller is not None:
            self.controller.disconnect()
        elif self.port is not None:
            self.port.closePort()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = FeetechVelNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
