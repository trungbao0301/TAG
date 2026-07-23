#!/usr/bin/env python3

import time

import rclpy
from rclpy.node import Node

from tag_interfaces.msg import HiwonderVel
from tag_interfaces.srv import HiwonderReset

try:
    import hid
except ImportError:
    hid = None

try:
    from pylx16a.lx16a import LX16A, ServoError
except ImportError:
    LX16A = None
    ServoError = Exception


DEFAULT_VID = 0x0483
DEFAULT_PID = 0x5750


def _u16(value):
    value = int(value) & 0xFFFF
    return value & 0xFF, (value >> 8) & 0xFF


def _clamp(value, low, high):
    return max(low, min(high, value))


def _step_toward(value, target, max_step):
    if max_step <= 0:
        return target
    delta = target - value
    if abs(delta) <= max_step:
        return target
    return value + _clamp(delta, -max_step, max_step)


class HiwonderHID:
    def __init__(self, vid=DEFAULT_VID, pid=DEFAULT_PID):
        self.vid = int(vid)
        self.pid = int(pid)
        self.dev = None
        self._use_65 = None
        self.path = None
        self.connect()

    def find_devices(self):
        if hid is None:
            return []
        devices = []
        for entry in hid.enumerate(self.vid, self.pid):
            path = entry.get("path")
            if isinstance(path, bytes):
                display_path = path.decode("utf-8", errors="replace")
            else:
                display_path = str(path)
            devices.append(
                {
                    "path": path,
                    "display_path": display_path,
                    "product": entry.get("product_string") or "",
                    "manufacturer": entry.get("manufacturer_string") or "",
                    "interface": entry.get("interface_number"),
                }
            )
        return devices

    def connect(self):
        if hid is None:
            return False

        self.close()
        devices = self.find_devices()
        candidates = devices or [{"path": None, "display_path": "vid/pid fallback"}]

        for entry in candidates:
            try:
                if entry["path"] is None:
                    self.dev = hid.Device(vid=self.vid, pid=self.pid)
                else:
                    self.dev = hid.Device(path=entry["path"])
                self.dev.nonblocking = True
                self._use_65 = None
                self.path = entry["display_path"]
                return True
            except Exception:
                self.dev = None
                self.path = None

        return False

    def close(self):
        try:
            if self.dev:
                self.dev.close()
        except Exception:
            pass
        self.dev = None
        self.path = None

    def _write64(self, payload):
        if self.dev is None:
            return False
        if len(payload) > 64:
            raise ValueError("Hiwonder HID payload is longer than 64 bytes")

        packet = payload + bytes(64 - len(payload))

        if self._use_65 is True:
            self.dev.write(bytes([0x00]) + packet)
            return True
        if self._use_65 is False:
            self.dev.write(packet)
            return True

        try:
            self.dev.write(bytes([0x00]) + packet)
            self._use_65 = True
        except Exception:
            self.dev.write(packet)
            self._use_65 = False

        return True

    def move_two(self, servo1_id, pos1, servo2_id, pos2, time_ms=30):
        if self.dev is None and not self.connect():
            return False

        pos1 = _clamp(int(pos1), 0, 1000)
        pos2 = _clamp(int(pos2), 0, 1000)
        time_ms = _clamp(int(time_ms), 0, 30000)

        t_low, t_high = _u16(time_ms)
        p1_low, p1_high = _u16(pos1)
        p2_low, p2_high = _u16(pos2)

        # CMD_SERVO_MOVE: header, len, command, count, time, servo data.
        payload = bytes(
            [
                0x55,
                0x55,
                0x0B,
                0x03,
                0x02,
                t_low,
                t_high,
                int(servo1_id) & 0xFF,
                p1_low,
                p1_high,
                int(servo2_id) & 0xFF,
                p2_low,
                p2_high,
            ]
        )

        try:
            return self._write64(payload)
        except Exception:
            self.close()
            self.connect()
            return False

    def read_temperature(self, servo_id):
        """Return servo temperature in Celsius, or None if unsupported.

        The current Hiwonder HID controller path used here exposes the simple
        controller move command, but it does not expose per-servo telemetry.
        Keep this method as the single integration point if the hardware/API is
        later changed to a bus-servo interface that can read temperature.
        """
        return None


class HiwonderCompatNode(Node):
    def __init__(self):
        super().__init__("tag_hiwonder_compat")

        self.declare_parameter("vid", DEFAULT_VID)
        self.declare_parameter("pid", DEFAULT_PID)
        self.declare_parameter("servo1_id", 1)
        self.declare_parameter("servo2_id", 2)
        self.declare_parameter("home_pos_1", 500)
        self.declare_parameter("home_pos_2", 500)
        self.declare_parameter("servo_min_1", 100)
        self.declare_parameter("servo_max_1", 900)
        self.declare_parameter("servo_min_2", 100)
        self.declare_parameter("servo_max_2", 900)
        self.declare_parameter("scale_1", 1.5)
        self.declare_parameter("scale_2", 1.5)
        self.declare_parameter("move_time_ms", 30)
        self.declare_parameter("reset_time_ms", 600)
        self.declare_parameter("reset_prehome_pos_1", 700)
        self.declare_parameter("reset_prehome_pos_2", 700)
        self.declare_parameter("reset_prehome_time_ms", 60)
        self.declare_parameter("reset_prehome_wait_sec", 0.5)
        self.declare_parameter("reconnect_interval", 2.0)
        self.declare_parameter("command_rate_hz", 30.0)
        self.declare_parameter("max_step_per_tick_1", 20.0)
        self.declare_parameter("max_step_per_tick_2", 20.0)
        self.declare_parameter("deadband", 1.0)
        self.declare_parameter("command_timeout_sec", 1.0)
        self.declare_parameter("timeout_go_home", True)
        self.declare_parameter("temp_watch_enabled", True)
        self.declare_parameter("temp_limit_c", 70.0)
        self.declare_parameter("overheat_pause_sec", 300.0)
        self.declare_parameter("temp_check_period_sec", 2.0)
        self.declare_parameter("temp_serial_port", "")
        self.declare_parameter("temp_serial_timeout", 0.02)
        self.declare_parameter("servo_temp_upper_limit_c", 90)

        self.vid = int(self.get_parameter("vid").value)
        self.pid = int(self.get_parameter("pid").value)
        self.servo1_id = int(self.get_parameter("servo1_id").value)
        self.servo2_id = int(self.get_parameter("servo2_id").value)
        self.home_pos_1 = int(self.get_parameter("home_pos_1").value)
        self.home_pos_2 = int(self.get_parameter("home_pos_2").value)
        self.servo_min_1 = int(self.get_parameter("servo_min_1").value)
        self.servo_max_1 = int(self.get_parameter("servo_max_1").value)
        self.servo_min_2 = int(self.get_parameter("servo_min_2").value)
        self.servo_max_2 = int(self.get_parameter("servo_max_2").value)
        self.scale_1 = float(self.get_parameter("scale_1").value)
        self.scale_2 = float(self.get_parameter("scale_2").value)
        self.move_time_ms = int(self.get_parameter("move_time_ms").value)
        self.reset_time_ms = int(self.get_parameter("reset_time_ms").value)
        self.reset_prehome_pos_1 = int(self.get_parameter("reset_prehome_pos_1").value)
        self.reset_prehome_pos_2 = int(self.get_parameter("reset_prehome_pos_2").value)
        self.reset_prehome_time_ms = int(
            self.get_parameter("reset_prehome_time_ms").value
        )
        self.reset_prehome_wait_sec = float(
            self.get_parameter("reset_prehome_wait_sec").value
        )
        self.reconnect_interval = float(
            self.get_parameter("reconnect_interval").value
        )
        self.command_rate_hz = float(self.get_parameter("command_rate_hz").value)
        self.max_step_per_tick_1 = float(
            self.get_parameter("max_step_per_tick_1").value
        )
        self.max_step_per_tick_2 = float(
            self.get_parameter("max_step_per_tick_2").value
        )
        self.deadband = float(self.get_parameter("deadband").value)
        self.command_timeout_sec = float(
            self.get_parameter("command_timeout_sec").value
        )
        self.timeout_go_home = bool(self.get_parameter("timeout_go_home").value)
        self.temp_watch_enabled = bool(
            self.get_parameter("temp_watch_enabled").value
        )
        self.temp_limit_c = float(self.get_parameter("temp_limit_c").value)
        self.overheat_pause_sec = float(
            self.get_parameter("overheat_pause_sec").value
        )
        self.temp_check_period_sec = max(
            0.2, float(self.get_parameter("temp_check_period_sec").value)
        )
        self.temp_serial_port = str(self.get_parameter("temp_serial_port").value)
        self.temp_serial_timeout = float(
            self.get_parameter("temp_serial_timeout").value
        )
        self.servo_temp_upper_limit_c = int(
            self.get_parameter("servo_temp_upper_limit_c").value
        )
        self.command_rate_hz = max(self.command_rate_hz, 1.0)

        if hid is None:
            self.get_logger().error(
                "Python module 'hid' is not installed. Install hid/hidapi before "
                "using the Hiwonder compatibility node."
            )

        self.hid = HiwonderHID(self.vid, self.pid)
        self.temp_servos = {}
        if self.hid.dev is not None:
            self.get_logger().info(
                f"Opened Hiwonder HID {self.vid:04x}:{self.pid:04x} "
                f"at {self.hid.path}"
            )
            time.sleep(0.1)
            self.reset_sequence()
        else:
            self.get_logger().error(
                "Could not open Hiwonder HID. The node will keep retrying. "
                f"Expected USB id {self.vid:04x}:{self.pid:04x}."
            )

        self.setup_temperature_reader()

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
        self.target_pos_1 = float(self.home_pos_1)
        self.target_pos_2 = float(self.home_pos_2)
        self.current_pos_1 = float(self.home_pos_1)
        self.current_pos_2 = float(self.home_pos_2)
        self.last_sent_pos_1 = None
        self.last_sent_pos_2 = None
        self.last_cmd_time = None
        self.timed_out = False
        self.paused_until = 0.0
        self.pause_reason = ""
        self.temp_unavailable_warned = False
        self.command_timer = self.create_timer(
            1.0 / self.command_rate_hz,
            self.send_smoothed_command,
        )
        self.reconnect_timer = self.create_timer(
            self.reconnect_interval,
            self.check_connection,
        )
        if self.temp_watch_enabled:
            self.temp_timer = self.create_timer(
                self.temp_check_period_sec,
                self.check_temperatures,
            )

    def setup_temperature_reader(self):
        if not self.temp_watch_enabled:
            return

        if not self.temp_serial_port:
            self.get_logger().warn(
                "Temperature watch is enabled, but temp_serial_port is empty. "
                "The current HID controller cannot read per-servo temperature. "
                "For LX-16A telemetry, install pylx16a and pass "
                "-p temp_serial_port:=/dev/ttyUSBX."
            )
            return

        if LX16A is None:
            self.get_logger().error(
                "temp_serial_port was provided, but pylx16a is not installed. "
                "Install with: python3 -m pip install pylx16a"
            )
            return

        try:
            LX16A.initialize(self.temp_serial_port, timeout=self.temp_serial_timeout)
            self.temp_servos = {
                self.servo1_id: LX16A(self.servo1_id),
                self.servo2_id: LX16A(self.servo2_id),
            }
            if self.servo_temp_upper_limit_c > 0:
                for servo_id, servo in self.temp_servos.items():
                    servo.set_temp_limit(self.servo_temp_upper_limit_c)
                    self.get_logger().info(
                        f"Set servo {servo_id} temperature upper limit to "
                        f"{self.servo_temp_upper_limit_c} C."
                    )
            self.get_logger().info(
                "LX-16A temperature reader enabled on "
                f"{self.temp_serial_port} for servos "
                f"{self.servo1_id}, {self.servo2_id}."
            )
        except ServoError as exc:
            self.temp_servos = {}
            self.get_logger().error(
                f"Could not initialize LX-16A temperature reader: {exc}"
            )
        except Exception as exc:
            self.temp_servos = {}
            self.get_logger().error(
                f"Could not initialize temperature serial port "
                f"{self.temp_serial_port}: {exc}"
            )

    def check_connection(self):
        if self.hid.dev is not None:
            return

        if self.hid.connect():
            self.get_logger().info(
                f"Reconnected to Hiwonder HID {self.vid:04x}:{self.pid:04x} "
                f"at {self.hid.path}"
            )
            time.sleep(0.1)
            self.reset_sequence()
            self.current_pos_1 = float(self.home_pos_1)
            self.current_pos_2 = float(self.home_pos_2)
            self.target_pos_1 = float(self.home_pos_1)
            self.target_pos_2 = float(self.home_pos_2)
            self.last_sent_pos_1 = None
            self.last_sent_pos_2 = None
        else:
            devices = self.hid.find_devices()
            if devices:
                paths = ", ".join(d["display_path"] for d in devices)
                self.get_logger().warn(
                    "Hiwonder HID reconnect failed even though matching USB "
                    f"device exists: {paths}. Check /dev/hidraw permissions."
                )
            else:
                self.get_logger().warn(
                    f"Hiwonder HID {self.vid:04x}:{self.pid:04x} not present; "
                    "replug USB/cable and the node will auto-reconnect."
                )

    def command_to_position(self, vel_1, vel_2):
        pos1 = self.home_pos_1 + float(vel_1) * self.scale_1
        pos2 = self.home_pos_2 + float(vel_2) * self.scale_2

        pos1 = _clamp(round(pos1), self.servo_min_1, self.servo_max_1)
        pos2 = _clamp(round(pos2), self.servo_min_2, self.servo_max_2)
        return int(pos1), int(pos2)

    def go_home(self, time_ms):
        return self.hid.move_two(
            self.servo1_id,
            self.home_pos_1,
            self.servo2_id,
            self.home_pos_2,
            time_ms,
        )

    def reset_sequence(self):
        ok = self.hid.move_two(
            self.servo1_id,
            self.reset_prehome_pos_1,
            self.servo2_id,
            self.reset_prehome_pos_2,
            self.reset_prehome_time_ms,
        )
        if not ok:
            return False

        if self.reset_prehome_wait_sec > 0:
            time.sleep(self.reset_prehome_wait_sec)

        return self.go_home(self.reset_time_ms)

    def is_paused(self, now=None):
        now = time.monotonic() if now is None else now
        if now < self.paused_until:
            return True
        if self.paused_until > 0.0:
            self.get_logger().warn("Hiwonder pause finished; accepting commands again.")
            self.paused_until = 0.0
            self.pause_reason = ""
        return False

    def pause_for_overheat(self, temps):
        now = time.monotonic()
        self.paused_until = max(self.paused_until, now + self.overheat_pause_sec)
        self.pause_reason = (
            f"temperature over {self.temp_limit_c:.1f} C: "
            + ", ".join(f"servo {sid}={temp:.1f} C" for sid, temp in temps)
        )
        self.target_pos_1 = float(self.home_pos_1)
        self.target_pos_2 = float(self.home_pos_2)
        self.last_cmd_time = None
        self.timed_out = True
        self.go_home(self.reset_time_ms)
        self.get_logger().error(
            f"Hiwonder overheat guard: {self.pause_reason}. "
            f"Pausing commands for {self.overheat_pause_sec:.0f}s."
        )

    def check_temperatures(self):
        if self.hid.dev is None or self.is_paused():
            return

        readings = []
        for servo_id in (self.servo1_id, self.servo2_id):
            try:
                if servo_id in self.temp_servos:
                    temp = float(self.temp_servos[servo_id].get_temp())
                else:
                    temp = self.hid.read_temperature(servo_id)
            except Exception as exc:
                self.get_logger().warn(
                    f"Failed to read Hiwonder servo {servo_id} temperature: {exc}",
                    throttle_duration_sec=10.0,
                )
                temp = None
            readings.append((servo_id, temp))

        if all(temp is None for _, temp in readings):
            if not self.temp_unavailable_warned:
                self.get_logger().warn(
                    "Hiwonder temperature telemetry is not available through "
                    "the current HID controller API; overheat auto-pause cannot "
                    "measure real servo temperature."
                )
                self.temp_unavailable_warned = True
            return

        hot = [
            (servo_id, temp)
            for servo_id, temp in readings
            if temp is not None and temp >= self.temp_limit_c
        ]
        if hot:
            self.pause_for_overheat(hot)

    def on_cmd(self, msg):
        if self.is_paused():
            return
        pos1, pos2 = self.command_to_position(msg.vel_1, msg.vel_2)
        self.target_pos_1 = float(pos1)
        self.target_pos_2 = float(pos2)
        self.last_cmd_time = time.monotonic()
        self.timed_out = False

    def send_smoothed_command(self):
        if self.hid.dev is None:
            return

        if self.is_paused():
            return

        if self.last_cmd_time is None:
            return

        now = time.monotonic()
        if (
            self.command_timeout_sec > 0
            and now - self.last_cmd_time > self.command_timeout_sec
        ):
            if not self.timeout_go_home:
                return
            self.target_pos_1 = float(self.home_pos_1)
            self.target_pos_2 = float(self.home_pos_2)
            if not self.timed_out:
                self.get_logger().warn(
                    "Command timeout; returning Hiwonder servos home"
                )
                self.timed_out = True

        next_pos_1 = _step_toward(
            self.current_pos_1,
            self.target_pos_1,
            self.max_step_per_tick_1,
        )
        next_pos_2 = _step_toward(
            self.current_pos_2,
            self.target_pos_2,
            self.max_step_per_tick_2,
        )

        if (
            self.last_sent_pos_1 is not None
            and abs(next_pos_1 - self.last_sent_pos_1) <= self.deadband
            and abs(next_pos_2 - self.last_sent_pos_2) <= self.deadband
        ):
            return

        pos1 = int(round(next_pos_1))
        pos2 = int(round(next_pos_2))
        ok = self.hid.move_two(
            self.servo1_id,
            pos1,
            self.servo2_id,
            pos2,
            self.move_time_ms,
        )

        if not ok:
            self.get_logger().warn("Hiwonder command failed; will reconnect")
            self.hid.close()
            return

        self.current_pos_1 = float(pos1)
        self.current_pos_2 = float(pos2)
        self.last_sent_pos_1 = float(pos1)
        self.last_sent_pos_2 = float(pos2)
        self.get_logger().info(
            f"target=({self.target_pos_1:.1f}, {self.target_pos_2:.1f}) "
            f"pos=({pos1}, {pos2})",
            throttle_duration_sec=2.0,
        )

    def on_reset(self, request, response):
        ok = self.reset_sequence()
        self.target_pos_1 = float(self.home_pos_1)
        self.target_pos_2 = float(self.home_pos_2)
        self.current_pos_1 = float(self.home_pos_1)
        self.current_pos_2 = float(self.home_pos_2)
        self.last_sent_pos_1 = None
        self.last_sent_pos_2 = None
        self.last_cmd_time = None
        self.timed_out = False
        response.success = 1 if ok else 0
        if not ok:
            self.hid.close()
        return response


def main(args=None):
    rclpy.init(args=args)
    node = HiwonderCompatNode()
    try:
        rclpy.spin(node)
    finally:
        node.hid.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
