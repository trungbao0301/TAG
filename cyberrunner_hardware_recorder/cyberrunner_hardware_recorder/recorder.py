#!/usr/bin/env python3

"""Subscriber-only CyberRunner hardware telemetry recorder."""

import argparse
import csv
import datetime as dt
import json
import math
import os
import platform
import signal
import socket
import sys
import time
from pathlib import Path

import numpy as np
import rclpy
from cyberrunner_interfaces.msg import DynamixelVel, StateEstimate, StateEstimateSub
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from sensor_msgs.msg import Image


VERSION = "0.1.0"

CAMERA_FIELDS = [
    "receipt_monotonic_ns",
    "receipt_elapsed_sec",
    "source_stamp_ns",
    "source_sequence",
    "height",
    "width",
    "encoding",
    "step_bytes",
    "data_bytes",
    "frame_id",
    "saved_frame",
]

STATE_FIELDS = [
    "receipt_monotonic_ns",
    "receipt_elapsed_sec",
    "source_topic",
    "source_stamp_ns",
    "x_b",
    "y_b",
    "x_b_dot",
    "y_b_dot",
    "alpha",
    "beta",
    "ball_detected",
]

COMMAND_FIELDS = [
    "receipt_monotonic_ns",
    "receipt_elapsed_sec",
    "source_topic",
    "vel_1",
    "vel_2",
]

EPISODE_FIELDS = [
    "episode_index",
    "start_receipt_monotonic_ns",
    "end_receipt_monotonic_ns",
    "duration_sec",
    "outcome",
    "inference_source",
    "state_samples",
    "detected_samples",
    "missing_samples",
    "command_samples",
    "command_abs_max_1",
    "command_abs_max_2",
    "alpha_min",
    "alpha_max",
    "beta_min",
    "beta_max",
]


def _utc_now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _stamp_ns(header):
    if header is None:
        return ""
    return int(header.stamp.sec) * 1_000_000_000 + int(header.stamp.nanosec)


def _finite(value):
    return math.isfinite(float(value))


class PassiveHardwareRecorder(Node):
    """Record receipts from existing topics without controlling the system."""

    def __init__(self, args):
        super().__init__(
            "cyberrunner_passive_hardware_recorder",
            enable_rosout=False,
            start_parameter_services=False,
        )
        # Humble's rclpy creates /parameter_events even when parameter services
        # are disabled. This node never declares or changes parameters, so
        # remove that infrastructure endpoint to remain literally subscriber-only.
        self.destroy_publisher(self._parameter_event_publisher)
        self.args = args
        self.start_monotonic_ns = time.monotonic_ns()
        self.start_utc = _utc_now()
        self.stop_requested = False

        session_name = args.session_name or dt.datetime.now().strftime(
            "session_%Y%m%d_%H%M%S"
        )
        self.session_dir = Path(args.output_root).expanduser().resolve() / session_name
        self.session_dir.mkdir(parents=True, exist_ok=False)
        self.frames_dir = self.session_dir / "frames"
        if args.frame_every_sec > 0:
            self.frames_dir.mkdir()

        self.files = {}
        self.writers = {}
        self._open_csv("camera", "camera_timing.csv", CAMERA_FIELDS)
        self._open_csv("state", "state.csv", STATE_FIELDS)
        self._open_csv("command", "motor_commands.csv", COMMAND_FIELDS)
        self._open_csv("episode", "episodes.csv", EPISODE_FIELDS)

        self.counts = {
            "camera": 0,
            "state": 0,
            "state_subimg": 0,
            "command": 0,
            "saved_frames": 0,
        }
        self.first_camera = None
        self.last_frame_save_ns = None
        self.episode_index = 0
        self.active_episode = None
        self.missing_since_ns = None

        camera_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        telemetry_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=100,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )

        self._subscription_keepalive = [
            self.create_subscription(
                Image, args.camera_topic, self._on_camera, camera_qos
            ),
            self.create_subscription(
                StateEstimate, args.state_topic, self._on_state, telemetry_qos
            ),
            self.create_subscription(
                DynamixelVel, args.command_topic, self._on_command, telemetry_qos
            ),
        ]
        if args.state_subimg_topic:
            self._subscription_keepalive.append(
                self.create_subscription(
                    StateEstimateSub,
                    args.state_subimg_topic,
                    self._on_state_subimg,
                    telemetry_qos,
                )
            )

        self.metadata = {
            "schema_version": 1,
            "recorder_version": VERSION,
            "session_name": session_name,
            "session_directory": str(self.session_dir),
            "start_utc": self.start_utc,
            "start_monotonic_ns": self.start_monotonic_ns,
            "host": socket.gethostname(),
            "platform": platform.platform(),
            "python": sys.version,
            "pid": os.getpid(),
            "ros_domain_id": os.environ.get("ROS_DOMAIN_ID", "default"),
            "rmw_implementation": os.environ.get("RMW_IMPLEMENTATION", "default"),
            "requested_duration_sec": args.duration_sec,
            "topics": {
                "camera": args.camera_topic,
                "state": args.state_topic,
                "state_subimg": args.state_subimg_topic,
                "motor_commands": args.command_topic,
                "dreamer_action": args.command_topic,
                "dreamer_episode_events": None,
            },
            "qos_requested": {
                "camera": "best_effort, volatile, keep_last depth=5",
                "telemetry": "reliable, volatile, keep_last depth=100",
            },
            "camera_payload_policy": {
                "record_every_image": False,
                "frame_every_sec": args.frame_every_sec,
                "jpeg_quality": args.jpeg_quality,
            },
            "command_limits": {
                "vel_1": [-args.command_limit_1, args.command_limit_1],
                "vel_2": [-args.command_limit_2, args.command_limit_2],
                "source": "recorder arguments; inspect bridge runtime separately",
            },
            "episode_policy": {
                "official_event_topic_available": False,
                "method": "inferred from finite ball-state intervals",
                "missing_grace_sec": args.episode_missing_grace_sec,
                "warning": "Not an official Dreamer episode boundary.",
            },
            "safety": {
                "subscriber_only": True,
                "publishes_messages": False,
                "creates_services": False,
                "creates_service_clients": False,
                "creates_action_clients": False,
                "controls_motors": False,
            },
        }
        self._write_metadata()
        self._write_topic_report()
        self.get_logger().info(f"Passive recording session: {self.session_dir}")

    def _open_csv(self, key, filename, fields):
        handle = (self.session_dir / filename).open("w", newline="", buffering=1)
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        self.files[key] = handle
        self.writers[key] = writer

    def _elapsed_sec(self, receipt_ns):
        return (receipt_ns - self.start_monotonic_ns) / 1e9

    def _on_camera(self, msg):
        receipt_ns = time.monotonic_ns()
        self.counts["camera"] += 1
        saved_frame = ""

        if self.first_camera is None:
            self.first_camera = {
                "height": int(msg.height),
                "width": int(msg.width),
                "encoding": str(msg.encoding),
                "step_bytes": int(msg.step),
                "data_bytes": len(msg.data),
                "frame_id": str(msg.header.frame_id),
            }
            self.metadata["camera_first_message"] = self.first_camera
            self._write_metadata()

        if self._should_save_frame(receipt_ns):
            saved_frame = self._save_frame(msg, receipt_ns)

        self.writers["camera"].writerow(
            {
                "receipt_monotonic_ns": receipt_ns,
                "receipt_elapsed_sec": f"{self._elapsed_sec(receipt_ns):.9f}",
                "source_stamp_ns": _stamp_ns(msg.header),
                "source_sequence": self.counts["camera"],
                "height": int(msg.height),
                "width": int(msg.width),
                "encoding": msg.encoding,
                "step_bytes": int(msg.step),
                "data_bytes": len(msg.data),
                "frame_id": msg.header.frame_id,
                "saved_frame": saved_frame,
            }
        )

    def _should_save_frame(self, receipt_ns):
        if self.args.frame_every_sec <= 0:
            return False
        if self.last_frame_save_ns is None:
            return True
        return (
            receipt_ns - self.last_frame_save_ns
            >= self.args.frame_every_sec * 1_000_000_000
        )

    def _save_frame(self, msg, receipt_ns):
        try:
            import cv2

            encoding = msg.encoding.lower()
            data = np.frombuffer(msg.data, dtype=np.uint8)
            if encoding in ("mono8", "8uc1"):
                image = data.reshape(msg.height, msg.step)[:, : msg.width]
            elif encoding in ("bgr8", "rgb8"):
                image = data.reshape(msg.height, msg.step)[:, : msg.width * 3]
                image = image.reshape(msg.height, msg.width, 3)
                if encoding == "rgb8":
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            elif encoding in ("bgra8", "rgba8"):
                image = data.reshape(msg.height, msg.step)[:, : msg.width * 4]
                image = image.reshape(msg.height, msg.width, 4)
                conversion = (
                    cv2.COLOR_RGBA2BGR if encoding == "rgba8" else cv2.COLOR_BGRA2BGR
                )
                image = cv2.cvtColor(image, conversion)
            else:
                return "unsupported:" + msg.encoding

            filename = f"frame_{self.counts['camera']:08d}_{receipt_ns}.jpg"
            path = self.frames_dir / filename
            ok = cv2.imwrite(
                str(path),
                image,
                [int(cv2.IMWRITE_JPEG_QUALITY), self.args.jpeg_quality],
            )
            if not ok:
                return "write_failed"
            self.last_frame_save_ns = receipt_ns
            self.counts["saved_frames"] += 1
            return str(Path("frames") / filename)
        except Exception as exc:
            self.get_logger().warn(f"Sparse frame save failed: {exc}")
            return "error"

    def _on_state(self, msg):
        receipt_ns = time.monotonic_ns()
        self.counts["state"] += 1
        self._write_state(msg, receipt_ns, self.args.state_topic, "")
        self._update_episode(msg, receipt_ns)

    def _on_state_subimg(self, msg):
        receipt_ns = time.monotonic_ns()
        self.counts["state_subimg"] += 1
        self._write_state(
            msg.state,
            receipt_ns,
            self.args.state_subimg_topic,
            _stamp_ns(msg.subimg.header),
        )

    def _write_state(self, msg, receipt_ns, topic, source_stamp_ns):
        detected = _finite(msg.x_b) and _finite(msg.y_b)
        self.writers["state"].writerow(
            {
                "receipt_monotonic_ns": receipt_ns,
                "receipt_elapsed_sec": f"{self._elapsed_sec(receipt_ns):.9f}",
                "source_topic": topic,
                "source_stamp_ns": source_stamp_ns,
                "x_b": repr(float(msg.x_b)),
                "y_b": repr(float(msg.y_b)),
                "x_b_dot": repr(float(msg.x_b_dot)),
                "y_b_dot": repr(float(msg.y_b_dot)),
                "alpha": repr(float(msg.alpha)),
                "beta": repr(float(msg.beta)),
                "ball_detected": int(detected),
            }
        )

    def _on_command(self, msg):
        receipt_ns = time.monotonic_ns()
        self.counts["command"] += 1
        vel_1 = float(msg.vel_1)
        vel_2 = float(msg.vel_2)
        self.writers["command"].writerow(
            {
                "receipt_monotonic_ns": receipt_ns,
                "receipt_elapsed_sec": f"{self._elapsed_sec(receipt_ns):.9f}",
                "source_topic": self.args.command_topic,
                "vel_1": repr(vel_1),
                "vel_2": repr(vel_2),
            }
        )
        if self.active_episode is not None:
            episode = self.active_episode
            episode["command_samples"] += 1
            episode["command_abs_max_1"] = max(
                episode["command_abs_max_1"], abs(vel_1)
            )
            episode["command_abs_max_2"] = max(
                episode["command_abs_max_2"], abs(vel_2)
            )

    def _new_episode(self, receipt_ns):
        self.episode_index += 1
        self.active_episode = {
            "episode_index": self.episode_index,
            "start_receipt_monotonic_ns": receipt_ns,
            "state_samples": 0,
            "detected_samples": 0,
            "missing_samples": 0,
            "command_samples": 0,
            "command_abs_max_1": 0.0,
            "command_abs_max_2": 0.0,
            "alpha_min": math.inf,
            "alpha_max": -math.inf,
            "beta_min": math.inf,
            "beta_max": -math.inf,
        }

    def _update_episode(self, msg, receipt_ns):
        detected = _finite(msg.x_b) and _finite(msg.y_b)
        if detected and self.active_episode is None:
            self._new_episode(receipt_ns)

        if self.active_episode is None:
            return

        episode = self.active_episode
        episode["state_samples"] += 1
        episode["detected_samples"] += int(detected)
        episode["missing_samples"] += int(not detected)
        if _finite(msg.alpha):
            episode["alpha_min"] = min(episode["alpha_min"], float(msg.alpha))
            episode["alpha_max"] = max(episode["alpha_max"], float(msg.alpha))
        if _finite(msg.beta):
            episode["beta_min"] = min(episode["beta_min"], float(msg.beta))
            episode["beta_max"] = max(episode["beta_max"], float(msg.beta))

        if detected:
            self.missing_since_ns = None
            return

        if self.missing_since_ns is None:
            self.missing_since_ns = receipt_ns
            return

        grace_ns = int(self.args.episode_missing_grace_sec * 1e9)
        if receipt_ns - self.missing_since_ns >= grace_ns:
            self._finish_episode(self.missing_since_ns, "ball_lost_inferred")
            self.missing_since_ns = None

    def _finish_episode(self, end_ns, outcome):
        episode = self.active_episode
        if episode is None:
            return

        def clean(value):
            return "" if not math.isfinite(value) else repr(float(value))

        self.writers["episode"].writerow(
            {
                "episode_index": episode["episode_index"],
                "start_receipt_monotonic_ns": episode[
                    "start_receipt_monotonic_ns"
                ],
                "end_receipt_monotonic_ns": end_ns,
                "duration_sec": (
                    f"{(end_ns - episode['start_receipt_monotonic_ns']) / 1e9:.9f}"
                ),
                "outcome": outcome,
                "inference_source": "ball_visibility",
                "state_samples": episode["state_samples"],
                "detected_samples": episode["detected_samples"],
                "missing_samples": episode["missing_samples"],
                "command_samples": episode["command_samples"],
                "command_abs_max_1": repr(episode["command_abs_max_1"]),
                "command_abs_max_2": repr(episode["command_abs_max_2"]),
                "alpha_min": clean(episode["alpha_min"]),
                "alpha_max": clean(episode["alpha_max"]),
                "beta_min": clean(episode["beta_min"]),
                "beta_max": clean(episode["beta_max"]),
            }
        )
        self.active_episode = None

    def _write_metadata(self):
        temp = self.session_dir / "session_metadata.json.tmp"
        temp.write_text(json.dumps(self.metadata, indent=2, sort_keys=True) + "\n")
        temp.replace(self.session_dir / "session_metadata.json")

    def _endpoint_lines(self, topic):
        lines = []
        for label, getter in (
            ("publishers", self.get_publishers_info_by_topic),
            ("subscriptions", self.get_subscriptions_info_by_topic),
        ):
            try:
                endpoints = getter(topic)
            except Exception as exc:
                lines.append(f"  {label}: graph query failed: {exc}")
                continue
            lines.append(f"  {label}: {len(endpoints)}")
            for endpoint in endpoints:
                qos = endpoint.qos_profile
                lines.append(
                    "    "
                    f"{endpoint.node_namespace}{endpoint.node_name} "
                    f"type={endpoint.topic_type} "
                    f"reliability={qos.reliability.name} "
                    f"durability={qos.durability.name} "
                    f"history={qos.history.name} depth={qos.depth}"
                )
        return lines

    def _write_topic_report(self):
        lines = [
            "CyberRunner passive hardware recorder topic report",
            f"generated_utc: {_utc_now()}",
            "",
            "Safety:",
            "  This recorder creates subscriptions only.",
            "  It does not publish, call services/actions, reset hardware, or control motors.",
            "",
            "Configured topics:",
        ]
        topic_types = [
            (self.args.camera_topic, "sensor_msgs/msg/Image"),
            (self.args.state_topic, "cyberrunner_interfaces/msg/StateEstimate"),
            (
                self.args.state_subimg_topic,
                "cyberrunner_interfaces/msg/StateEstimateSub",
            ),
            (
                self.args.command_topic,
                "cyberrunner_interfaces/msg/DynamixelVel",
            ),
        ]
        for topic, expected_type in topic_types:
            if not topic:
                continue
            lines.extend(["", f"{topic}", f"  expected_type: {expected_type}"])
            lines.extend(self._endpoint_lines(topic))
        lines.extend(
            [
                "",
                "DreamerV3 ROS visibility:",
                f"  action proxy: {self.args.command_topic}",
                "  episode-event topic: not observed",
                "  episodes.csv boundaries are explicitly passive inferences.",
            ]
        )
        (self.session_dir / "topic_report.txt").write_text("\n".join(lines) + "\n")

    def finish(self, reason):
        if self.stop_requested:
            return
        self.stop_requested = True
        end_ns = time.monotonic_ns()
        self._finish_episode(end_ns, "recording_stopped")
        self.metadata.update(
            {
                "end_utc": _utc_now(),
                "end_monotonic_ns": end_ns,
                "duration_sec": (end_ns - self.start_monotonic_ns) / 1e9,
                "stop_reason": reason,
                "message_counts": dict(self.counts),
            }
        )
        self._write_topic_report()
        self._write_metadata()
        for handle in self.files.values():
            handle.flush()
            handle.close()


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="hardware_recordings")
    parser.add_argument("--session-name", default="")
    parser.add_argument("--duration-sec", type=float, default=0.0)
    parser.add_argument("--camera-topic", default="/cyberrunner_camera/image")
    parser.add_argument(
        "--state-topic", default="/cyberrunner_state_estimation/estimate"
    )
    parser.add_argument(
        "--state-subimg-topic",
        default="/cyberrunner_state_estimation/estimate_subimg",
    )
    parser.add_argument("--command-topic", default="/cyberrunner_dynamixel/cmd")
    parser.add_argument("--command-limit-1", type=float, default=180.0)
    parser.add_argument("--command-limit-2", type=float, default=180.0)
    parser.add_argument("--episode-missing-grace-sec", type=float, default=0.35)
    parser.add_argument(
        "--frame-every-sec",
        type=float,
        default=0.0,
        help="Save one JPEG at this interval; 0 records timing only.",
    )
    parser.add_argument("--jpeg-quality", type=int, default=85)
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    if args.duration_sec < 0:
        raise SystemExit("--duration-sec must be nonnegative")
    if args.frame_every_sec < 0:
        raise SystemExit("--frame-every-sec must be nonnegative")
    args.jpeg_quality = max(1, min(100, args.jpeg_quality))

    rclpy.init()
    node = PassiveHardwareRecorder(args)

    def request_stop(signum, _frame):
        node.get_logger().info(f"Stopping on signal {signum}")
        node.stop_requested = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    deadline = (
        node.start_monotonic_ns + int(args.duration_sec * 1e9)
        if args.duration_sec > 0
        else None
    )
    reason = "signal"
    try:
        while rclpy.ok() and not node.stop_requested:
            rclpy.spin_once(node, timeout_sec=0.1)
            if deadline is not None and time.monotonic_ns() >= deadline:
                reason = "duration_complete"
                break
    except KeyboardInterrupt:
        reason = "keyboard_interrupt"
    finally:
        node.stop_requested = False
        node.finish(reason)
        session_dir = node.session_dir
        node.destroy_node()
        rclpy.shutdown()
        print(session_dir)


if __name__ == "__main__":
    main()
