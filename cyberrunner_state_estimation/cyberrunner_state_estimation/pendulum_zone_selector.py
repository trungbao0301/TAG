"""Passive live selector for board-relative pendulum occlusion zones."""

import json
import os
from pathlib import Path
import time

from ament_index_python.packages import get_package_share_directory
import cv2
from cv_bridge import CvBridge
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image

from cyberrunner_state_estimation.core.detection import Detector
from cyberrunner_state_estimation.core.hole_mask import (
    BOARD_HEIGHT_M,
    BOARD_WIDTH_M,
    MARKERS_CENTERED_M,
)


WINDOW = "Pendulum Occlusion Zone Selector - Passive"


class PendulumZoneSelector(Node):
    """Subscribe-only GUI that converts image rectangles into board coordinates."""

    def __init__(self):
        super().__init__("cyberrunner_pendulum_zone_selector")
        self.declare_parameter("camera_topic", "/cyberrunner_camera/image")
        self.declare_parameter("output_path", "pendulum_occlusion_zones.json")

        share = get_package_share_directory("cyberrunner_state_estimation")
        markers = np.loadtxt(os.path.join(share, "markers.csv"), delimiter=",")
        self.corner_tracker = Detector(markers[4:], ai_mode="off")
        self.bridge = CvBridge()
        self.output_path = Path(
            str(self.get_parameter("output_path").value)
        ).expanduser().resolve()

        self.zones = []
        self.drag_start = None
        self.drag_current = None
        self.image_to_board = None
        self.board_to_image = None
        self.markers_current = False
        self.status = "Hold board near home, then drag a pendulum-covered area"

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        topic = str(self.get_parameter("camera_topic").value)
        self.create_subscription(Image, topic, self._on_image, qos)

        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(WINDOW, self._on_mouse)
        self.get_logger().info(
            f"Passive pendulum-zone selector started; camera={topic}; "
            f"output={self.output_path}; no publishers or control interfaces."
        )

    def _update_homography(self, frame):
        corners_rc = self.corner_tracker.detect_corners(frame)
        self.markers_current = not self.corner_tracker.corners_missing
        if not self.markers_current or not np.all(np.isfinite(corners_rc)):
            return
        try:
            image_xy = corners_rc[:, ::-1].astype(np.float32)
            self.image_to_board = cv2.getPerspectiveTransform(
                image_xy, MARKERS_CENTERED_M
            )
            self.board_to_image = cv2.getPerspectiveTransform(
                MARKERS_CENTERED_M, image_xy
            )
        except cv2.error:
            self.markers_current = False

    def _image_rectangle_to_zone(self, start, end):
        if not self.markers_current or self.image_to_board is None:
            return None
        x0, x1 = sorted((float(start[0]), float(end[0])))
        y0, y1 = sorted((float(start[1]), float(end[1])))
        if x1 - x0 < 4.0 or y1 - y0 < 4.0:
            return None
        image_corners = np.asarray(
            [[[x0, y0]], [[x1, y0]], [[x1, y1]], [[x0, y1]]],
            dtype=np.float32,
        )
        board_centered = cv2.perspectiveTransform(
            image_corners, self.image_to_board
        )[:, 0]
        board_lower_left = board_centered + np.asarray(
            [BOARD_WIDTH_M / 2.0, BOARD_HEIGHT_M / 2.0], dtype=np.float32
        )
        minimum = np.maximum(np.min(board_lower_left, axis=0), [0.0, 0.0])
        maximum = np.minimum(
            np.max(board_lower_left, axis=0), [BOARD_WIDTH_M, BOARD_HEIGHT_M]
        )
        if np.any(maximum <= minimum):
            return None
        return {
            "xmin_m": float(minimum[0]),
            "xmax_m": float(maximum[0]),
            "ymin_m": float(minimum[1]),
            "ymax_m": float(maximum[1]),
        }

    def _zone_image_polygon(self, zone):
        if self.board_to_image is None:
            return None
        offset = np.asarray(
            [BOARD_WIDTH_M / 2.0, BOARD_HEIGHT_M / 2.0], dtype=np.float32
        )
        board_lower_left = np.asarray(
            [
                [zone["xmin_m"], zone["ymin_m"]],
                [zone["xmax_m"], zone["ymin_m"]],
                [zone["xmax_m"], zone["ymax_m"]],
                [zone["xmin_m"], zone["ymax_m"]],
            ],
            dtype=np.float32,
        )
        points = cv2.perspectiveTransform(
            (board_lower_left - offset).reshape(-1, 1, 2), self.board_to_image
        )[:, 0]
        return np.round(points).astype(np.int32)

    @staticmethod
    def _env_value(zones):
        return ";".join(
            f'{zone["xmin_m"]:.6f}:{zone["xmax_m"]:.6f}:'
            f'{zone["ymin_m"]:.6f}:{zone["ymax_m"]:.6f}'
            for zone in zones
        )

    def _save(self):
        if not self.zones:
            self.status = "Nothing saved: add at least one rectangle"
            return
        payload = {
            "format": "cyberrunner_pendulum_occlusion_zones_v1",
            "coordinate_frame": "board_lower_left_m",
            "board_width_m": BOARD_WIDTH_M,
            "board_height_m": BOARD_HEIGHT_M,
            "created_unix_sec": time.time(),
            "zones": self.zones,
            "env_value": self._env_value(self.zones),
        }
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.output_path.with_suffix(self.output_path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        temporary.replace(self.output_path)
        self.status = f"Saved {len(self.zones)} zone(s): {self.output_path}"
        self.get_logger().info(self.status)
        self.get_logger().info(
            "env_tcp: export CYBERRUNNER_OCCLUSION_ZONES_FILE="
            f"{self.output_path}"
        )

    def _on_mouse(self, event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drag_start = (x, y)
            self.drag_current = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.drag_start is not None:
            self.drag_current = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and self.drag_start is not None:
            zone = self._image_rectangle_to_zone(self.drag_start, (x, y))
            self.drag_start = None
            self.drag_current = None
            if zone is None:
                self.status = "Rectangle rejected: markers unavailable or area too small"
            else:
                self.zones.append(zone)
                self.status = f"Added zone {len(self.zones)}; press S to save"

    def _on_image(self, message):
        frame = self.bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
        self._update_homography(frame)
        display = frame.copy()

        overlay = display.copy()
        for index, zone in enumerate(self.zones, start=1):
            polygon = self._zone_image_polygon(zone)
            if polygon is None:
                continue
            cv2.fillPoly(overlay, [polygon], (0, 0, 255))
            cv2.polylines(display, [polygon], True, (0, 0, 255), 2)
            center = tuple(np.mean(polygon, axis=0).astype(int))
            cv2.putText(
                display,
                str(index),
                center,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        cv2.addWeighted(overlay, 0.25, display, 0.75, 0.0, display)

        if self.drag_start is not None and self.drag_current is not None:
            cv2.rectangle(display, self.drag_start, self.drag_current, (0, 255, 255), 2)

        marker_text = "markers=CURRENT" if self.markers_current else "markers=NOT FOUND"
        marker_color = (0, 255, 0) if self.markers_current else (0, 0, 255)
        cv2.putText(
            display,
            f"{marker_text} | zones={len(self.zones)}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            marker_color,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            display,
            "Drag=add | U=undo | C=clear | S=save | Q=quit",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            display,
            self.status,
            (10, display.shape[0] - 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

        cv2.imshow(WINDOW, display)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            rclpy.shutdown()
        elif key == ord("u"):
            if self.zones:
                self.zones.pop()
            self.status = f"Undo; {len(self.zones)} zone(s) remain"
        elif key == ord("c"):
            self.zones.clear()
            self.status = "Cleared all unsaved zones"
        elif key == ord("s"):
            self._save()


def main(args=None):
    rclpy.init(args=args)
    node = PendulumZoneSelector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    cv2.destroyAllWindows()
    if rclpy.ok():
        node.destroy_node()
        rclpy.shutdown()

