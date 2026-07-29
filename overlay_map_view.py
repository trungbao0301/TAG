#!/usr/bin/env python3

import math
import os
import numpy as np
import cv2

import rclpy
from rclpy.node import Node

from cyberrunner_interfaces.msg import StateEstimateSub

from cyberrunner_dreamer.cyberrunner_layout_custom import cyberrunner_dxf_layout
from cyberrunner_dreamer.path import LinearPath


LAYOUT = cyberrunner_dxf_layout
BOARD_W = float(LAYOUT["board_width"])
BOARD_H = float(LAYOUT["board_height"])
WALL_R = float(LAYOUT.get("ball_radius", 0.006))
HOLE_R = float(max(LAYOUT.get("hole_radii", [0.0075])))

VIEW_W = 900
VIEW_H = int(VIEW_W * BOARD_H / BOARD_W)

# State estimate uses centered coordinates; the generated DXF layout uses a
# lower-left origin.
OFFSET = np.array([BOARD_W, BOARD_H], dtype=np.float32) / 2.0


def world_to_px(x, y):
    """World meters: x right, y up. Image pixels: x right, y down."""
    px = int(round(x / BOARD_W * VIEW_W))
    py = int(round((BOARD_H - y) / BOARD_H * VIEW_H))
    return px, py


def draw_text(img, text, org, scale=0.55, thickness=1):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness, cv2.LINE_AA)


class OverlayMapView(Node):
    def __init__(self):
        super().__init__("overlay_map_view")

        self.layout = LAYOUT
        self.path = LinearPath.load(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "cyberrunner_dreamer",
                "data",
                "path_custom.pkl",
            )
        )

        self.latest_msg = None

        self.sub = self.create_subscription(
            StateEstimateSub,
            "/cyberrunner_state_estimation/estimate_subimg",
            self.on_state,
            10,
        )

        self.timer = self.create_timer(1.0 / 30.0, self.draw)
        self.get_logger().info("Overlay map viewer started.")

    def on_state(self, msg):
        self.latest_msg = msg

    def base_map(self):
        img = np.ones((VIEW_H, VIEW_W, 3), dtype=np.uint8) * 245

        # Board boundary
        cv2.rectangle(img, world_to_px(0, BOARD_H), world_to_px(BOARD_W, 0), (0, 0, 0), 2)

        # Holes
        hole_px = max(2, int(HOLE_R / BOARD_W * VIEW_W))
        for x, y in self.layout["holes"]:
            cv2.circle(img, world_to_px(x, y), hole_px, (40, 40, 40), -1)
            cv2.circle(img, world_to_px(x, y), hole_px, (0, 0, 0), 1)

        # Horizontal walls: [x1, x2, y]
        wall_px = max(2, int(WALL_R / BOARD_W * VIEW_W))
        for x1, x2, y in self.layout["walls_h"]:
            p1 = world_to_px(x1, y)
            p2 = world_to_px(x2, y)
            cv2.line(img, p1, p2, (120, 120, 120), wall_px * 2)

        # Vertical walls: [y1, y2, x]
        for y1, y2, x in self.layout["walls_v"]:
            p1 = world_to_px(x, y1)
            p2 = world_to_px(x, y2)
            cv2.line(img, p1, p2, (120, 120, 120), wall_px * 2)

        for x1, y1, x2, y2 in self.layout.get("walls_angled", []):
            cv2.line(
                img,
                world_to_px(x1, y1),
                world_to_px(x2, y2),
                (120, 120, 120),
                wall_px * 2,
            )

        # Path points
        pts = np.asarray(self.path.points, dtype=np.float32)
        if len(pts) > 1:
            pix = np.array([world_to_px(float(x), float(y)) for x, y in pts], dtype=np.int32)
            cv2.polylines(img, [pix], False, (0, 180, 255), 2)

        # Waypoints
        for i, (x, y) in enumerate(self.layout["waypoints"]):
            cv2.circle(img, world_to_px(x, y), 3, (255, 0, 0), -1)
            if i == 0:
                draw_text(img, "START", world_to_px(x, y), 0.45, 1)
            elif i == len(self.layout["waypoints"]) - 1:
                draw_text(img, "END", world_to_px(x, y), 0.45, 1)

        return img

    def draw(self):
        img = self.base_map()

        if self.latest_msg is not None:
            s = self.latest_msg.state

            # Convert centered state estimate to board coordinates.
            x = float(s.x_b) + float(OFFSET[0])
            y = float(s.y_b) + float(OFFSET[1])
            alpha = float(s.alpha)
            beta = float(s.beta)

            ball_ok = not (math.isnan(x) or math.isnan(y))

            if ball_ok:
                ball_px = max(4, int(0.006 / BOARD_W * VIEW_W))
                bx, by = world_to_px(x, y)

                # Ball
                cv2.circle(img, (bx, by), ball_px + 3, (255, 255, 255), -1)
                cv2.circle(img, (bx, by), ball_px, (0, 0, 255), -1)
                cv2.circle(img, (bx, by), ball_px + 3, (0, 0, 0), 1)

                # Closest path point
                idx, closest = self.path.closest_point(np.array([x, y], dtype=np.float32))
                if idx >= 0:
                    cx, cy = world_to_px(float(closest[0]), float(closest[1]))
                    cv2.circle(img, (cx, cy), 5, (0, 255, 0), -1)
                    cv2.line(img, (bx, by), (cx, cy), (0, 255, 0), 1)

                    progress = 100.0 * idx / max(1, self.path.num_points - 1)
                    draw_text(img, f"path idx: {idx}/{self.path.num_points}  progress: {progress:.1f}%", (15, 25))
                else:
                    draw_text(img, "OFF PATH", (15, 25))

                draw_text(img, f"x={x:.3f} m  y={y:.3f} m", (15, 50))
                draw_text(img, f"alpha={alpha:.3f}  beta={beta:.3f}", (15, 75))
            else:
                draw_text(img, "BALL NOT DETECTED", (15, 25), 0.7, 2)
        else:
            draw_text(img, "Waiting for /cyberrunner_state_estimation/estimate_subimg ...", (15, 25), 0.6, 1)

        cv2.imshow("CyberRunner Overlay Map View", img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            rclpy.shutdown()


def main():
    rclpy.init()
    node = OverlayMapView()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
