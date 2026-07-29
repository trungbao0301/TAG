#!/usr/bin/env python3

import os
import sys
import math
import argparse
import time
import cv2
import numpy as np

# When this script is run directly from the repository, an older ROS overlay can
# otherwise win the import and hide files that only exist in this checkout.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_DREAMER_ROOT = os.path.join(SCRIPT_DIR, "tag_dreamer")
LOCAL_DREAMER_PACKAGE = os.path.join(LOCAL_DREAMER_ROOT, "tag_dreamer")
if os.path.isfile(os.path.join(LOCAL_DREAMER_PACKAGE, "__init__.py")):
    sys.path.insert(0, LOCAL_DREAMER_ROOT)

import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

from ament_index_python.packages import get_package_share_directory
from tag_interfaces.msg import StateEstimate

from tag_dreamer.tag_layout_custom import tag_dxf_layout
from tag_dreamer.path import LinearPath


LAYOUT = tag_dxf_layout
BOARD_W = float(LAYOUT["board_width"])
BOARD_H = float(LAYOUT["board_height"])
BALL_R = float(LAYOUT.get("ball_radius", 0.006))

# Half thickness of the boundary walls, derived instead of guessed.
#
# board_width/board_height are the wall CENTRELINE span: layout["walls_h"] has
# segments at y = 0 and y = board_height exactly, walls_v at x = 0 and
# x = board_width, all modelled as zero-thickness lines. So the marble centre
# limit is half the span minus (wall half thickness + ball radius), and the
# wall half thickness is not in the layout.
#
# It is bounded, though: a hole is cut in the floor and cannot overlap a wall,
# so the closest any hole edge comes to a boundary centreline is an upper bound.
# Measured on this layout: left 1.2, right 2.0, bottom 1.0, top 0.7 mm. The
# previous hardcoded 0.0025 exceeded that bound by 1.8 mm, which made REACH_Y
# 1.8 mm too tight and manufactured most of the "UNREACHABLE" overshoot.
_HOLE_XY = np.asarray(LAYOUT["holes"], dtype=float)
_HOLE_R = np.asarray(LAYOUT["hole_radii"], dtype=float)
WALL_R = max(
    0.0,
    float(
        min(
            (_HOLE_XY[:, 0] - _HOLE_R).min(),
            (BOARD_W - (_HOLE_XY[:, 0] + _HOLE_R)).min(),
            (_HOLE_XY[:, 1] - _HOLE_R).min(),
            (BOARD_H - (_HOLE_XY[:, 1] + _HOLE_R)).min(),
        )
    ),
)

# The view used to be exactly the playable rectangle, so anything reported
# outside the board was silently clipped by OpenCV and a marble 15 mm past the
# edge looked identical to one sitting on it. Render a margin so out-of-board
# positions are actually visible.
MARGIN_M = 0.022
SPAN_W = BOARD_W + 2.0 * MARGIN_M
SPAN_H = BOARD_H + 2.0 * MARGIN_M

VIEW_W = 900
VIEW_H = int(VIEW_W * SPAN_H / SPAN_W)
SIDE_W = 260
CANVAS_W = VIEW_W + SIDE_W
OFFSET = np.array([BOARD_W, BOARD_H], dtype=np.float32) / 2.0

# Moving corner-dot centre spacing (PlatePoseEstimator.C2C_X / C2C_Y). The state
# origin is the CENTROID OF THESE DOTS, not necessarily the playable centre --
# OFFSET above assumes the two coincide. Drawing the dot quad makes that
# assumption checkable: it should sit concentrically outside the playable
# rectangle by (269-259)/2 = 5 mm in x and (237-229)/2 = 4 mm in y.
# Fitted from 40 tilted views / 745 hole observations by
# tools/fit_marker_geometry.py, NOT the ETH original's 0.269 x 0.237. Those were
# inherited nominal values for different hardware; this board runs a custom maze.
# Profiling the dot-plane height h against the known DXF hole positions gives a
# clear minimum at h = 10 mm (median residual 2.13 mm at h=0, 1.49 mm at h=10,
# 1.92 mm at h=20), independently reproducing a 1 cm ruler measurement, and at
# that optimum the spacing is 249.2 x 222.3 mm. Median hole residual over the
# whole set improves 5.67 -> 1.49 mm versus the old constants.
C2C_X = 0.2492
C2C_Y = 0.2223

# A marble centre cannot get closer to the playable edge than wall + radius.
REACH_X = BOARD_W / 2.0 - (WALL_R + BALL_R)
REACH_Y = BOARD_H / 2.0 - (WALL_R + BALL_R)

WINDOW_NAME = "TAG Overlay Map View"

# BGR colors for OpenCV
COLOR_BG = (245, 245, 245)
COLOR_BORDER = (0, 0, 0)
COLOR_HOLE = (30, 30, 30)
COLOR_WALL = (120, 120, 120)
COLOR_PATH = (0, 160, 255)
COLOR_PATH_CLOSEST = (0, 255, 0)

COLOR_WP_FUTURE = (255, 0, 0)      # Blue
COLOR_WP_PASSED = (0, 200, 0)      # Green
COLOR_WP_NEXT = (0, 220, 255)      # Yellow
COLOR_BALL = (0, 0, 255)           # Red
COLOR_CHECKPOINT_RADIUS = (0, 190, 255)
COLOR_DANGER_ZONE = (80, 80, 255)
COLOR_SEGMENT_GUARD = (180, 0, 180)
COLOR_DOT_QUAD = (200, 120, 0)      # corner-dot quad outline
COLOR_REACH_LIMIT = (0, 140, 0)     # marble-centre reachable limit


def world_to_px(x, y):
    px = int(round((x + MARGIN_M) / SPAN_W * VIEW_W))
    py = int(round((SPAN_H - (y + MARGIN_M)) / SPAN_H * VIEW_H))
    return px, py


def meters_to_px(distance):
    """Convert a physical map distance to display pixels."""
    return max(1, int(round(distance / SPAN_W * VIEW_W)))


def draw_label(img, text, org, scale=0.50):
    """Readable small text with white outline."""
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 1, cv2.LINE_AA)


def draw_side_panel(canvas, lines, next_target=None):
    """Draw status + legend in the right side panel, not on top of the map."""
    x0 = VIEW_W

    # Panel background.
    cv2.rectangle(canvas, (x0, 0), (CANVAS_W - 1, VIEW_H - 1), (250, 250, 250), -1)
    cv2.line(canvas, (x0, 0), (x0, VIEW_H - 1), (40, 40, 40), 2)

    x = x0 + 18
    y = 34

    cv2.putText(canvas, "STATUS", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2, cv2.LINE_AA)
    y += 28

    if not lines:
        lines = ["No status"]

    for line in lines[:8]:
        cv2.putText(canvas, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 1, cv2.LINE_AA)
        y += 24

    y += 18
    cv2.putText(canvas, "LEGEND", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2, cv2.LINE_AA)
    y += 28

    legend_items = [
        ("passed waypoint", COLOR_WP_PASSED),
        ("next target", COLOR_WP_NEXT),
        ("future waypoint", COLOR_WP_FUTURE),
        ("ball", COLOR_BALL),
        ("closest path", COLOR_PATH_CLOSEST),
    ]

    for name, color in legend_items:
        cv2.circle(canvas, (x + 9, y - 5), 7, color, -1)
        cv2.circle(canvas, (x + 9, y - 5), 8, (0, 0, 0), 1)
        cv2.putText(canvas, name, (x + 28, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 1, cv2.LINE_AA)
        y += 25

    y += 18
    cv2.putText(canvas, "CONTROLS", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2, cv2.LINE_AA)
    y += 28
    cv2.putText(canvas, "q = quit", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 1, cv2.LINE_AA)


class OverlayNode(Node):
    # A state older than this is not shown at all. Without an expiry the overlay
    # keeps drawing the last received ball forever, so an estimator that stalls,
    # crashes, or gets restarted looks exactly like a marble sitting still.
    STALE_AFTER_SEC = 0.25

    def __init__(self):
        super().__init__("overlay_map_view_simple")
        self.latest = None
        self.latest_time = None

        # The overlay only needs the newest estimate; dropped display frames are
        # preferable to retransmitting stale states.
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.create_subscription(
            StateEstimate,
            "/tag_state_estimation/estimate",
            self.cb,
            qos,
        )
        self.get_logger().info("Overlay subscriber started.")

    def cb(self, msg):
        self.latest = msg
        self.latest_time = time.monotonic()

    def is_stale(self):
        return (
            self.latest_time is None
            or (time.monotonic() - self.latest_time) > self.STALE_AFTER_SEC
        )

    def stale_for(self):
        if self.latest_time is None:
            return float("inf")
        return time.monotonic() - self.latest_time


def draw_waypoint_marker(img, px, py, color, number, radius=10, ring_thickness=1):
    """Draw waypoint using color only for status. No PASS/NEXT words."""
    cv2.circle(img, (px, py), radius + 3, (255, 255, 255), -1)
    cv2.circle(img, (px, py), radius, color, -1)
    cv2.circle(img, (px, py), radius + 3, (0, 0, 0), ring_thickness)

    # Put the waypoint number inside the circle.
    label = str(number)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    cv2.putText(
        img,
        label,
        (px - tw // 2, py + th // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )


def make_waypoint_path_indices(path):
    indices = [0]
    total = 0
    for i in range(1, path.orig_waypoints.shape[0]):
        segment = path.orig_waypoints[i] - path.orig_waypoints[i - 1]
        total += int(np.floor(np.linalg.norm(segment) / path.distance)) + 1
        indices.append(min(total, path.num_points - 1))
    return np.asarray(indices, dtype=np.int32)


def make_base_map():
    img = np.ones((VIEW_H, VIEW_W, 3), dtype=np.uint8) * np.array(COLOR_BG, dtype=np.uint8)

    layout = LAYOUT

    walls_h = np.asarray(layout["walls_h"], dtype=np.float32)
    walls_v = np.asarray(layout["walls_v"], dtype=np.float32)
    walls_angled = np.asarray(layout.get("walls_angled", []), dtype=np.float32)
    holes = np.asarray(layout["holes"], dtype=np.float32)
    hole_radii = np.asarray(
        layout.get("hole_radii", [0.0075] * len(holes)),
        dtype=np.float32,
    )
    waypoints = np.asarray(layout["waypoints"], dtype=np.float32)
    danger_zones = layout.get("danger_zones", [])
    danger_lines = layout.get("danger_lines", [])
    segment_guards = layout.get("segment_guards", [])

    # Playable-area border, in world coords now that the view has a margin.
    cv2.rectangle(img, world_to_px(0.0, 0.0), world_to_px(BOARD_W, BOARD_H),
                  COLOR_BORDER, 2)

    # Corner-dot quad, centred on the STATE origin. If the state origin really is
    # the playable centre, this sits concentrically 5 mm (x) / 4 mm (y) outside
    # the border above. If the marble is ever seen outside the border but inside
    # this quad, the two are NOT concentric and OFFSET is wrong.
    dq0 = world_to_px(float(OFFSET[0]) - C2C_X / 2.0, float(OFFSET[1]) - C2C_Y / 2.0)
    dq1 = world_to_px(float(OFFSET[0]) + C2C_X / 2.0, float(OFFSET[1]) + C2C_Y / 2.0)
    cv2.rectangle(img, dq0, dq1, COLOR_DOT_QUAD, 1)
    draw_label(img, "corner-dot quad (state origin)", (dq0[0] + 4, dq1[1] - 6), 0.42)

    # Limit the marble CENTRE can physically reach (wall + ball radius inset).
    rl0 = world_to_px(float(OFFSET[0]) - REACH_X, float(OFFSET[1]) - REACH_Y)
    rl1 = world_to_px(float(OFFSET[0]) + REACH_X, float(OFFSET[1]) + REACH_Y)
    cv2.rectangle(img, rl0, rl1, COLOR_REACH_LIMIT, 1)

    # Holes
    for (x, y), radius in zip(holes, hole_radii):
        cv2.circle(
            img,
            world_to_px(float(x), float(y)),
            meters_to_px(float(radius)),
            COLOR_HOLE,
            -1,
        )

    # Segment guards and danger zones, so the protected pocket is visible.
    for guard in segment_guards:
        x_min = guard.get("x_min")
        x_max = guard.get("x_max")
        y_min = guard.get("y_min")
        y_max = guard.get("y_max")
        if None in (x_min, x_max, y_min, y_max):
            continue
        gx1, gy1 = world_to_px(float(x_min), float(y_max))
        gx2, gy2 = world_to_px(float(x_max), float(y_min))
        overlay = img.copy()
        cv2.rectangle(overlay, (gx1, gy1), (gx2, gy2), COLOR_SEGMENT_GUARD, -1)
        cv2.addWeighted(overlay, 0.14, img, 0.86, 0.0, img)
        cv2.rectangle(img, (gx1, gy1), (gx2, gy2), COLOR_SEGMENT_GUARD, 2)

    for zone in danger_zones:
        center = zone.get("center")
        radius = zone.get("radius")
        if center is None or radius is None:
            continue
        zx, zy = world_to_px(float(center[0]), float(center[1]))
        zr = max(1, int(round(float(radius) / SPAN_W * VIEW_W)))
        overlay = img.copy()
        cv2.circle(overlay, (zx, zy), zr, COLOR_DANGER_ZONE, -1)
        cv2.addWeighted(overlay, 0.18, img, 0.82, 0.0, img)
        cv2.circle(img, (zx, zy), zr, COLOR_DANGER_ZONE, 2)

    for line in danger_lines:
        p1 = line.get("p1")
        p2 = line.get("p2")
        if p1 is None or p2 is None:
            continue
        width = float(line.get("width", 0.001))
        thickness = max(2, int(round(width / SPAN_W * VIEW_W * 2.0)))
        cv2.line(
            img,
            world_to_px(float(p1[0]), float(p1[1])),
            world_to_px(float(p2[0]), float(p2[1])),
            COLOR_DANGER_ZONE,
            thickness,
            cv2.LINE_AA,
        )

    # Horizontal walls: x1, x2, y
    for x1, x2, y in walls_h:
        cv2.line(
            img,
            world_to_px(float(x1), float(y)),
            world_to_px(float(x2), float(y)),
            COLOR_WALL,
            6,
        )

    # Vertical walls: y1, y2, x
    for y1, y2, x in walls_v:
        cv2.line(
            img,
            world_to_px(float(x), float(y1)),
            world_to_px(float(x), float(y2)),
            COLOR_WALL,
            6,
        )

    # Angled walls: x1, y1, x2, y2
    for x1, y1, x2, y2 in walls_angled:
        cv2.line(
            img,
            world_to_px(float(x1), float(y1)),
            world_to_px(float(x2), float(y2)),
            COLOR_WALL,
            6,
        )

    # Load official saved path if available.
    try:
        local_path_file = os.path.join(LOCAL_DREAMER_ROOT, "data", "path_custom.pkl")
        if os.path.isfile(local_path_file):
            path_file = local_path_file
        else:
            share = get_package_share_directory("tag_dreamer")
            path_file = os.path.join(share, "path_custom.pkl")
        path = LinearPath.load(path_file)
        pts = np.asarray(path.points, dtype=np.float32)
        pix = np.array([world_to_px(float(x), float(y)) for x, y in pts], dtype=np.int32)
        cv2.polylines(img, [pix], False, COLOR_PATH, 2)
    except Exception as e:
        print("Could not load saved path, drawing waypoints only:", e)
        path = None

    # Precompute waypoint pixel locations.
    waypoint_px = np.array([world_to_px(float(x), float(y)) for x, y in waypoints], dtype=np.int32)

    # Match the checkpoint indices used by the Thomas training environments.
    if path is not None:
        waypoint_path_indices = make_waypoint_path_indices(path)
    else:
        waypoint_path_indices = np.full(len(waypoints), -1, dtype=np.int32)

    # Static labels only for START and END. No repeated W/PASS/NEXT clutter.
    if len(waypoint_px) > 0:
        sx, sy = waypoint_px[0]
        ex, ey = waypoint_px[-1]
        draw_label(img, "START", (sx + 12, sy - 8), scale=0.48)
        draw_label(img, "END", (ex + 12, ey - 8), scale=0.48)

    return img, path, waypoints, waypoint_px, waypoint_path_indices, danger_zones, segment_guards


def find_next_waypoint(idx, waypoint_path_indices):
    """First waypoint whose path index is ahead of the current ball path index."""
    for wi, widx in enumerate(waypoint_path_indices):
        if widx >= 0 and widx > idx:
            return wi
    return None


def track_path_point(path, point):
    """Find progress from the current ball position, independent of history."""
    point = np.asarray(point, dtype=np.float32)
    idx, closest = path.closest_point(point)
    if idx < 0:
        # The precomputed closest-point map deliberately contains invalid cells
        # near walls and holes. Use the nearest geometric point for display.
        idx = int(np.argmin(np.linalg.norm(path.points - point, axis=1)))
        closest = path.points[idx]
    return idx, closest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--display_hz",
        type=float,
        default=15.0,
        help="Maximum overlay redraw rate.",
    )
    parser.add_argument(
        "--checkpoint_radius_m",
        type=float,
        default=float(os.environ.get("TAG_CHECKPOINT_RADIUS_M", "0.010")),
        help="Radius around the current checkpoint that counts as a pass.",
    )
    parser.add_argument(
        "--reach_warning",
        action="store_true",
        help="Flag marble positions outside the physically reachable rectangle. "
        "Off by default: the check assumes the state origin is exactly the "
        "playable centre, so a few mm of calibration bias trips it on "
        "positions that are otherwise correct.",
    )
    args, ros_args = parser.parse_known_args()

    display_hz = max(1.0, args.display_hz)
    frame_period = 1.0 / display_hz
    checkpoint_radius_px = max(
        1, int(round(max(0.0, args.checkpoint_radius_m) / SPAN_W * VIEW_W))
    )

    rclpy.init(args=ros_args)
    node = OverlayNode()

    base, path, waypoints, waypoint_px, waypoint_path_indices, danger_zones, segment_guards = make_base_map()

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, CANVAS_W, VIEW_H)
    cv2.moveWindow(WINDOW_NAME, 100, 80)

    print("Overlay window should be open now. Press q to quit.")
    print("Waypoint colors: green=passed, yellow=next target, blue=future.")
    print(f"Overlay display limited to {display_hz:.1f} FPS.")

    tracked_idx = None
    # Session extremes, to tell a SCALE error from an ORIGIN OFFSET. Roll the
    # marble against all four walls: symmetric overshoot on both ends of an axis
    # means the marker spacing constant for that axis is wrong, while one end
    # over by d and the other short by d means the dot-quad centroid is offset
    # from the playable centre by d (i.e. OFFSET is wrong, not the scale).
    extremes = {"x": [math.inf, -math.inf], "y": [math.inf, -math.inf]}

    while rclpy.ok():
        frame_start = time.monotonic()

        # Queue=1 keeps only the newest state while the display is throttled.
        rclpy.spin_once(node, timeout_sec=0.0)

        img = base.copy()
        panel_lines = []

        # Defaults if no valid path/ball data yet.
        current_idx = tracked_idx
        next_target = (
            find_next_waypoint(tracked_idx, waypoint_path_indices)
            if tracked_idx is not None
            else (0 if len(waypoints) > 0 else None)
        )

        if node.latest is None:
            panel_lines.append("Waiting for state topic...")
        elif node.is_stale():
            # Draw nothing rather than a ball frozen at its last position.
            panel_lines.append("NO DATA")
            panel_lines.append(f"estimator silent {node.stale_for():.1f}s")
        else:
            s = node.latest

            state_x = float(s.x_b)
            state_y = float(s.y_b)

            if math.isnan(state_x) or math.isnan(state_y):
                panel_lines.append("BALL NOT DETECTED")
            else:
                # StateEstimate uses a board-centered origin. The DXF map and
                # waypoint path use a lower-left origin. Keep both names so
                # the status panel does not make the display conversion look
                # like an estimator/calibration error.
                map_x, map_y = (
                    np.array([state_x, state_y], dtype=np.float32) + OFFSET
                ).astype(float)
                bx, by = world_to_px(map_x, map_y)

                # Ball marker.
                cv2.circle(img, (bx, by), 14, (255, 255, 255), -1)
                cv2.circle(img, (bx, by), 9, COLOR_BALL, -1)
                cv2.circle(img, (bx, by), 14, (0, 0, 0), 1)

                panel_lines.append(
                    f"state(center): x={state_x:.3f}, y={state_y:.3f} m"
                )
                panel_lines.append(
                    f"map(lower-left): x={map_x:.3f}, y={map_y:.3f} m"
                )

                # Flag a position the marble centre cannot physically occupy.
                # Either the estimate is biased or OFFSET is wrong -- the
                # corner-dot quad on the map tells you which.
                for axis, value in (("x", state_x), ("y", state_y)):
                    extremes[axis][0] = min(extremes[axis][0], value)
                    extremes[axis][1] = max(extremes[axis][1], value)
                for axis, reach in (("x", REACH_X), ("y", REACH_Y)):
                    low, high = extremes[axis]
                    if math.isfinite(low) and math.isfinite(high):
                        panel_lines.append(
                            "%s seen %+.1f..%+.1f vs +-%.1f mm"
                            % (axis, low * 1000, high * 1000, reach * 1000)
                        )

                over_x = abs(state_x) - REACH_X
                over_y = abs(state_y) - REACH_Y
                if args.reach_warning and (over_x > 0.0 or over_y > 0.0):
                    worst = "y" if over_y >= over_x else "x"
                    panel_lines.append(
                        f"!! UNREACHABLE {worst} by "
                        f"{max(over_x, over_y) * 1000:.1f} mm"
                    )
                    cv2.circle(img, (bx, by), 22, (0, 0, 255), 2)
                panel_lines.append(f"alpha={float(s.alpha):.3f}, beta={float(s.beta):.3f}")

                if path is not None:
                    idx, closest = track_path_point(
                        path,
                        np.array([map_x, map_y], dtype=np.float32),
                    )

                    if idx >= 0:
                        tracked_idx = int(idx)
                        current_idx = tracked_idx

                        cx, cy = world_to_px(float(closest[0]), float(closest[1]))
                        cv2.circle(img, (cx, cy), 6, COLOR_PATH_CLOSEST, -1)
                        cv2.line(img, (bx, by), (cx, cy), COLOR_PATH_CLOSEST, 2)

                        progress = 100.0 * tracked_idx / max(1, path.num_points - 1)
                        panel_lines.append(f"path: {progress:.1f}%")

                        next_target = find_next_waypoint(tracked_idx, waypoint_path_indices)

                        if next_target is not None:
                            ntx, nty = waypoints[next_target]
                            ntx_px, nty_px = waypoint_px[next_target]

                            # Show the same acceptance radius used by training.
                            radius_layer = img.copy()
                            cv2.circle(
                                radius_layer,
                                (int(ntx_px), int(nty_px)),
                                checkpoint_radius_px,
                                COLOR_CHECKPOINT_RADIUS,
                                -1,
                            )
                            cv2.addWeighted(radius_layer, 0.16, img, 0.84, 0.0, img)
                            cv2.circle(
                                img,
                                (int(ntx_px), int(nty_px)),
                                checkpoint_radius_px,
                                COLOR_CHECKPOINT_RADIUS,
                                2,
                            )

                            # Draw arrow from ball to next target.
                            cv2.arrowedLine(
                                img,
                                (bx, by),
                                (int(ntx_px), int(nty_px)),
                                COLOR_WP_NEXT,
                                3,
                                tipLength=0.08,
                            )

                            dist_next = float(np.linalg.norm(
                                np.array([map_x, map_y], dtype=np.float32)
                                - np.array([float(ntx), float(nty)], dtype=np.float32)
                            ))
                            panel_lines.append(f"target: W{next_target}, {dist_next*1000:.0f} mm")
                            panel_lines.append(
                                f"pass radius: {args.checkpoint_radius_m*1000:.0f} mm"
                            )
                        else:
                            panel_lines.append("target: FINISH")
                else:
                    # Fallback: target first waypoint not yet reached is not available.
                    panel_lines.append("path file not loaded")

        # Draw waypoints after arrow so waypoint circles stay visible.
        for wi, (px, py) in enumerate(waypoint_px):
            px, py = int(px), int(py)

            if current_idx is not None and path is not None:
                widx = waypoint_path_indices[wi]

                if widx >= 0 and widx <= current_idx:
                    color = COLOR_WP_PASSED       # passed: green
                    radius = 10
                    ring = 1
                elif wi == next_target:
                    color = COLOR_WP_NEXT         # next: yellow
                    radius = 14
                    ring = 2
                else:
                    color = COLOR_WP_FUTURE       # future: blue
                    radius = 10
                    ring = 1
            else:
                color = COLOR_WP_NEXT if wi == next_target else COLOR_WP_FUTURE
                radius = 14 if wi == next_target else 10
                ring = 2 if wi == next_target else 1

            draw_waypoint_marker(img, px, py, color, wi, radius=radius, ring_thickness=ring)

        # Put the map on the left and the legend/status panel on the right.
        # Nothing is drawn over the maze area.
        canvas = np.ones((VIEW_H, CANVAS_W, 3), dtype=np.uint8) * 250
        canvas[:, :VIEW_W] = img
        draw_side_panel(canvas, panel_lines, next_target=next_target)

        cv2.imshow(WINDOW_NAME, canvas)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        remaining = frame_period - (time.monotonic() - frame_start)
        if remaining > 0:
            time.sleep(remaining)

    cv2.destroyAllWindows()
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == "__main__":
    main()
