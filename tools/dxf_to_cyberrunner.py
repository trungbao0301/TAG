#!/usr/bin/env python3
"""Convert CyberRunner map and path DXFs into training and live-view assets.

The clean ``map.DXF`` supplies walls and holes. The connected centerline in
``path.DXF`` is followed exactly from START to GOAL; its original corners are
never replaced by a planned or smoothed route.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import importlib
import pprint
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree

from dxf_geometry import read_geometry


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MAP = ROOT / "cyberrunner_dreamer" / "data" / "map.DXF"
DEFAULT_PATH = ROOT / "cyberrunner_dreamer" / "data" / "path.DXF"
MAIN_LAYOUT = (
    ROOT
    / "cyberrunner_dreamer"
    / "cyberrunner_dreamer"
    / "cyberrunner_layout_custom.py"
)
THOMAS_LAYOUT = (
    ROOT
    / "cyberrunner_dreamer_thomas"
    / "cyberrunner_dreamer_thomas"
    / "cyberrunner_layout_custom.py"
)
MAIN_PICKLE = ROOT / "cyberrunner_dreamer" / "data" / "path_custom.pkl"
THOMAS_PICKLE = ROOT / "cyberrunner_dreamer_thomas" / "data" / "path_custom.pkl"
ESTIMATOR_LAYOUT = (
    ROOT
    / "cyberrunner_state_estimation"
    / "cyberrunner_state_estimation"
    / "core"
    / "maze_layout.py"
)


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _classify_lines(geometry, scale, origin, axis_tolerance_m=0.00015):
    walls_h = []
    walls_v = []
    walls_angled = []
    for line in geometry.lines:
        p1 = (np.asarray(line.start) - origin) * scale
        p2 = (np.asarray(line.end) - origin) * scale
        dx, dy = p2 - p1
        if abs(dy) <= axis_tolerance_m:
            walls_h.append(
                [float(min(p1[0], p2[0])), float(max(p1[0], p2[0])), float(np.mean([p1[1], p2[1]]))]
            )
        elif abs(dx) <= axis_tolerance_m:
            walls_v.append(
                [float(min(p1[1], p2[1])), float(max(p1[1], p2[1])), float(np.mean([p1[0], p2[0]]))]
            )
        else:
            walls_angled.append(
                [float(p1[0]), float(p1[1]), float(p2[0]), float(p2[1])]
            )
    return walls_h, walls_v, walls_angled


def _point_to_cell(point, resolution, height_cells):
    x = int(round(float(point[0]) / resolution))
    y_up = int(round(float(point[1]) / resolution))
    return height_cells - 1 - y_up, x


def _cell_to_point(cell, resolution, height_cells):
    row, col = cell
    return np.asarray(
        [col * resolution, (height_cells - 1 - row) * resolution],
        dtype=np.float64,
    )


def _draw_obstacles(layout, resolution, clearance):
    width = int(round(layout["board_width"] / resolution)) + 1
    height = int(round(layout["board_height"] / resolution)) + 1
    occupied = np.zeros((height, width), dtype=np.uint8)
    thickness = max(1, int(round(2.0 * clearance / resolution)) + 1)

    def pixel(point):
        row, col = _point_to_cell(point, resolution, height)
        return col, row

    for x1, x2, y in layout["walls_h"]:
        cv2.line(occupied, pixel((x1, y)), pixel((x2, y)), 255, thickness)
    for y1, y2, x in layout["walls_v"]:
        cv2.line(occupied, pixel((x, y1)), pixel((x, y2)), 255, thickness)
    for x1, y1, x2, y2 in layout.get("walls_angled", []):
        cv2.line(occupied, pixel((x1, y1)), pixel((x2, y2)), 255, thickness)
    for center, radius in zip(layout["holes"], layout["hole_radii"]):
        cv2.circle(
            occupied,
            pixel(center),
            max(1, int(round((radius + clearance) / resolution))),
            255,
            -1,
        )
    return occupied


def _path_waypoints_from_dxf(path_dxf, origin, start, goal):
    """Return the exact connected DXF line chain nearest START and GOAL."""
    geometry = read_geometry(path_dxf)
    if not geometry.lines:
        raise RuntimeError(f"{path_dxf}: path drawing contains no LINE entities")

    scale = 0.001

    def transform(point):
        return (np.asarray(point, dtype=np.float64) - origin) * scale

    # DXF writers can differ in insignificant trailing decimals. Quantization
    # is used only to connect coincident endpoints; returned coordinates remain
    # the original transformed DXF coordinates.
    def key(point):
        return tuple(np.round(point, decimals=9))

    graph = {}
    coordinates = {}
    for line in geometry.lines:
        point_a = transform(line.start)
        point_b = transform(line.end)
        key_a = key(point_a)
        key_b = key(point_b)
        coordinates.setdefault(key_a, point_a)
        coordinates.setdefault(key_b, point_b)
        distance = float(np.linalg.norm(point_b - point_a))
        graph.setdefault(key_a, []).append((key_b, distance))
        graph.setdefault(key_b, []).append((key_a, distance))

    start_key = min(
        coordinates,
        key=lambda item: float(np.linalg.norm(coordinates[item] - start)),
    )
    goal_key = min(
        coordinates,
        key=lambda item: float(np.linalg.norm(coordinates[item] - goal)),
    )

    queue = [(0.0, start_key)]
    best = {start_key: 0.0}
    previous = {}
    while queue:
        distance, current = heapq.heappop(queue)
        if distance != best.get(current):
            continue
        if current == goal_key:
            break
        for neighbor, segment_length in graph.get(current, []):
            candidate = distance + segment_length
            if candidate >= best.get(neighbor, float("inf")):
                continue
            best[neighbor] = candidate
            previous[neighbor] = current
            heapq.heappush(queue, (candidate, neighbor))

    if goal_key not in best:
        raise RuntimeError(
            f"{path_dxf}: no connected LINE chain joins the endpoints nearest "
            "START and GOAL"
        )

    chain = [goal_key]
    while chain[-1] != start_key:
        chain.append(previous[chain[-1]])
    chain.reverse()
    return np.asarray([coordinates[item] for item in chain], dtype=np.float64)


def _format_layout(layout, map_name, path_name):
    payload = pprint.pformat(layout, width=100, sort_dicts=False)
    return (
        f"# Generated from {map_name} and {path_name} by tools/dxf_to_cyberrunner.py\n"
        "# Do not hand-edit; rerun the converter so training and live view stay aligned.\n"
        f"cyberrunner_dxf_layout = {payload}\n"
    )


def _write_estimator_layout(layout, map_name, path_name):
    holes = pprint.pformat(layout["holes"], width=100)
    radii = pprint.pformat(layout["hole_radii"], width=100)
    return (
        f"# Generated from {map_name} and {path_name} by tools/dxf_to_cyberrunner.py\n"
        f"BOARD_WIDTH_M = {layout['board_width']!r}\n"
        f"BOARD_HEIGHT_M = {layout['board_height']!r}\n"
        f"HOLES_LOWER_LEFT_M = {holes}\n"
        f"HOLE_RADII_M = {radii}\n"
        f"SOURCE_MAP_SHA256 = {layout['source_map_sha256']!r}\n"
        f"SOURCE_PATH_SHA256 = {layout['source_path_sha256']!r}\n"
    )


def _build_path_pickle(
    package_root,
    module_name,
    output,
    layout,
    occupied_planner,
    path_tolerance,
):
    sys.path.insert(0, str(package_root))
    try:
        module = importlib.import_module(module_name)
        linear_path = module.LinearPath(
            np.asarray(layout["waypoints"], dtype=np.float32),
            distance=0.0002,
            board_width=layout["board_width"],
            board_height=layout["board_height"],
            wall_r=layout["ball_radius"],
        )
    finally:
        sys.path.pop(0)

    distance = linear_path.distance
    width = int(layout["board_width"] / distance) + 1
    height = int(layout["board_height"] / distance) + 1
    closest = np.full((height, width), -1, dtype=np.int32)
    tree = cKDTree(np.asarray(linear_path.points, dtype=np.float64))

    planner_h, planner_w = occupied_planner.shape
    chunk_rows = 64
    x_values = np.arange(width, dtype=np.float64) * distance
    for row0 in range(0, height, chunk_rows):
        row1 = min(height, row0 + chunk_rows)
        y_values = np.arange(row0, row1, dtype=np.float64) * distance
        xx, yy = np.meshgrid(x_values, y_values)
        query = np.column_stack((xx.ravel(), yy.ravel()))
        distances, indices = tree.query(query, workers=-1)
        valid = distances <= path_tolerance

        planner_cols = np.clip(
            np.rint(query[:, 0] / layout["planner_resolution_m"]).astype(int),
            0,
            planner_w - 1,
        )
        planner_rows = np.clip(
            planner_h
            - 1
            - np.rint(query[:, 1] / layout["planner_resolution_m"]).astype(int),
            0,
            planner_h - 1,
        )
        valid &= occupied_planner[planner_rows, planner_cols] == 0
        block = np.full(query.shape[0], -1, dtype=np.int32)
        block[valid] = indices[valid].astype(np.int32)
        closest[row0:row1] = block.reshape(row1 - row0, width)

    linear_path.closest_dim_x = width
    linear_path.closest_dim_y = height
    linear_path.closest_idx = closest
    output.parent.mkdir(parents=True, exist_ok=True)
    linear_path.save(output)
    return linear_path


def _parse_xy(value):
    parts = value.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("expected X,Y in millimeters")
    return np.asarray([float(parts[0]), float(parts[1])], dtype=np.float64) / 1000.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--map-dxf", type=Path, default=DEFAULT_MAP)
    parser.add_argument("--path-dxf", type=Path, default=DEFAULT_PATH)
    parser.add_argument(
        "--start-mm",
        type=_parse_xy,
        default=np.asarray([139.235, 220.409]) / 1000.0,
        help="policy START in clean-map coordinates, default: 139.235,220.409",
    )
    parser.add_argument(
        "--goal-mm",
        type=_parse_xy,
        default=np.asarray([250.602, 114.311]) / 1000.0,
        help="policy GOAL in clean-map coordinates, default: 250.602,114.311",
    )
    parser.add_argument("--ball-radius-mm", type=float, default=6.0)
    parser.add_argument("--clearance-mm", type=float, default=5.5)
    parser.add_argument("--planner-resolution-mm", type=float, default=0.5)
    parser.add_argument("--path-tolerance-mm", type=float, default=6.0)
    parser.add_argument(
        "--preview",
        type=Path,
        default=ROOT / "cyberrunner_dreamer" / "data" / "map_preview.png",
    )
    args = parser.parse_args()

    geometry = read_geometry(args.map_dxf)
    points = np.asarray(
        [point for line in geometry.lines for point in (line.start, line.end)],
        dtype=np.float64,
    )
    origin = points.min(axis=0)
    dimensions_mm = points.max(axis=0) - origin
    scale = 0.001
    walls_h, walls_v, walls_angled = _classify_lines(
        geometry, scale, origin
    )
    holes = [
        ((np.asarray(circle.center) - origin) * scale).astype(float).tolist()
        for circle in geometry.circles
    ]
    hole_radii = [float(circle.radius * scale) for circle in geometry.circles]
    planner_resolution = args.planner_resolution_mm / 1000.0
    layout = {
        "board_width": float(dimensions_mm[0] * scale),
        "board_height": float(dimensions_mm[1] * scale),
        "ball_radius": args.ball_radius_mm / 1000.0,
        "walls_h": walls_h,
        "walls_v": walls_v,
        "walls_angled": walls_angled,
        "holes": holes,
        "hole_radii": hole_radii,
        "source_map_sha256": _sha256(args.map_dxf),
        "source_path_sha256": _sha256(args.path_dxf),
        "planner_resolution_m": planner_resolution,
        "planner_clearance_m": args.clearance_mm / 1000.0,
    }
    occupied = _draw_obstacles(
        layout,
        planner_resolution,
        args.clearance_mm / 1000.0,
    )
    waypoints = _path_waypoints_from_dxf(
        args.path_dxf,
        origin,
        args.start_mm,
        args.goal_mm,
    )
    layout["waypoints"] = waypoints.astype(float).tolist()
    layout["start_requested"] = args.start_mm.astype(float).tolist()
    layout["goal_requested"] = args.goal_mm.astype(float).tolist()
    layout["start_planned"] = waypoints[0].astype(float).tolist()
    layout["goal_planned"] = waypoints[-1].astype(float).tolist()

    generated = _format_layout(layout, args.map_dxf.name, args.path_dxf.name)
    MAIN_LAYOUT.write_text(generated, encoding="utf-8")
    THOMAS_LAYOUT.write_text(generated, encoding="utf-8")
    ESTIMATOR_LAYOUT.write_text(
        _write_estimator_layout(layout, args.map_dxf.name, args.path_dxf.name),
        encoding="utf-8",
    )

    main_path = _build_path_pickle(
        ROOT / "cyberrunner_dreamer",
        "cyberrunner_dreamer.path",
        MAIN_PICKLE,
        layout,
        occupied,
        args.path_tolerance_mm / 1000.0,
    )
    thomas_path = _build_path_pickle(
        ROOT / "cyberrunner_dreamer_thomas",
        "cyberrunner_dreamer_thomas.path",
        THOMAS_PICKLE,
        layout,
        occupied,
        args.path_tolerance_mm / 1000.0,
    )

    args.preview.parent.mkdir(parents=True, exist_ok=True)
    preview = cv2.cvtColor(occupied, cv2.COLOR_GRAY2BGR)
    preview[occupied == 0] = (245, 245, 245)
    preview[occupied != 0] = (70, 70, 70)
    route_pixels = []
    for point in waypoints:
        row, col = _point_to_cell(point, planner_resolution, occupied.shape[0])
        route_pixels.append((col, row))
    cv2.polylines(
        preview,
        [np.asarray(route_pixels, dtype=np.int32)],
        False,
        (0, 150, 255),
        max(1, int(round(1.5 / args.planner_resolution_mm))),
        cv2.LINE_AA,
    )
    for index, point in enumerate(route_pixels):
        color = (0, 200, 0) if index == 0 else (0, 0, 255)
        cv2.circle(preview, point, 3, color, -1)
    cv2.imwrite(str(args.preview), preview)

    print(
        f"Generated {len(walls_h)} horizontal, {len(walls_v)} vertical, "
        f"{len(walls_angled)} angled walls, and {len(holes)} holes."
    )
    print(
        f"Loaded {len(waypoints)} exact DXF waypoints and "
        f"{main_path.num_points} path points "
        f"from {layout['start_planned']} to {layout['goal_planned']}."
    )
    print(f"Wrote {MAIN_LAYOUT}, {MAIN_PICKLE}, and {args.preview}.")
    print(f"Wrote Thomas equivalents with {thomas_path.num_points} path points.")


if __name__ == "__main__":
    main()
