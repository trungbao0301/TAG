"""Reject camera detections that coincide with known maze holes."""

import time

import cv2
import numpy as np

from .maze_layout import (
    BOARD_HEIGHT_M,
    BOARD_WIDTH_M,
    HOLE_RADII_M,
    HOLES_LOWER_LEFT_M,
)

# Coordinates are generated from the currently installed CyberRunner DXF maze.
# The layout uses a lower-left origin, while state estimation uses a board-
# centered origin. Keep this small geometry-only module independent of Dreamer
# so the ROS estimator does not need to import the training package.
MOVING_MARKER_SPACING_X_M = 0.269
MOVING_MARKER_SPACING_Y_M = 0.237

HOLES_LOWER_LEFT_M = np.asarray(HOLES_LOWER_LEFT_M, dtype=np.float32)
HOLE_RADII_M = np.asarray(HOLE_RADII_M, dtype=np.float32)

HOLES_CENTERED_M = HOLES_LOWER_LEFT_M - np.asarray(
    [BOARD_WIDTH_M / 2.0, BOARD_HEIGHT_M / 2.0], dtype=np.float32
)

MARKERS_CENTERED_M = np.asarray(
    [
        [-MOVING_MARKER_SPACING_X_M / 2.0, -MOVING_MARKER_SPACING_Y_M / 2.0],
        [+MOVING_MARKER_SPACING_X_M / 2.0, -MOVING_MARKER_SPACING_Y_M / 2.0],
        [+MOVING_MARKER_SPACING_X_M / 2.0, +MOVING_MARKER_SPACING_Y_M / 2.0],
        [-MOVING_MARKER_SPACING_X_M / 2.0, +MOVING_MARKER_SPACING_Y_M / 2.0],
    ],
    dtype=np.float32,
)


class TimedHoleRejector:
    """Reject only after one hole contains a candidate continuously."""

    def __init__(self, delay_sec=2.0):
        self.delay_sec = max(0.0, float(delay_sec))
        self.active_hole = None
        self.started_at = None
        self.elapsed_sec = 0.0

    def reset(self):
        self.active_hole = None
        self.started_at = None
        self.elapsed_sec = 0.0

    def update(self, hole_index, now=None):
        now = time.monotonic() if now is None else float(now)
        if hole_index is None:
            self.reset()
            return False, 0.0
        hole_index = int(hole_index)
        if hole_index != self.active_hole or self.started_at is None:
            self.active_hole = hole_index
            self.started_at = now
        self.elapsed_sec = max(0.0, now - self.started_at)
        return self.elapsed_sec >= self.delay_sec, self.elapsed_sec


def candidate_hole_index(candidate_rc, moving_corners_rc, margin_m=0.0025):
    """Return the containing hole index, or ``None`` when the candidate is safe.

    Both image inputs use the estimator's internal ``[row, column]`` order.
    A frame-local homography makes the exclusion zones follow the tilting board.
    Invalid/missing marker geometry fails open so marker loss alone cannot make
    every marble observation disappear.
    """

    candidate_rc = np.asarray(candidate_rc, dtype=np.float32)
    corners_rc = np.asarray(moving_corners_rc, dtype=np.float32)
    if candidate_rc.shape != (2,) or corners_rc.shape != (4, 2):
        return None
    if not np.all(np.isfinite(candidate_rc)) or not np.all(np.isfinite(corners_rc)):
        return None

    try:
        image_xy = corners_rc[:, ::-1].astype(np.float32)
        image_to_board = cv2.getPerspectiveTransform(image_xy, MARKERS_CENTERED_M)
        candidate_xy = candidate_rc[::-1].reshape(1, 1, 2)
        board_xy = cv2.perspectiveTransform(candidate_xy, image_to_board)[0, 0]
    except cv2.error:
        return None

    if not np.all(np.isfinite(board_xy)):
        return None
    distances = np.linalg.norm(HOLES_CENTERED_M - board_xy, axis=1)
    index = int(np.argmin(distances))
    radius_m = float(HOLE_RADII_M[index]) + max(0.0, float(margin_m))
    return index if float(distances[index]) <= radius_m else None


def project_holes_to_image(moving_corners_rc, margin_m=0.0025):
    """Project hole centers and guarded radii into image ``[x, y]`` pixels.

    Returns ``(centers_xy, radii_px)`` or ``(None, None)`` when the current
    moving-marker geometry is unusable.
    """

    corners_rc = np.asarray(moving_corners_rc, dtype=np.float32)
    if corners_rc.shape != (4, 2) or not np.all(np.isfinite(corners_rc)):
        return None, None
    try:
        board_to_image = cv2.getPerspectiveTransform(
            MARKERS_CENTERED_M, corners_rc[:, ::-1].astype(np.float32)
        )
        radius_m = HOLE_RADII_M + max(0.0, float(margin_m))
        centers = HOLES_CENTERED_M.reshape(-1, 1, 2)
        x_edges = (
            HOLES_CENTERED_M + np.column_stack((radius_m, np.zeros_like(radius_m)))
        ).reshape(-1, 1, 2)
        y_edges = (
            HOLES_CENTERED_M + np.column_stack((np.zeros_like(radius_m), radius_m))
        ).reshape(-1, 1, 2)
        centers_xy = cv2.perspectiveTransform(centers, board_to_image)[:, 0]
        x_edges_xy = cv2.perspectiveTransform(x_edges, board_to_image)[:, 0]
        y_edges_xy = cv2.perspectiveTransform(y_edges, board_to_image)[:, 0]
    except cv2.error:
        return None, None
    radii_px = np.maximum(
        np.linalg.norm(x_edges_xy - centers_xy, axis=1),
        np.linalg.norm(y_edges_xy - centers_xy, axis=1),
    )
    if not np.all(np.isfinite(centers_xy)) or not np.all(np.isfinite(radii_px)):
        return None, None
    return centers_xy, radii_px
