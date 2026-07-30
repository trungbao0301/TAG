"""The masked HSV marble source.

The whole reason an HSV marble candidate is allowed to stand on its own is that
the search is clipped to the maze interior, which the eight blue reference dots
sit outside. So the mask is the thing worth pinning down: if it ever admits a dot
the detector reports the marble parked on the rim, and if it ever clips the
marble it throws away the frames the pairing exists to rescue.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

cv2 = pytest.importorskip("cv2")

from tag_state_estimation.core import hsv_marble as H  # noqa: E402
from tag_state_estimation.core.ai_map_state import (  # noqa: E402
    MARBLE_RADIUS_M,
    MOVING_MARKERS_CENTERED_M as DOTS,
)
from tag_state_estimation.core.maze_layout import (  # noqa: E402
    BOARD_HEIGHT_M,
    BOARD_WIDTH_M,
)

SCALE = 500.0 / 0.269          # px per metre, a 500 px wide dot ring
ORIGIN = np.array([320.0, 200.0])


def dot_quad():
    """The four moving dots in image [row, col], as the trackers carry them."""
    image_xy = (DOTS * SCALE + ORIGIN).astype(np.float32)
    return image_xy[:, ::-1], image_xy


def inside(polygon, point):
    return cv2.pointPolygonTest(
        polygon.astype(np.float32), (float(point[0]), float(point[1])), False
    ) >= 0


def board_to_px(x_m, y_m):
    return np.asarray([x_m, y_m]) * SCALE + ORIGIN


def test_mask_excludes_every_reference_dot():
    moving_rc, image_xy = dot_quad()
    polygon = H.maze_polygon_px(moving_rc)
    assert polygon is not None
    for dot in image_xy:
        assert not inside(polygon, dot)


def test_mask_admits_the_marble_at_its_closest_legal_approach():
    """The centre can sit one radius from the maze edge and must still count."""
    moving_rc, _ = dot_quad()
    polygon = H.maze_polygon_px(moving_rc)
    for x_m, y_m in (
        (0.0, 0.0),
        (BOARD_WIDTH_M / 2.0 - MARBLE_RADIUS_M, 0.0),
        (-(BOARD_WIDTH_M / 2.0 - MARBLE_RADIUS_M), 0.0),
        (0.0, BOARD_HEIGHT_M / 2.0 - MARBLE_RADIUS_M),
        (0.0, -(BOARD_HEIGHT_M / 2.0 - MARBLE_RADIUS_M)),
    ):
        assert inside(polygon, board_to_px(x_m, y_m))


def test_mask_is_smaller_than_the_dot_ring():
    moving_rc, image_xy = dot_quad()
    polygon = H.maze_polygon_px(moving_rc)
    assert cv2.contourArea(polygon.astype(np.float32)) < cv2.contourArea(
        image_xy.astype(np.float32)
    )


def test_a_bad_quad_yields_no_mask_rather_than_a_guess():
    assert H.maze_polygon_px(np.full((4, 2), np.nan)) is None
    assert H.maze_polygon_px(np.zeros((3, 2))) is None


def blue_frame(centres, radius_px):
    frame = np.zeros((400, 640, 3), np.uint8)
    hsv = np.zeros((1, 1, 3), np.uint8)
    hsv[0, 0] = (88, 210, 180)                      # inside DEFAULT_HSV_LO/HI
    blue = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0].tolist()
    for centre in centres:
        cv2.circle(frame, (int(centre[0]), int(centre[1])), radius_px, blue, -1)
    return frame


def test_a_blue_marble_in_the_maze_is_found():
    moving_rc, _ = dot_quad()
    target = board_to_px(0.02, -0.03)
    frame = blue_frame([target], 6)
    found = H.detect_marble(frame, moving_rc)
    assert found is not None
    assert np.linalg.norm(found - target) < 3.0


def test_a_blue_dot_on_the_rim_is_not_reported():
    """The failure this exists to prevent: a marker standing in as the marble."""
    moving_rc, image_xy = dot_quad()
    frame = blue_frame(image_xy, 6)
    assert H.detect_marble(frame, moving_rc) is None


def test_the_marble_wins_over_dots_present_in_the_same_frame():
    moving_rc, image_xy = dot_quad()
    target = board_to_px(-0.04, 0.05)
    frame = blue_frame(list(image_xy) + [target], 6)
    found = H.detect_marble(frame, moving_rc)
    assert found is not None
    assert np.linalg.norm(found - target) < 3.0


def test_nothing_blue_means_nothing_reported():
    moving_rc, _ = dot_quad()
    assert H.detect_marble(np.zeros((400, 640, 3), np.uint8), moving_rc) is None


def test_a_speck_is_rejected_by_the_area_floor():
    moving_rc, _ = dot_quad()
    frame = blue_frame([board_to_px(0.0, 0.0)], 1)
    assert H.detect_marble(frame, moving_rc) is None
