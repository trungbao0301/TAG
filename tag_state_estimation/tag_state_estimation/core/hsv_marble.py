"""Find the marble by colour, but only inside the maze.

The marble is blue and so are the eight reference dots, which is why an HSV
marble candidate was never allowed to stand on its own: it locks onto a dot and
reports the marble parked on the rim. Restricting the search geometrically
removes the ambiguity instead of trying to tell two blue blobs apart by size.

The dots sit on the moving rim, outside the play surface. Measured on this board
the dot ring is 269.0 x 237.0 mm against a 259.0 x 229.0 mm maze, so every dot is
5.0 mm (x) or 4.0 mm (y) beyond the maze edge, while the marble's 6.0 mm radius
keeps its centre at least 6.0 mm inside that same edge. So a mask at the maze
rectangle already separates them, and cannot clip the marble.

The mask is built from the four moving dots each frame, so it tilts with the
plate rather than being fixed in image space.
"""
import cv2
import numpy as np

from .ai_map_state import MOVING_MARKERS_CENTERED_M
from .maze_layout import BOARD_HEIGHT_M, BOARD_WIDTH_M

# Blue, matching detection.DEFAULT_HSV_BALL, which was sampled on this camera
# with white balance and exposure locked.
DEFAULT_HSV_LO = (60, 162, 50)
DEFAULT_HSV_HI = (116, 255, 243)

# Pulled in from the maze edge before searching. The dots are already 4-5 mm
# outside that edge, so this only adds margin against homography error and the
# dots' own radius; 2 mm keeps 6-7 mm of clearance while still admitting a marble
# centre that can legitimately sit 6 mm from the edge.
DEFAULT_INSET_M = 0.002

# A marble is far larger than a dot in the image. Kept as a sanity bound rather
# than the primary guard -- the mask is the guard -- so it only has to reject
# specular glints and sensor noise.
DEFAULT_AREA_PX2 = (25.0, 2000.0)


def maze_polygon_px(moving_rc, inset_m=DEFAULT_INSET_M):
    """The maze rectangle in image pixels, from the four moving dots.

    moving_rc is [row, col] as the trackers carry it. Returns float32 [x, y]
    corners, or None if the quad is unusable.
    """
    corners = np.asarray(moving_rc, dtype=np.float64)
    if corners.shape != (4, 2) or not np.all(np.isfinite(corners)):
        return None
    image_xy = corners[:, ::-1].astype(np.float32)
    try:
        board_to_image = cv2.getPerspectiveTransform(
            MOVING_MARKERS_CENTERED_M.astype(np.float32), image_xy
        )
    except cv2.error:
        return None

    half_x = BOARD_WIDTH_M / 2.0 - inset_m
    half_y = BOARD_HEIGHT_M / 2.0 - inset_m
    if half_x <= 0.0 or half_y <= 0.0:
        return None
    # Same corner order as MOVING_MARKERS_CENTERED_M, so the mapping is direct.
    maze = np.asarray(
        [
            [-half_x, -half_y],
            [+half_x, -half_y],
            [+half_x, +half_y],
            [-half_x, +half_y],
        ],
        dtype=np.float32,
    ).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(maze, board_to_image).reshape(-1, 2)


def detect_marble(
    frame,
    moving_rc,
    hsv_lo=DEFAULT_HSV_LO,
    hsv_hi=DEFAULT_HSV_HI,
    inset_m=DEFAULT_INSET_M,
    area_px2=DEFAULT_AREA_PX2,
):
    """Marble centre as [x, y] pixels from colour alone, or None.

    Returns None rather than a guess whenever the quad is unusable or nothing
    inside the maze matches, so the caller can treat it as "HSV had nothing" and
    fall back on the learned detector.
    """
    if frame is None or frame.ndim != 3:
        return None
    polygon = maze_polygon_px(moving_rc, inset_m=inset_m)
    if polygon is None:
        return None

    height, width = frame.shape[:2]
    mask_area = np.zeros((height, width), np.uint8)
    cv2.fillConvexPoly(mask_area, polygon.astype(np.int32), 255)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    colour = cv2.inRange(hsv, np.asarray(hsv_lo, np.uint8), np.asarray(hsv_hi, np.uint8))
    colour = cv2.bitwise_and(colour, mask_area)
    # One open-close pass: drop single-pixel noise, close the specular hole a
    # round marble usually has in the middle.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    colour = cv2.morphologyEx(colour, cv2.MORPH_OPEN, kernel)
    colour = cv2.morphologyEx(colour, cv2.MORPH_CLOSE, kernel)

    count, _, stats, centroids = cv2.connectedComponentsWithStats(colour, 8)
    lo, hi = area_px2
    best, best_area = None, 0.0
    for index in range(1, count):
        area = float(stats[index, cv2.CC_STAT_AREA])
        if area < lo or area > hi or area <= best_area:
            continue
        best_area, best = area, centroids[index]
    if best is None:
        return None
    return np.asarray([best[0], best[1]], dtype=np.float64)
