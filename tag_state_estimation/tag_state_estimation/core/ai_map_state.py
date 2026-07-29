"""Geometry and filtering for the AI-only, map-native state estimator."""

from dataclasses import dataclass

import cv2
import numpy as np

from .maze_layout import BOARD_HEIGHT_M, BOARD_WIDTH_M


# Fitted from 40 tilted views / 745 hole observations by
# tools/fit_marker_geometry.py, NOT the ETH original's 0.269 x 0.237. Those were
# inherited nominal values for different hardware; this board runs a custom maze.
# Profiling the dot-plane height h against the known DXF hole positions gives a
# clear minimum at h = 10 mm (median residual 2.13 mm at h=0, 1.49 mm at h=10,
# 1.92 mm at h=20), independently reproducing a 1 cm ruler measurement, and at
# that optimum the spacing is 249.2 x 222.3 mm. Median hole residual over the
# whole set improves 5.67 -> 1.49 mm versus the old constants.
MOVING_MARKER_SPACING_X_M = 0.2492
MOVING_MARKER_SPACING_Y_M = 0.2223
MARBLE_RADIUS_M = 0.006

MOVING_MARKERS_CENTERED_M = np.asarray(
    [
        [-MOVING_MARKER_SPACING_X_M / 2, -MOVING_MARKER_SPACING_Y_M / 2],
        [+MOVING_MARKER_SPACING_X_M / 2, -MOVING_MARKER_SPACING_Y_M / 2],
        [+MOVING_MARKER_SPACING_X_M / 2, +MOVING_MARKER_SPACING_Y_M / 2],
        [-MOVING_MARKER_SPACING_X_M / 2, +MOVING_MARKER_SPACING_Y_M / 2],
    ],
    dtype=np.float32,
)


@dataclass(frozen=True)
class MapMeasurement:
    valid: bool
    centered_xy: np.ndarray
    lower_left_xy: np.ndarray
    camera_height_m: float
    reason: str


class MarkerQuadGuard:
    """Reject marker substitutions and bridge short, local occlusions.

    Fixed markers may only move as a coherent camera-motion group. Moving
    markers may move independently by a small amount, but their quadrilateral
    must remain continuous and rigid from one frame to the next.
    """

    def __init__(
        self,
        initial_corners_rc,
        mode,
        occlusion_grace_sec=0.20,
        max_speed_px_s=None,
        jitter_px=2.0,
        max_group_residual_px=3.0,
        max_shape_change_fraction=0.08,
        smoothing=0.45,
        acquire_radius_px=14.0,
        anchor_radius_px=20.0,
    ):
        if mode not in ("fixed", "moving"):
            raise ValueError("mode must be 'fixed' or 'moving'")
        corners = np.asarray(initial_corners_rc, dtype=np.float64)
        if corners.shape != (4, 2) or not np.all(np.isfinite(corners)):
            raise ValueError("initial_corners_rc must be finite shape (4,2)")
        self.mode = mode
        self.corners = corners.copy()
        # Absolute anchor for the FIXED quad, and the whole reason it exists:
        # every other gate here is a per-FRAME limit. jitter_px +
        # max_speed_px_s * dt bounds how far a corner may move between frames
        # (~3.7 px at 60 fps), and found-but-untrusted corners are advanced by
        # group_delta every frame regardless. So displacement from the true dot
        # was never bounded at all -- a corner could walk arbitrarily far a few
        # px at a time, which is what happens when a hand or the blue hose
        # drifts through the search window for a while. Observed live: the fixed
        # quad wandered tens of px away over a long run while the same frames
        # replayed from the seeds tracked correctly.
        #
        # The outer frame is bolted down: measured over 119 frames spanning the
        # full tilt range, its dots move 0.9 px peak-to-peak. So a corner more
        # than anchor_radius_px from where it was calibrated is wrong by
        # definition, and clamping to the anchor is always the better guess.
        self.anchor = corners.copy()
        self.anchor_radius_px = max(0.0, float(anchor_radius_px))
        self.occlusion_grace_sec = max(0.0, float(occlusion_grace_sec))
        self.max_speed_px_s = float(
            max_speed_px_s
            if max_speed_px_s is not None
            else (100.0 if mode == "fixed" else 300.0)
        )
        self.jitter_px = max(0.0, float(jitter_px))
        self.max_group_residual_px = max(0.0, float(max_group_residual_px))
        self.max_shape_change_fraction = max(
            0.0, float(max_shape_change_fraction)
        )
        self.smoothing = float(np.clip(smoothing, 0.0, 1.0))
        # The initial corners come from hand clicks in select_markers, so they
        # are typically several pixels off the blue dots' actual centroids. The
        # per-frame motion gate (jitter + max_speed * dt, a few px at 60 fps) is
        # far tighter than that offset, so without a wider one-time acquisition
        # window the very first frame is rejected, the guard holds the clicked
        # seeds, and every later frame is rejected against those same seeds --
        # the tracker never latches onto the real dots and reselecting markers
        # cannot help. Acquire once with a window matching the detector's local
        # corner search (DEFAULT_SIZE_CROP_CORNERS / 2 ~ 11 px), then tighten.
        self.acquire_radius_px = max(0.0, float(acquire_radius_px))
        self.acquired = False
        self.timestamp = None
        self.loss_since = None

    @staticmethod
    def _shape_signature(corners_rc):
        xy = np.asarray(corners_rc, dtype=np.float64)[:, ::-1]
        edges = np.linalg.norm(np.roll(xy, -1, axis=0) - xy, axis=1)
        diagonals = np.asarray(
            [np.linalg.norm(xy[2] - xy[0]), np.linalg.norm(xy[3] - xy[1])]
        )
        area = abs(float(cv2.contourArea(xy.astype(np.float32))))
        return np.concatenate((edges, diagonals, [area]))

    def _shape_continuous(self, proposed, limit=None):
        valid, _ = marker_quad_valid(proposed, min_area_px2=5_000.0)
        if not valid:
            return False
        before = self._shape_signature(self.corners)
        after = self._shape_signature(proposed)
        ratios = after / np.maximum(before, 1.0e-6)
        if limit is None:
            limit = self.max_shape_change_fraction
        return bool(np.all((ratios >= 1.0 - limit) & (ratios <= 1.0 + limit)))

    def _held_result(self, timestamp, reason):
        if self.mode == "fixed":
            # The outer frame is physically fixed. Holding its last calibrated
            # position is safer than replacing it with an unrelated blue blob.
            return self.corners.copy(), True, reason
        if self.loss_since is None:
            self.loss_since = timestamp
        if timestamp - self.loss_since <= self.occlusion_grace_sec:
            return self.corners.copy(), True, "moving_marker_occlusion_grace"
        return self.corners.copy(), False, "moving_markers_timeout"

    def update(self, candidate_rc, found_mask, timestamp_sec):
        timestamp = float(timestamp_sec)
        candidate = np.asarray(candidate_rc, dtype=np.float64)
        found = np.asarray(found_mask, dtype=bool).reshape(-1)
        if candidate.shape != (4, 2) or found.shape != (4,):
            return self._held_result(timestamp, f"{self.mode}_markers_invalid")
        found &= np.all(np.isfinite(candidate), axis=1)

        dt = 1.0 / 60.0 if self.timestamp is None else timestamp - self.timestamp
        if not np.isfinite(dt) or dt <= 0.0 or dt > 0.5:
            dt = 1.0 / 60.0
        acquiring = not self.acquired
        max_step = self.jitter_px + self.max_speed_px_s * dt
        residual_limit = self.max_group_residual_px
        shape_limit = self.max_shape_change_fraction
        if acquiring:
            # Click errors are independent per corner, so they look like an
            # incoherent, shape-changing jump. Relax all three gates for the
            # single acquisition frame; marker_quad_valid still guards sanity.
            max_step = max(max_step, self.acquire_radius_px)
            residual_limit = max(residual_limit, self.acquire_radius_px)
            shape_limit = max(shape_limit, 0.25)
        deltas = candidate - self.corners

        if self.mode == "fixed":
            if np.count_nonzero(found) < 3:
                self.timestamp = timestamp
                return self._held_result(timestamp, "fixed_markers_held")
            group_delta = np.median(deltas[found], axis=0)
            residual = np.linalg.norm(deltas - group_delta, axis=1)
            trusted = (
                found
                & (residual <= residual_limit)
                & (np.linalg.norm(deltas, axis=1) <= max_step)
            )
            if np.count_nonzero(trusted) < 3:
                self.timestamp = timestamp
                return self._held_result(timestamp, "fixed_marker_jump_rejected")
            proposed = self.corners + group_delta
            proposed[trusted] = candidate[trusted]
            # Clamp to the calibrated anchor. Without this the per-frame gates
            # bound velocity but not displacement, so the quad can drift away a
            # few px at a time. See self.anchor.
            drift = np.linalg.norm(proposed - self.anchor, axis=1)
            runaway = drift > self.anchor_radius_px
            if np.any(runaway):
                proposed[runaway] = self.anchor[runaway]
                if np.count_nonzero(runaway) >= 3:
                    # The whole quad has walked off: re-seed rather than creep.
                    self.corners = self.anchor.copy()
                    self.timestamp = timestamp
                    return self._held_result(timestamp, "fixed_marker_drift_reset")
        else:
            trusted = found & (np.linalg.norm(deltas, axis=1) <= max_step)
            if np.count_nonzero(trusted) < 3:
                self.timestamp = timestamp
                return self._held_result(timestamp, "moving_marker_jump_rejected")
            group_delta = np.median(deltas[trusted], axis=0)
            proposed = self.corners + group_delta
            proposed[trusted] = candidate[trusted]

        if not self._shape_continuous(proposed, limit=shape_limit):
            self.timestamp = timestamp
            return self._held_result(timestamp, f"{self.mode}_marker_shape_rejected")

        # Snap straight onto the dots on the acquisition frame; smoothing a
        # multi-pixel seed correction would spread it over frames that the now
        # tight motion gate would reject.
        blend = 1.0 if acquiring else self.smoothing
        self.corners = (1.0 - blend) * self.corners + blend * proposed
        self.timestamp = timestamp
        self.acquired = True
        all_observed = bool(np.count_nonzero(trusted) == 4)
        if all_observed:
            self.loss_since = None
            return self.corners.copy(), True, "valid"
        if self.loss_since is None:
            self.loss_since = timestamp
        if self.mode == "fixed":
            return self.corners.copy(), True, "fixed_marker_group_recovered"
        if timestamp - self.loss_since <= self.occlusion_grace_sec:
            return self.corners.copy(), True, "moving_marker_occlusion_grace"
        return self.corners.copy(), False, "moving_markers_timeout"


def marker_quad_valid(corners_rc, min_area_px2=10_000.0):
    corners = np.asarray(corners_rc, dtype=np.float32)
    if corners.shape != (4, 2) or not np.all(np.isfinite(corners)):
        return False, "moving_markers_missing"
    xy = corners[:, ::-1]
    if not cv2.isContourConvex(xy.astype(np.int32)):
        return False, "moving_marker_order_invalid"
    area = abs(float(cv2.contourArea(xy)))
    if area < float(min_area_px2):
        return False, "moving_marker_quad_too_small"
    edges = np.linalg.norm(np.roll(xy, -1, axis=0) - xy, axis=1)
    if float(edges.min()) < 25.0 or float(edges.max() / edges.min()) > 3.0:
        return False, "moving_marker_geometry_invalid"
    return True, "ok"


def camera_center_in_board(T_camera_board):
    """Camera center in the moving-marker coordinate frame."""
    transform = np.asarray(T_camera_board, dtype=np.float64)
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    return -(rotation.T @ translation)


def map_ai_pixel(
    ball_xy_px,
    moving_corners_rc,
    T_camera_board=None,
    camera_height_fallback_m=0.20,
    marker_plane_height_m=0.0,
    marble_radius_m=MARBLE_RADIUS_M,
):
    """Map an AI pixel to board coordinates and compensate marble parallax.

    The homography produces the intersection with the moving-marker plane.
    The detected marble center is above that plane, so move the planar
    intersection toward the camera's projection by ``height / camera_height``.
    """
    valid, reason = marker_quad_valid(moving_corners_rc)
    missing = np.full(2, np.nan, dtype=np.float64)
    if not valid:
        return MapMeasurement(False, missing, missing, np.nan, reason)
    ball = np.asarray(ball_xy_px, dtype=np.float32)
    if ball.shape != (2,) or not np.all(np.isfinite(ball)):
        return MapMeasurement(False, missing, missing, np.nan, "ai_marble_missing")

    image_xy = np.asarray(moving_corners_rc, dtype=np.float32)[:, ::-1]
    try:
        image_to_board = cv2.getPerspectiveTransform(
            image_xy, MOVING_MARKERS_CENTERED_M
        )
        planar_xy = cv2.perspectiveTransform(
            ball.reshape(1, 1, 2), image_to_board
        )[0, 0].astype(np.float64)
    except (cv2.error, np.linalg.LinAlgError):
        return MapMeasurement(False, missing, missing, np.nan, "homography_failed")

    camera_height = float(camera_height_fallback_m)
    camera_xy = np.zeros(2, dtype=np.float64)
    if T_camera_board is not None:
        center = camera_center_in_board(T_camera_board)
        if np.all(np.isfinite(center)) and abs(float(center[2])) > 0.05:
            camera_height = abs(float(center[2]))
            camera_xy = center[:2]

    center_height = float(marble_radius_m) - float(marker_plane_height_m)
    if camera_height <= center_height + 0.01:
        return MapMeasurement(
            False, missing, missing, camera_height, "camera_height_invalid"
        )
    parallax_scale = 1.0 - center_height / camera_height
    centered = camera_xy + (planar_xy - camera_xy) * parallax_scale
    lower_left = centered + np.asarray(
        [BOARD_WIDTH_M / 2.0, BOARD_HEIGHT_M / 2.0], dtype=np.float64
    )
    return MapMeasurement(True, centered, lower_left, camera_height, "valid")


class AlphaBetaKinematics:
    """Timestamp-aware alpha-beta filter for position and velocity."""

    def __init__(self, alpha=0.65, beta=0.12, max_speed_mps=2.0):
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.max_speed_mps = float(max_speed_mps)
        self.position = None
        self.velocity = np.zeros(2, dtype=np.float64)
        self.timestamp = None

    def reset(self):
        self.position = None
        self.velocity[:] = 0.0
        self.timestamp = None

    def update(self, measurement_xy, timestamp_sec):
        measurement = np.asarray(measurement_xy, dtype=np.float64)
        timestamp = float(timestamp_sec)
        if measurement.shape != (2,) or not np.all(np.isfinite(measurement)):
            return np.full(2, np.nan), np.full(2, np.nan), "measurement_invalid"
        if self.position is None or self.timestamp is None:
            self.position = measurement.copy()
            self.velocity[:] = 0.0
            self.timestamp = timestamp
            return self.position.copy(), self.velocity.copy(), "initialized"

        dt = timestamp - self.timestamp
        if not np.isfinite(dt) or dt <= 1.0e-4 or dt > 0.5:
            self.position = measurement.copy()
            self.velocity[:] = 0.0
            self.timestamp = timestamp
            return self.position.copy(), self.velocity.copy(), "time_reset"

        predicted = self.position + self.velocity * dt
        residual = measurement - predicted
        implied_speed = float(np.linalg.norm(measurement - self.position) / dt)
        if implied_speed > self.max_speed_mps:
            return np.full(2, np.nan), np.full(2, np.nan), "speed_gate"

        self.position = predicted + self.alpha * residual
        self.velocity = self.velocity + (self.beta / dt) * residual
        self.timestamp = timestamp
        return self.position.copy(), self.velocity.copy(), "valid"
