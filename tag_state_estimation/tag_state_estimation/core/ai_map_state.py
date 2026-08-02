"""Geometry and filtering for the AI-only, map-native state estimator."""

import itertools
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
# CONFIRMED BY MEASUREMENT on this board. Not inherited: this is a custom board,
# not the CyberRunner one, so its plate dimensions had to be measured rather than
# assumed. Physically consistent too -- the maze is 259 x 229 mm, so the dots sit
# 5 mm outside its edge in x and 4 mm in y, i.e. on the rim where they are.
#
# DO NOT re-derive these from the DXF hole positions. That was attempted and failed
# three different ways: a joint fit returned 249.2 x 222.3 mm, which would put the
# dots INSIDE the maze on the marble's path; two tilt-corrected pixel ratios
# returned 275.8 and 285.3 mm. Those also disagree on ASPECT, which no
# camera-geometry effect can produce, so the fault is in the hole measurement --
# most likely its bounding box, which is outlier-sensitive and once picked up 22
# blobs for 21 holes. Measured values win; fits do not.
MOVING_MARKER_SPACING_X_M = 0.269
MOVING_MARKER_SPACING_Y_M = 0.237
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
        reacquire_after_sec=1.0,
        reacquire_radius_px=28.0,
        snap_after_frames=25,
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
        # Recovery from a MOVING-quad lock-up.
        #
        # ai_map_estimator_node feeds this guard's output back into the detector
        # (moving_tracker.corners = accepted), so once the guard starts holding, the
        # detector's local search window follows the STALE quad. If the plate has
        # since tilted, the real dots are outside that window, they are never found
        # again, and the guard holds forever: observed live as
        # moving_markers_timeout on 1097 of 1108 frames while the marble detector
        # was still reporting 0.995 confidence. Only restarting the node cleared it.
        #
        # So after reacquire_after_sec of continuous loss, snap back to the
        # calibrated anchor and re-enter acquisition, which relaxes the gates for
        # one frame. reacquire_radius_px is wider than acquire_radius_px because the
        # dots may have moved up to ~30 px within the tilt envelope while lost;
        # marker_quad_valid and the shape check still reject the fixed quad, whose
        # span differs from the moving one by more than 20%.
        self.reacquire_after_sec = max(0.0, float(reacquire_after_sec))
        self.reacquire_radius_px = max(0.0, float(reacquire_radius_px))
        # Per-corner rescue for the "one marker parked in the wrong place" failure.
        #
        # trusted requires |candidate - self.corners| <= max_step (~9.5 px at
        # 60 fps). If the held quad ever drifts from the true dots on ONE corner,
        # that corner can never be trusted again: the other three keep tracking, the
        # rejected one is dragged along by group_delta, and because all_observed is
        # never true the guard reports moving_markers_timeout indefinitely while the
        # bad corner sits visibly off its dot. Observed live as timeout on 1550 of
        # 1556 frames with all four dots detectable and the marble detector at
        # 0.995 confidence.
        #
        # So a corner that is FOUND but rejected for this many consecutive frames is
        # accepted. It is safe: the detector did find a real dot there, and
        # _shape_continuous plus marker_quad_valid still validate the whole quad.
        self.snap_after_frames = max(0, int(snap_after_frames))
        self.untrusted_frames = np.zeros(4, dtype=np.int64)
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
        if (
            self.reacquire_after_sec > 0.0
            and timestamp - self.loss_since > self.reacquire_after_sec
        ):
            # Break the stale-feedback loop -- see self.reacquire_after_sec.
            self.corners = self.anchor.copy()
            self.acquired = False
            self.loss_since = timestamp
            self.timestamp = None
            return self.corners.copy(), False, "moving_marker_reacquire"
        return self.corners.copy(), False, "moving_markers_timeout"

    def reseed(self, corners_rc):
        """Adopt an externally found quad and re-enter acquisition.

        Used with find_marker_quad_global to break out of a lock-up: the caller
        supplies a quad located by shape rather than by proximity, and the guard
        restarts tracking from it with the relaxed acquisition gates.
        """
        corners = np.asarray(corners_rc, dtype=np.float64)
        if corners.shape != (4, 2) or not np.all(np.isfinite(corners)):
            return False
        self.corners = corners.copy()
        self.acquired = False
        self.loss_since = None
        self.timestamp = None
        self.untrusted_frames[:] = 0
        return True

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
            # After a lock-up recovery the dots may be further out, so the moving
            # quad gets the wider reacquire radius once it has lost sync before.
            radius = self.acquire_radius_px
            if self.mode == "moving" and self.loss_since is not None:
                radius = max(radius, self.reacquire_radius_px)
            max_step = max(max_step, radius)
            residual_limit = max(residual_limit, radius)
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
            # Rescue corners stuck outside the budget -- see self.snap_after_frames.
            self.untrusted_frames[found & ~trusted] += 1
            self.untrusted_frames[trusted | ~found] = 0
            if self.snap_after_frames > 0:
                stale = found & (self.untrusted_frames >= self.snap_after_frames)
                if np.any(stale):
                    trusted = trusted | stale
                    self.untrusted_frames[stale] = 0
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


def find_marker_quad_global(
    frame,
    reference_rc,
    hsv_lo=(43, 125, 9),
    hsv_hi=(140, 255, 255),
    area_px2=(12.0, 400.0),
    aspect=(0.5, 2.0),
    search_radius_px=70.0,
    max_shape_error=0.20,
):
    """Find a marker quad anywhere near ``reference_rc`` by shape-matching.

    The per-frame tracker only searches a small window around its last accepted
    position, so a stale position sends it looking in the wrong place and it can
    never recover. This is the escape hatch: threshold the WHOLE frame for the
    marker colour, then pick the four blobs whose quadrilateral best matches the
    reference's shape signature. Identity comes from geometry, not from proximity
    to a position that may already be wrong.

    Measured on frames captured during a live lock-up: 28-43 blue blobs per frame
    (the pneumatic hose and background clutter also pass the colour test), yet the
    shape match landed within 6.5 px of the calibrated corners on 12 of 12 frames.

    Candidates are restricted to ``search_radius_px`` of a reference corner, which
    keeps the combinatorics small enough to run inline and stops a distant blue
    object from being considered at all. Returns (4,2) in [row, column] order to
    match the rest of this module, or None.
    """
    reference = np.asarray(reference_rc, dtype=np.float64)
    if reference.shape != (4, 2):
        return None
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.asarray(hsv_lo), np.asarray(hsv_hi))
    count, _, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)

    reference_xy = reference[:, ::-1]
    candidates = []
    for index in range(1, count):
        area = float(stats[index, cv2.CC_STAT_AREA])
        width = float(stats[index, cv2.CC_STAT_WIDTH])
        height = float(stats[index, cv2.CC_STAT_HEIGHT])
        if not area_px2[0] < area < area_px2[1]:
            continue
        if not aspect[0] < width / max(height, 1.0) < aspect[1]:
            continue
        point = np.asarray(centroids[index], dtype=np.float64)
        if np.min(np.linalg.norm(reference_xy - point, axis=1)) <= search_radius_px:
            candidates.append(point)
    if len(candidates) < 4:
        return None
    candidates = np.asarray(candidates)

    target = MarkerQuadGuard._shape_signature(reference)
    best_error, best_quad = max_shape_error, None
    for combo in itertools.combinations(range(len(candidates)), 4):
        quad = candidates[list(combo)]
        centre = quad.mean(axis=0)
        ordered = quad[np.argsort(-np.arctan2(quad[:, 1] - centre[1],
                                              quad[:, 0] - centre[0]))]
        for roll in range(4):
            trial = np.roll(ordered, roll, axis=0)[:, ::-1]  # -> (row, column)
            ratios = MarkerQuadGuard._shape_signature(trial) / np.maximum(target, 1e-9)
            error = float(np.abs(ratios - 1.0).max())
            if error < best_error:
                valid, _ = marker_quad_valid(trial, min_area_px2=5_000.0)
                if valid:
                    best_error, best_quad = error, trial
    return best_quad


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
        # max_speed_mps <= 0 (or non-finite) disables the gate.
        if self.max_speed_mps > 0.0 and np.isfinite(self.max_speed_mps):
            implied_speed = float(
                np.linalg.norm(measurement - self.position) / dt
            )
            if implied_speed > self.max_speed_mps:
                return np.full(2, np.nan), np.full(2, np.nan), "speed_gate"

        self.position = predicted + self.alpha * residual
        self.velocity = self.velocity + (self.beta / dt) * residual
        self.timestamp = timestamp
        return self.position.copy(), self.velocity.copy(), "valid"
