import numpy as np

import cv2
from tag_state_estimation.core.ai_map_state import (
    find_marker_quad_global,
    AlphaBetaKinematics,
    MarkerQuadGuard,
    MOVING_MARKERS_CENTERED_M,
    camera_center_in_board,
    map_ai_pixel,
    marker_quad_valid,
)


def _board_to_pixel(board_xy):
    # x_px = 320 + 1000*x_m, y_px = 200 - 1000*y_m
    return np.asarray(
        [200.0 - board_xy[1] * 1000.0, 320.0 + board_xy[0] * 1000.0],
        dtype=np.float32,
    )


def _synthetic_corners_rc():
    return np.asarray(
        [_board_to_pixel(point) for point in MOVING_MARKERS_CENTERED_M],
        dtype=np.float32,
    )


def test_marker_quad_validation_accepts_ordered_board_corners():
    valid, reason = marker_quad_valid(_synthetic_corners_rc())
    assert valid
    assert reason == "ok"


def test_map_center_pixel_maps_to_board_center_and_lower_left_center():
    result = map_ai_pixel(
        [320.0, 200.0],
        _synthetic_corners_rc(),
        camera_height_fallback_m=0.20,
        marble_radius_m=0.0,
    )
    assert result.valid
    assert np.allclose(result.centered_xy, [0.0, 0.0], atol=1.0e-6)
    assert np.allclose(result.lower_left_xy, [0.1295, 0.1145], atol=1.0e-6)


def test_parallax_correction_uses_12mm_marble_and_20cm_camera_height():
    transform = np.eye(4)
    transform[2, 3] = 0.20
    assert np.allclose(camera_center_in_board(transform), [0.0, 0.0, -0.20])
    # Pixel corresponds to the marker-plane point x=0.100 m.
    result = map_ai_pixel(
        [420.0, 200.0],
        _synthetic_corners_rc(),
        T_camera_board=transform,
        marble_radius_m=0.006,
    )
    assert result.valid
    assert abs(result.centered_xy[0] - 0.097) < 1.0e-6
    assert abs(result.camera_height_m - 0.20) < 1.0e-9


def test_kinematics_reports_velocity_and_rejects_impossible_jump():
    tracker = AlphaBetaKinematics(alpha=1.0, beta=1.0, max_speed_mps=2.0)
    position, velocity, _ = tracker.update([0.0, 0.0], 1.0)
    assert np.allclose(position, [0.0, 0.0])
    assert np.allclose(velocity, [0.0, 0.0])
    position, velocity, status = tracker.update([0.01, 0.0], 1.1)
    assert status == "valid"
    assert np.allclose(position, [0.01, 0.0])
    assert np.allclose(velocity, [0.1, 0.0])
    position, velocity, status = tracker.update([1.0, 0.0], 1.2)
    assert status == "speed_gate"
    assert np.all(np.isnan(position))
    assert np.all(np.isnan(velocity))


def test_a_zero_speed_limit_turns_the_gate_off():
    """The rig runs with the gate off; HybridBallTracker bounds speed instead.

    It fired on 18 of 6000 live frames, every one of them a marble both
    detectors had located, rejected only for running faster than 2 m/s.
    """
    tracker = AlphaBetaKinematics(alpha=1.0, beta=1.0, max_speed_mps=0.0)
    tracker.update([0.0, 0.0], 1.0)
    position, velocity, status = tracker.update([1.0, 0.0], 1.1)
    assert status == "valid"
    assert np.all(np.isfinite(position))
    assert np.all(np.isfinite(velocity))


def test_fixed_guard_holds_position_when_one_marker_is_covered():
    corners = _synthetic_corners_rc()
    guard = MarkerQuadGuard(corners, mode="fixed", smoothing=1.0)
    candidate = corners.copy()
    candidate[2] += [80.0, -100.0]
    accepted, valid, status = guard.update(
        candidate, [True, True, False, True], 1.0
    )
    assert valid
    assert status == "fixed_marker_group_recovered"
    assert np.allclose(accepted, corners)


def test_fixed_guard_accepts_small_consistent_camera_motion():
    corners = _synthetic_corners_rc()
    guard = MarkerQuadGuard(corners, mode="fixed", smoothing=1.0)
    shifted = corners + [1.0, 2.0]
    accepted, valid, status = guard.update(shifted, [True] * 4, 1.0)
    assert valid
    assert status == "valid"
    assert np.allclose(accepted, shifted)


def test_fixed_guard_acquires_dots_a_few_pixels_off_the_hand_clicked_seeds():
    # select_markers clicks land several px from the dots' centroids, which is
    # more than jitter + max_speed * dt allows. Without a one-time acquisition
    # window the guard held the clicks forever and never tracked the dots.
    clicked = _synthetic_corners_rc()
    dots = clicked + [[6.0, -7.0], [-8.0, 5.0], [4.0, 9.0], [-5.0, -6.0]]
    guard = MarkerQuadGuard(clicked, mode="fixed")

    accepted, valid, status = guard.update(dots, [True] * 4, 1.0)
    assert valid
    assert status == "valid"
    assert np.allclose(accepted, dots)

    # Acquisition is one-shot: the tight gate is back on the next frame.
    jumped = dots + [[0.0, 0.0], [0.0, 0.0], [30.0, 30.0], [30.0, 30.0]]
    accepted, valid, status = guard.update(jumped, [True] * 4, 1.0 + 1.0 / 60.0)
    assert valid
    assert status == "fixed_marker_jump_rejected"
    assert np.allclose(accepted, dots)


def test_fixed_guard_cannot_creep_away_from_its_calibrated_anchor():
    # Every other gate is a per-FRAME limit, so a corner nudged 2 px per frame
    # passes all of them and the quad walks arbitrarily far. Observed live: the
    # fixed markers ended up tens of px off the real dots after a long run.
    corners = _synthetic_corners_rc()
    guard = MarkerQuadGuard(corners, mode="fixed", smoothing=1.0)
    creeping = corners.copy()
    seen = []
    for step in range(200):
        creeping = creeping + [0.0, 2.0]  # 2 px/frame, inside every gate
        accepted, valid, status = guard.update(
            creeping, [True] * 4, 1.0 + step / 60.0
        )
        seen.append(status)
        assert valid  # fixed mode stays usable by holding position
        drift = np.linalg.norm(accepted - corners, axis=1).max()
        assert drift <= guard.anchor_radius_px + 1.0, (
            f"drifted {drift:.1f} px by step {step}"
        )
    # The anchor must have fired; afterwards the runaway candidate is genuinely
    # far from the held quad, so plain jump rejection is the correct report.
    assert "fixed_marker_drift_reset" in seen
    assert seen[-1] in ("fixed_marker_drift_reset", "fixed_marker_jump_rejected")


def test_fixed_guard_still_follows_genuine_small_camera_motion():
    # The anchor must not block real motion inside its radius.
    corners = _synthetic_corners_rc()
    guard = MarkerQuadGuard(corners, mode="fixed", smoothing=1.0)
    shifted = corners + [1.0, 2.0]
    accepted, valid, status = guard.update(shifted, [True] * 4, 1.0)
    assert valid and status == "valid"
    assert np.allclose(accepted, shifted)


def test_moving_guard_recovers_from_a_lock_up_instead_of_holding_forever():
    # Live failure: the estimator feeds this guard's output back into the detector,
    # so once it holds, the detector's search window follows the stale quad and can
    # never re-find dots that have since moved. Observed as moving_markers_timeout
    # on 1097/1108 frames while the marble detector reported 0.995 confidence.
    corners = _synthetic_corners_rc()
    guard = MarkerQuadGuard(
        corners, mode="moving", occlusion_grace_sec=0.20,
        reacquire_after_sec=0.5, smoothing=1.0,
    )
    guard.update(corners, [True] * 4, 1.0)          # acquire cleanly

    # Detector finds nothing at all (the stale-window situation).
    t, statuses = 1.0, []
    for _ in range(80):
        t += 1.0 / 60.0
        statuses.append(guard.update(corners, [False] * 4, t)[2])
    assert "moving_marker_reacquire" in statuses, statuses[-5:]
    assert not guard.acquired, "must re-enter acquisition so the gates relax"
    assert np.allclose(guard.corners, guard.anchor), "must snap back to the anchor"

    # With the wider re-acquire window it now latches onto dots well beyond the
    # normal per-frame budget.
    moved = np.asarray(guard.anchor) + [9.0, -11.0]
    accepted, valid, status = guard.update(moved, [True] * 4, t + 1.0 / 60.0)
    assert valid and status == "valid", status
    assert np.allclose(accepted, moved)


def test_moving_guard_rejects_outside_blob_then_times_out():
    corners = _synthetic_corners_rc()
    guard = MarkerQuadGuard(
        corners, mode="moving", occlusion_grace_sec=0.20, smoothing=1.0
    )
    candidate = corners.copy()
    candidate[1] += [100.0, 100.0]
    accepted, valid, status = guard.update(candidate, [True] * 4, 1.0)
    assert valid
    assert status == "moving_marker_occlusion_grace"
    assert np.allclose(accepted, corners)
    accepted, valid, status = guard.update(candidate, [True] * 4, 1.25)
    assert not valid
    assert status == "moving_markers_timeout"
    assert np.allclose(accepted, corners)


def test_moving_guard_accepts_small_smooth_tilt_motion():
    corners = _synthetic_corners_rc()
    guard = MarkerQuadGuard(corners, mode="moving", smoothing=1.0)
    candidate = corners.copy()
    candidate[:, 0] += [-1.0, 1.0, 1.5, -1.5]
    accepted, valid, status = guard.update(candidate, [True] * 4, 1.0)
    assert valid
    assert status == "valid"
    assert np.allclose(accepted, candidate)


def test_global_quad_finder_ignores_decoy_blobs_and_matches_on_shape():
    # The real frame has 28-43 blue blobs (hose, background), so identity must come
    # from the quad's shape, not from proximity to a possibly-stale position.
    truth = np.float32([[300, 150], [300, 420], [90, 420], [90, 150]])  # (row, col)
    frame = np.zeros((480, 640, 3), np.uint8)
    blue = (200, 110, 40)  # BGR, inside DEFAULT_HSV_CORNERS
    for r, c in truth:
        cv2.circle(frame, (int(c), int(r)), 4, blue, -1)
    # decoys: same colour and size, wrong geometry, inside the search radius
    for r, c in [(300, 190), (95, 380), (250, 150), (140, 430), (300, 380)]:
        cv2.circle(frame, (int(c), int(r)), 4, blue, -1)

    found = find_marker_quad_global(frame, truth)
    assert found is not None, "should locate the quad among the decoys"
    assert np.linalg.norm(found - truth, axis=1).max() < 3.0, found


def test_global_quad_finder_returns_none_when_there_is_no_quad():
    frame = np.zeros((480, 640, 3), np.uint8)
    cv2.circle(frame, (100, 100), 4, (200, 110, 40), -1)
    assert find_marker_quad_global(frame, _synthetic_corners_rc()) is None


def test_reseed_restarts_acquisition_from_the_supplied_quad():
    corners = _synthetic_corners_rc()
    guard = MarkerQuadGuard(corners, mode="moving", smoothing=1.0)
    guard.update(corners, [True] * 4, 1.0)
    elsewhere = np.asarray(corners, dtype=np.float64) + [18.0, -14.0]
    assert guard.reseed(elsewhere)
    assert not guard.acquired and np.allclose(guard.corners, elsewhere)
    assert not guard.reseed(np.zeros((3, 2)))          # rejects bad shape
