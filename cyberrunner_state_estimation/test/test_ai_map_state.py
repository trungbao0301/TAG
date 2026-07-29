import numpy as np

from cyberrunner_state_estimation.core.ai_map_state import (
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
