import numpy as np

from tag_state_estimation.ai_marble_common import MarbleDetection
from tag_state_estimation.core.detection import Detector
from tag_state_estimation.core.hybrid_ball import HybridBallTracker
from tag_state_estimation.core.hole_mask import (
    HOLES_CENTERED_M,
    MARKERS_CENTERED_M,
    TimedHoleRejector,
    candidate_hole_index,
    project_holes_to_image,
)


def _seed_tracker(tracker, position):
    for _ in range(tracker.far_reacquire_confirm_frames):
        tracker.update(hsv_position=position, ai_position=position)


def test_fuses_agreeing_hsv_and_ai_positions():
    tracker = HybridBallTracker(
        ai_fusion_weight=0.5, far_reacquire_confirm_frames=1
    )
    tracker.update(
        hsv_position=np.array([100.0, 200.0]),
        ai_position=np.array([104.0, 202.0]),
    )
    result = tracker.update(
        hsv_position=np.array([100.0, 200.0]),
        ai_position=np.array([104.0, 202.0]),
    )
    assert result.source == "fused"
    assert np.allclose(result.measurement, [102.0, 201.0])
    assert abs(result.disagreement_px - np.sqrt(20.0)) < 1e-6


def test_ai_resumes_immediately_after_a_short_gap():
    tracker = HybridBallTracker(
        max_reacquire_jump_px=25.0, far_reacquire_confirm_frames=3
    )
    _seed_tracker(tracker, np.array([100.0, 200.0]))
    tracker.update()
    result = tracker.update(ai_position=np.array([108.0, 212.0]))
    assert result.source == "ai_resumed"
    assert np.allclose(result.measurement, [108.0, 212.0])


def test_reacquisition_is_confirmed_once_the_gap_outlasts_the_fast_path():
    tracker = HybridBallTracker(
        max_reacquire_jump_px=25.0,
        far_reacquire_confirm_frames=3,
        fast_reacquire_frames=1,
    )
    _seed_tracker(tracker, np.array([100.0, 200.0]))
    tracker.update()
    tracker.update()
    first = tracker.update(ai_position=np.array([108.0, 212.0]))
    second = tracker.update(ai_position=np.array([109.0, 212.0]))
    third = tracker.update(ai_position=np.array([108.0, 213.0]))
    assert first.source == "kalman_occlusion"
    assert second.source == "kalman_occlusion"
    assert third.source == "ai_reacquired_confirmed"


def test_ai_only_startup_requires_consistent_confirmation():
    tracker = HybridBallTracker(far_reacquire_confirm_frames=3)
    first = tracker.update(ai_position=np.array([100.0, 100.0]))
    second = tracker.update(ai_position=np.array([101.0, 100.0]))
    third = tracker.update(ai_position=np.array([100.0, 101.0]))
    assert first.source == "lost"
    assert second.source == "lost"
    assert third.source == "ai_reacquired_confirmed"


def test_missing_measurement_requests_kalman_then_becomes_lost():
    tracker = HybridBallTracker(occlusion_grace_frames=2)
    _seed_tracker(tracker, np.array([100.0, 200.0]))
    first = tracker.update()
    second = tracker.update()
    third = tracker.update()
    assert first.source == "kalman_occlusion"
    assert second.source == "kalman_occlusion"
    assert third.source == "lost"
    assert np.all(np.isnan(first.measurement))


def test_far_ai_reacquisition_requires_consistent_confirmation():
    tracker = HybridBallTracker(
        max_reacquire_jump_px=10.0,
        agreement_radius_px=5.0,
        far_reacquire_confirm_frames=3,
    )
    _seed_tracker(tracker, np.array([20.0, 20.0]))
    first = tracker.update(ai_position=np.array([100.0, 100.0]))
    second = tracker.update(ai_position=np.array([101.0, 99.0]))
    third = tracker.update(ai_position=np.array([100.0, 101.0]))
    assert first.source == "kalman_occlusion"
    assert second.source == "kalman_occlusion"
    assert third.source == "ai_reacquired_confirmed"
    assert np.linalg.norm(third.measurement - [100.25, 100.25]) < 1.0


def test_disagreement_prefers_candidate_closer_to_previous_position():
    tracker = HybridBallTracker(
        agreement_radius_px=5.0, max_reacquire_jump_px=30.0
    )
    _seed_tracker(tracker, np.array([100.0, 100.0]))
    result = tracker.update(
        hsv_position=np.array([140.0, 140.0]),
        ai_position=np.array([105.0, 104.0]),
    )
    assert result.source == "ai_disagreement"
    assert np.allclose(result.measurement, [105.0, 104.0])


def test_hsv_only_candidate_is_never_accepted_or_used_to_reset_loss():
    tracker = HybridBallTracker(occlusion_grace_frames=2)
    startup = tracker.update(hsv_position=np.array([100.0, 200.0]))
    assert startup.source == "lost"
    assert np.all(np.isnan(startup.measurement))

    _seed_tracker(tracker, np.array([100.0, 200.0]))
    first = tracker.update(hsv_position=np.array([101.0, 201.0]))
    second = tracker.update(hsv_position=np.array([101.0, 201.0]))
    third = tracker.update(hsv_position=np.array([101.0, 201.0]))
    assert first.source == "kalman_occlusion"
    assert second.source == "kalman_occlusion"
    assert third.source == "lost"


def test_nearby_ai_only_candidate_remains_accepted():
    tracker = HybridBallTracker(
        max_reacquire_jump_px=25.0, far_reacquire_confirm_frames=1
    )
    tracker.update(
        hsv_position=np.array([100.0, 200.0]),
        ai_position=np.array([100.0, 200.0]),
    )
    result = tracker.update(ai_position=np.array([105.0, 207.0]))
    assert result.source == "ai_reacquired"
    assert np.allclose(result.measurement, [105.0, 207.0])


def test_moving_marble_is_reacquired_rather_than_outrunning_confirmation():
    """A marble both detectors can see must not stay lost because it is moving.

    Ten pixels per frame is 0.42 m/s on this board -- ordinary play, not a
    corner case. The confirmation test used to compare each candidate against a
    running average of the previous ones, which a marble at that speed outran,
    so the counter reset every second frame and never confirmed.
    """
    tracker = HybridBallTracker(
        agreement_radius_px=12.0,
        max_reacquire_jump_px=25.0,
        far_reacquire_confirm_frames=3,
        fast_reacquire_frames=0,
        trust_hsv_alone=True,
    )
    start = np.array([300.0, 180.0])
    _seed_tracker(tracker, start)
    tracker.update()

    for step in range(1, 5):
        moving = start + np.array([10.0 * step, 0.0])
        result = tracker.update(hsv_position=moving, ai_position=moving)
        if np.all(np.isfinite(result.measurement)):
            assert result.source == "fused_reacquired_confirmed"
            # Accepted where the marble is now, not averaged over the window.
            assert np.allclose(result.measurement, moving)
            break
    else:
        raise AssertionError("moving marble was never reacquired")


def test_colour_carries_the_frame_when_the_ai_jumps_onto_a_hole():
    """The measured failure at the bottom-right hole.

    On roughly 3% of the frames it fires, the learned detector reports a black
    hole as the marble, at around 0.87 confidence. One live case put it 131 mm
    from where colour had the marble. Deferring to the AI there loses the frame
    twice: its position fails the jump test so nothing is published, and the
    good colour candidate goes with it.
    """
    tracker = HybridBallTracker(
        agreement_radius_px=12.0, max_reacquire_jump_px=48.0,
        trust_hsv_alone=True,
    )
    start = np.array([200.0, 200.0])
    _seed_tracker(tracker, start)
    on_track = start + np.array([6.0, 0.0])
    a_hole = start + np.array([135.0, 90.0])

    result = tracker.update(hsv_position=on_track, ai_position=a_hole)
    assert result.source == "hsv_disagreement"
    assert np.allclose(result.measurement, on_track)


def test_a_nearby_ai_candidate_still_outranks_colour_on_disagreement():
    """Colour only takes over when the AI has left the track, not before."""
    tracker = HybridBallTracker(
        agreement_radius_px=5.0, max_reacquire_jump_px=48.0,
        trust_hsv_alone=True,
    )
    start = np.array([200.0, 200.0])
    _seed_tracker(tracker, start)
    result = tracker.update(
        hsv_position=start + np.array([2.0, 0.0]),
        ai_position=start + np.array([20.0, 0.0]),
    )
    assert result.source == "ai_disagreement"
    assert np.allclose(result.measurement, start + np.array([20.0, 0.0]))


def test_one_detector_carries_a_marble_up_to_the_full_speed_budget():
    """HSV alone must keep a fast marble, not just a slow one.

    max_reacquire_jump_px is now the only speed bound in the estimator, sized
    from 2 m/s at 1.03 px/mm and ~45 fps. A marble inside that budget has to be
    accepted on every frame; at 25 px it was capped at 1.05 m/s and anything
    quicker was dropped frame after frame, because _confirm_reacquisition tests
    against the same bound and so could never recover it either.
    """
    tracker = HybridBallTracker(max_reacquire_jump_px=48.0, trust_hsv_alone=True)
    start = np.array([200.0, 200.0])
    _seed_tracker(tracker, start)
    for step in range(1, 7):
        moving = start + np.array([40.0 * step, 0.0])
        result = tracker.update(hsv_position=moving)
        assert result.source == "hsv_only", f"lost the marble on step {step}"
        assert np.allclose(result.measurement, moving)


def test_a_candidate_beyond_the_speed_budget_is_still_refused():
    tracker = HybridBallTracker(max_reacquire_jump_px=48.0, trust_hsv_alone=True)
    _seed_tracker(tracker, np.array([200.0, 200.0]))
    result = tracker.update(hsv_position=np.array([400.0, 200.0]))
    assert result.source == "kalman_occlusion"
    assert np.all(np.isnan(result.measurement))


def test_fused_position_after_loss_requires_confirmation():
    tracker = HybridBallTracker(
        occlusion_grace_frames=10,
        far_reacquire_confirm_frames=3,
        fast_reacquire_frames=0,
    )
    tracker.update(
        hsv_position=np.array([100.0, 200.0]),
        ai_position=np.array([100.0, 200.0]),
    )
    tracker.update(
        hsv_position=np.array([100.0, 200.0]),
        ai_position=np.array([100.0, 200.0]),
    )
    tracker.update(
        hsv_position=np.array([100.0, 200.0]),
        ai_position=np.array([100.0, 200.0]),
    )
    tracker.update()
    first = tracker.update(
        hsv_position=np.array([103.0, 202.0]),
        ai_position=np.array([103.0, 202.0]),
    )
    second = tracker.update(
        hsv_position=np.array([104.0, 202.0]),
        ai_position=np.array([104.0, 202.0]),
    )
    third = tracker.update(
        hsv_position=np.array([103.0, 203.0]),
        ai_position=np.array([103.0, 203.0]),
    )
    assert first.source == "kalman_occlusion"
    assert second.source == "kalman_occlusion"
    assert third.source == "fused_reacquired_confirmed"


def test_hsv_predictive_crop_resets_while_kalman_grace_continues():
    class MissingAiDetector:
        @staticmethod
        def detect(_frame):
            return MarbleDetection(False, x_px=0.0, y_px=0.0, confidence=0.1)

    detector = Detector(np.zeros((4, 2)), ai_mode="off")
    detector.ai_mode = "hybrid"
    detector.ai_detector = MissingAiDetector()
    detector.hybrid_tracker.last_position = np.array([100.0, 200.0])
    detector.ball_pos = np.array([100.0, 200.0])
    detector.is_ball_found = True
    missing = np.array([np.nan, np.nan])
    frame = np.zeros((400, 640, 3), dtype=np.uint8)

    for _ in range(Detector.HSV_CROP_RESET_MISSING_FRAMES - 1):
        detector._select_hybrid_ball(frame, missing)
        assert detector.is_ball_found

    detector._select_hybrid_ball(frame, missing)
    assert detector.last_ball_source == "kalman_occlusion"
    assert detector.ball_pos is None
    assert not detector.is_ball_found


def test_detector_converts_ai_xy_to_internal_row_column():
    class FakeAiDetector:
        @staticmethod
        def detect(_frame):
            return MarbleDetection(True, x_px=220.0, y_px=110.0, confidence=0.99)

    detector = Detector(
        np.zeros((4, 2)),
        ai_mode="off",
        ai_far_reacquire_confirm_frames=1,
        ai_hole_rejection_delay_sec=0.0,
    )
    detector.ai_mode = "hybrid"
    detector.ai_detector = FakeAiDetector()
    result = detector._select_hybrid_ball(
        np.zeros((400, 640, 3), dtype=np.uint8),
        np.array([np.nan, np.nan]),
    )
    assert np.allclose(result, [110.0, 220.0])


def _board_to_test_pixel(board_xy):
    # Synthetic view: 1 metre maps to 1000 pixels, with an image offset.
    return np.asarray(
        [200.0 - board_xy[1] * 1000.0, 300.0 + board_xy[0] * 1000.0],
        dtype=np.float32,
    )


def test_hole_mask_follows_projected_board_coordinates():
    corners_rc = np.asarray(
        [_board_to_test_pixel(point) for point in MARKERS_CENTERED_M],
        dtype=np.float32,
    )
    candidate_rc = _board_to_test_pixel(HOLES_CENTERED_M[5])
    assert candidate_hole_index(candidate_rc, corners_rc, margin_m=0.0) == 5

    safe_rc = _board_to_test_pixel(np.array([0.0, 0.0], dtype=np.float32))
    assert candidate_hole_index(safe_rc, corners_rc, margin_m=0.0) is None

    centers_xy, radii_px = project_holes_to_image(corners_rc, margin_m=0.0)
    expected_xy = candidate_rc[::-1]
    assert np.linalg.norm(centers_xy[5] - expected_xy) < 1.0e-3
    assert abs(float(radii_px[5]) - 7.5) < 1.0e-3


def test_hole_rejection_requires_two_continuous_seconds():
    rejector = TimedHoleRejector(delay_sec=2.0)
    assert rejector.update(5, now=10.0) == (False, 0.0)
    rejected, elapsed = rejector.update(5, now=11.99)
    assert not rejected
    assert abs(elapsed - 1.99) < 1.0e-9
    assert rejector.update(5, now=12.0) == (True, 2.0)

    # Leaving the hole, or entering a different one, restarts the timer.
    assert rejector.update(None, now=12.1) == (False, 0.0)
    assert rejector.update(6, now=20.0) == (False, 0.0)


def test_zero_delay_rejects_hole_on_first_frame():
    rejector = TimedHoleRejector(delay_sec=0.0)
    assert rejector.update(5, now=10.0) == (True, 0.0)


def test_hybrid_rejects_visible_ai_candidate_on_a_hole():
    corners_rc = np.asarray(
        [_board_to_test_pixel(point) for point in MARKERS_CENTERED_M],
        dtype=np.float32,
    )
    hole_rc = _board_to_test_pixel(HOLES_CENTERED_M[5])

    class HoleAiDetector:
        @staticmethod
        def detect(_frame):
            return MarbleDetection(
                True,
                x_px=float(hole_rc[1]),
                y_px=float(hole_rc[0]),
                confidence=0.999,
            )

    detector = Detector(
        np.zeros((4, 2)),
        ai_mode="off",
        ai_far_reacquire_confirm_frames=1,
        ai_hole_rejection_delay_sec=0.0,
    )
    detector.ai_mode = "hybrid"
    detector.ai_detector = HoleAiDetector()
    detector.corners = corners_rc
    detector.corners_missing = False
    result = detector._select_hybrid_ball(
        np.zeros((400, 640, 3), dtype=np.uint8),
        np.array([np.nan, np.nan]),
    )
    assert np.all(np.isnan(result))
    assert detector.last_ball_source == "ai_hole_rejected_6"
