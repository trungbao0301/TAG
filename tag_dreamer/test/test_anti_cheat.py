"""Anti-cheat decision tests.

The two rejection rules are what stand between the reward and a marble that
hops 25 mm into a corridor 811 mm further along the path, so they are worth
pinning down. TcpEnv itself needs ROS and a camera bridge to construct, so
these call the two methods against a stub holding only the attributes they
touch -- which is also a useful check that they touch nothing else.
"""
import os
import sys
import types

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

gym = pytest.importorskip("gym", reason="env_tcp imports gym")
pytest.importorskip(
    "ament_index_python", reason="env_tcp imports ament_index_python"
)

from tag_dreamer.env_tcp import TagGym  # noqa: E402

PATH_STEP_M = 0.0002


def make_env(**over):
    """A stub carrying only what the two methods under test read or write."""
    env = types.SimpleNamespace(
        anti_cheat_max_speed_mps=1.0,
        anti_cheat_min_step_m=0.010,
        anti_cheat_travel_ratio=3.0,
        travel_since_credit=0.0,
        last_step_pos=None,
        last_step_implausible=False,
        last_time=0.0,
    )
    for key, value in over.items():
        setattr(env, key, value)
    return env


def roll(env, *positions, step_dt=0.040):
    """Feed a sequence of ball positions, one per step, spaced step_dt apart."""
    for position in positions:
        # _accumulate_travel measures dt against wall clock, so anchor last_time
        # step_dt in the past for each call.
        env.last_time = __import__("time").time() - step_dt
        TagGym._accumulate_travel(env, np.asarray(position, dtype=np.float32))


def allowed_m(env):
    return max(
        env.anti_cheat_min_step_m,
        env.anti_cheat_travel_ratio * env.travel_since_credit,
    )


def test_first_step_banks_nothing():
    env = make_env()
    roll(env, (0.0, 0.0))
    assert env.travel_since_credit == 0.0
    assert env.last_step_implausible is False


def test_ordinary_roll_is_banked_and_plausible():
    env = make_env()
    # 5 mm per 40 ms step is 0.125 m/s, well inside the 1.0 m/s cap.
    roll(env, (0.0, 0.0), (0.005, 0.0), (0.010, 0.0))
    assert env.last_step_implausible is False
    assert env.travel_since_credit == pytest.approx(0.010, abs=1e-4)


def test_travel_accumulates_across_unscored_steps():
    """The freeze fix: distance rolled while nothing scored still counts."""
    env = make_env()
    roll(env, (0.0, 0.0), *[(0.005 * k, 0.0) for k in range(1, 16)])
    assert env.travel_since_credit == pytest.approx(0.075, abs=1e-3)
    # 75 mm rolled buys 225 mm of path at ratio 3, so rejoining the path 75 mm
    # further along is accepted rather than denied forever.
    claimed_points = int(0.075 / PATH_STEP_M)
    assert claimed_points * PATH_STEP_M <= allowed_m(env)


def test_sideways_hop_is_rejected():
    """The shortcut: 25 mm rolled cannot claim 811 mm of path."""
    env = make_env()
    roll(env, (0.0, 0.0), (0.025, 0.0))
    claimed_points = int(0.811 / PATH_STEP_M)
    assert claimed_points * PATH_STEP_M > allowed_m(env)
    # ...and it is not the speed rule that catches it: 25 mm in 40 ms is
    # 0.625 m/s, under the cap, so the roll itself looks entirely possible.
    assert env.last_step_implausible is False


def test_corner_cut_is_rejected():
    """4 mm rolled cannot claim the 25 mm of path measured on the real rig."""
    env = make_env()
    roll(env, (0.0, 0.0), (0.004, 0.0))
    claimed_points = int(0.025 / PATH_STEP_M)
    assert claimed_points * PATH_STEP_M > allowed_m(env)


def test_detector_flip_is_flagged_and_not_banked_in_full():
    """The flip: an impossible roll is flagged, and cannot buy allowance."""
    env = make_env()
    roll(env, (0.0, 0.0), (0.120, 0.0))
    assert env.last_step_implausible is True
    # 1.0 m/s over 40 ms caps the deposit at 40 mm, not the reported 120 mm.
    assert env.travel_since_credit == pytest.approx(0.040, abs=2e-3)


def test_min_step_floor_protects_jitter():
    """A claim under the floor is never rejected, however little was rolled."""
    env = make_env()
    roll(env, (0.0, 0.0), (0.0001, 0.0))
    claimed_points = int(0.009 / PATH_STEP_M)
    assert claimed_points * PATH_STEP_M <= allowed_m(env)


def test_nan_position_is_ignored():
    env = make_env()
    roll(env, (0.0, 0.0), (0.005, 0.0))
    banked = env.travel_since_credit
    roll(env, (float("nan"), 0.0))
    assert env.travel_since_credit == banked
    assert env.last_step_implausible is False


def test_slow_frame_relaxes_the_speed_cap_but_not_the_ratio():
    """A 400 ms frame permits 400 mm of roll; the ratio still governs claims."""
    env = make_env()
    roll(env, (0.0, 0.0), (0.120, 0.0), step_dt=0.400)
    assert env.last_step_implausible is False
    assert env.travel_since_credit == pytest.approx(0.120, abs=1e-3)
    claimed_points = int(0.811 / PATH_STEP_M)
    assert claimed_points * PATH_STEP_M > allowed_m(env)


def budget(env):
    """What _get_reward allows for a claim, ratio on or off."""
    if env.anti_cheat_travel_ratio > 0.0:
        return max(
            env.anti_cheat_min_step_m,
            env.anti_cheat_travel_ratio * env.travel_since_credit,
        )
    return env.anti_cheat_max_step_m


def test_ratio_zero_selects_the_original_flat_cap():
    """Ratio 0 must fall back to 57 mm, not collapse to the 10 mm floor."""
    env = make_env(anti_cheat_travel_ratio=0.0, anti_cheat_max_step_m=0.057)
    roll(env, (0.0, 0.0), (0.001, 0.0))
    assert budget(env) == pytest.approx(0.057)
    # A hop is still rejected by it...
    assert 0.811 > budget(env)
    # ...and ordinary motion is not, however little was banked.
    assert 0.020 < budget(env)


def blocked_env(blocked_cols):
    """A 100x100 grid at 0.2 mm/cell, with the given columns blanked."""
    grid = np.zeros((100, 100), dtype=np.int64)
    for c in blocked_cols:
        grid[:, c] = -1
    return types.SimpleNamespace(
        p=types.SimpleNamespace(closest_idx=grid, distance=0.0002)
    )


def crossed(env, x0_mm, x1_mm, y_mm=10.0):
    return TagGym._crossed_blocked(
        env,
        np.array([x0_mm / 1000.0, y_mm / 1000.0]),
        np.array([x1_mm / 1000.0, y_mm / 1000.0]),
    )


def test_step_over_a_thin_trap_is_caught():
    """The whole point: a 0.2 mm line must not be jumpable in one step."""
    env = blocked_env([50])                        # one cell, 0.2 mm wide, at 10 mm
    assert crossed(env, 8.0, 12.0) is True         # a 4 mm step straddling it


def test_step_that_stops_short_of_the_trap_is_not_caught():
    env = blocked_env([50])
    assert crossed(env, 5.0, 9.5) is False


def test_step_beyond_a_wide_trap_is_caught():
    env = blocked_env(range(40, 60))               # 4 mm wide, 8.0-12.0 mm
    assert crossed(env, 4.0, 16.0) is True


def test_a_step_shorter_than_a_cell_is_ignored():
    """Under one cell there is nothing to interpolate and no crossing to find."""
    env = blocked_env([50])
    assert crossed(env, 9.99, 10.0) is False


def test_no_previous_position_means_no_crossing():
    env = blocked_env([50])
    assert TagGym._crossed_blocked(env, None, np.array([0.02, 0.01])) is False


def test_nan_endpoint_means_no_crossing():
    env = blocked_env([50])
    assert TagGym._crossed_blocked(
        env, np.array([0.005, 0.01]), np.array([float("nan"), 0.01])
    ) is False


def test_long_step_is_sampled_densely_enough():
    """A 150 mm step -- the measured maximum -- still finds a single-cell trap."""
    env = blocked_env([50])
    assert crossed(env, 0.5, 150.0) is True


def test_ratio_on_is_tighter_than_the_flat_cap_for_a_small_roll():
    """With the ratio on, 4 mm rolled must not license the flat 57 mm."""
    env = make_env(anti_cheat_travel_ratio=3.0, anti_cheat_max_step_m=0.057)
    roll(env, (0.0, 0.0), (0.004, 0.0))
    assert budget(env) == pytest.approx(0.012)
    assert budget(env) < env.anti_cheat_max_step_m
