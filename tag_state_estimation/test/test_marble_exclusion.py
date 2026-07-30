"""Denying the corner search a marble it already knows about.

Corner dots and the marble are the same blue, so the corner windows separate them
by blob size: dots occupy 42-65 px2 and a marble 125-130 px2, which do not
overlap. That holds until the marble reaches the edge of a window. A marble is
12.8 px across and the window is 25x25, so a marble half inside presents about
64 px2 -- inside the dot range -- and being half inside the window is exactly what
"the marble rolled into the corner" means.

Position separates them where size cannot, because the marble was located on the
previous frame. These tests pin down that the erase disc removes a marble and
cannot reach a dot: their centres are 10 mm apart at closest, about 19 px here,
against a 10 px disc.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

cv2 = pytest.importorskip("cv2")

from tag_state_estimation.core.detection import Detector  # noqa: E402


class Stub:
    """Only what _erase_known_marble reads, so nothing else is under test."""

    def __init__(self, exclude_px, exclude_radius_px):
        self.exclude_px = exclude_px
        self.exclude_radius_px = exclude_radius_px


def erase(mask, exclude_px, radius, origin_rc):
    return Detector._erase_known_marble(Stub(exclude_px, radius), mask, origin_rc)


def blob(mask, centre_xy, radius):
    cv2.circle(mask, (int(centre_xy[0]), int(centre_xy[1])), radius, 255, -1)
    return mask


def window(origin_rc=(100, 200), size=25):
    """A blank corner window and its top-left in full-frame [row, col]."""
    return np.zeros((size, size), np.uint8), np.asarray(origin_rc, dtype=np.float64)


def test_a_marble_at_the_window_edge_is_erased():
    """The case size cannot catch: clipped marble, dot-sized area."""
    mask, origin = window()
    blob(mask, (23, 12), 6)                     # half out of a 25 px window
    assert mask.sum() > 0
    full_frame_xy = np.asarray([origin[1] + 23, origin[0] + 12])
    out = erase(mask, full_frame_xy, 10.0, origin)
    assert out.sum() == 0


def test_a_dot_at_the_window_centre_survives_a_marble_erase():
    """The disc must not reach the dot: centres are ~19 px apart at closest."""
    mask, origin = window()
    blob(mask, (12, 12), 4)                     # the dot
    before = mask.sum()
    marble_xy = np.asarray([origin[1] + 12 + 19, origin[0] + 12])
    out = erase(mask, marble_xy, 10.0, origin)
    assert out.sum() == before


def test_a_marble_outside_this_window_leaves_it_untouched():
    mask, origin = window()
    blob(mask, (12, 12), 4)
    before = mask.sum()
    far_xy = np.asarray([origin[1] + 200.0, origin[0] + 200.0])
    assert erase(mask, far_xy, 10.0, origin).sum() == before


def test_no_known_marble_changes_nothing():
    mask, origin = window()
    blob(mask, (12, 12), 4)
    before = mask.sum()
    assert erase(mask, None, 10.0, origin).sum() == before
    assert erase(mask, np.asarray([210.0, 112.0]), 0.0, origin).sum() == before


def test_a_nan_marble_position_changes_nothing():
    mask, origin = window()
    blob(mask, (12, 12), 4)
    before = mask.sum()
    nan_xy = np.asarray([np.nan, 112.0])
    assert erase(mask, nan_xy, 10.0, origin).sum() == before


def test_the_input_mask_is_not_modified_in_place():
    """detect_corner keeps using the original mask, so this must copy."""
    mask, origin = window()
    blob(mask, (12, 12), 4)
    before = mask.copy()
    erase(mask, np.asarray([origin[1] + 12, origin[0] + 12]), 10.0, origin)
    assert np.array_equal(mask, before)


def test_a_dot_and_a_marble_in_one_window_keeps_only_the_dot():
    mask, origin = window(size=45)
    blob(mask, (10, 10), 4)                     # dot
    blob(mask, (34, 34), 6)                     # marble
    both = mask.sum()
    marble_xy = np.asarray([origin[1] + 34, origin[0] + 34])
    out = erase(mask, marble_xy, 10.0, origin)
    assert 0 < out.sum() < both
    # what survives is on the dot's side of the window
    ys, xs = np.nonzero(out)
    assert xs.max() < 20 and ys.max() < 20
