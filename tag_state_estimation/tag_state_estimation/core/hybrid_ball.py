"""Pure state-selection logic for guarded HSV and AI marble tracking."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HybridBallResult:
    """Selected measurement and diagnostic state for one camera frame."""

    measurement: np.ndarray
    source: str
    disagreement_px: float
    missing_frames: int


def _valid(position):
    return position is not None and np.all(np.isfinite(position))


class HybridBallTracker:
    """Fuse detections, gate reacquisition, and identify prediction-only gaps."""

    def __init__(
        self,
        agreement_radius_px=12.0,
        max_reacquire_jump_px=25.0,
        occlusion_grace_frames=90,
        far_reacquire_confirm_frames=3,
        ai_fusion_weight=0.5,
        trust_hsv_alone=False,
        fast_reacquire_frames=3,
    ):
        self.agreement_radius_px = float(agreement_radius_px)
        self.max_reacquire_jump_px = float(max_reacquire_jump_px)
        self.occlusion_grace_frames = int(occlusion_grace_frames)
        self.far_reacquire_confirm_frames = int(far_reacquire_confirm_frames)
        self.ai_fusion_weight = float(ai_fusion_weight)
        # Only safe when the HSV source is restricted to the maze interior, so
        # that a reference dot cannot be the candidate.
        self.trust_hsv_alone = bool(trust_hsv_alone)
        # Length of gap that still counts as the SAME track rather than a
        # reacquisition. Without it every single-frame miss cost four frames:
        # one really missed, then three spent confirming a candidate the
        # detectors had already agreed on.
        self.fast_reacquire_frames = int(fast_reacquire_frames)
        self.last_position = None
        self.missing_frames = 0
        self.pending_position = None
        self.pending_frames = 0

    @staticmethod
    def _distance(first, second):
        return float(np.linalg.norm(np.asarray(first) - np.asarray(second)))

    def reset(self):
        self.last_position = None
        self.missing_frames = 0
        self.pending_position = None
        self.pending_frames = 0

    def _accept(self, position, source, disagreement=np.nan):
        self.last_position = np.asarray(position, dtype=np.float32).copy()
        self.missing_frames = 0
        self.pending_position = None
        self.pending_frames = 0
        return HybridBallResult(
            self.last_position.copy(), source, float(disagreement), 0
        )

    def _missing(self):
        self.missing_frames += 1
        measurement = np.array([np.nan, np.nan], dtype=np.float32)
        if (
            self.last_position is not None
            and self.missing_frames <= self.occlusion_grace_frames
        ):
            source = "kalman_occlusion"
        else:
            source = "lost"
        return HybridBallResult(
            measurement, source, np.nan, self.missing_frames
        )

    def _travel_budget(self, frames):
        """Pixels the marble may legitimately cover in ``frames`` frame periods."""
        return self.max_reacquire_jump_px * float(max(1, frames))

    def _resume(self, position, source):
        """Continue the existing track across a short gap, or return None.

        A gap of a frame or two does not invalidate a track: the marble is still
        within reach of where it was, and demanding the full multi-frame
        confirmation there turned one missed frame into four. The distance test
        scales with the gap because the marble keeps rolling while unobserved.
        """
        if self.last_position is None:
            return None
        if self.missing_frames > self.fast_reacquire_frames:
            return None
        travelled = self._distance(position, self.last_position)
        if travelled > self._travel_budget(self.missing_frames + 1):
            return None
        return self._accept(position, source)

    def _confirm_reacquisition(self, position, source):
        """Require a few frames of a coherent track before trusting a candidate.

        Coherent means "no further than a marble can travel in one frame", tested
        against the PREVIOUS candidate. It used to be tested against a running
        average of the candidates, using agreement_radius_px -- the threshold for
        two detectors agreeing about one frame, not for how far a marble moves
        between frames. A moving marble outran that average: at 10 px per frame
        (0.42 m/s here) the counter reset every second frame and never reached
        the threshold, so a marble that both detectors could see plainly stayed
        unreacquirable until it slowed down.
        """
        position = np.asarray(position, dtype=np.float32)
        if (
            self.pending_position is not None
            and self._distance(position, self.pending_position)
            <= self._travel_budget(1)
        ):
            self.pending_frames += 1
        else:
            self.pending_frames = 1
        self.pending_position = position.copy()
        if self.pending_frames >= self.far_reacquire_confirm_frames:
            # Accept where the marble is NOW. Accepting an average over the
            # confirmation window hands the filter a point the marble has
            # already left, and the next frame then reads as a jump.
            return self._accept(self.pending_position, source)
        return None

    def update(self, hsv_position=None, ai_position=None):
        hsv_valid = _valid(hsv_position)
        ai_valid = _valid(ai_position)
        disagreement = np.nan

        if hsv_valid and ai_valid:
            hsv_position = np.asarray(hsv_position, dtype=np.float32)
            ai_position = np.asarray(ai_position, dtype=np.float32)
            disagreement = self._distance(hsv_position, ai_position)
            if disagreement <= self.agreement_radius_px:
                ai_weight = self.ai_fusion_weight
                fused = (1.0 - ai_weight) * hsv_position + ai_weight * ai_position
                if self.last_position is None or self.missing_frames > 0:
                    resumed = self._resume(fused, "fused_resumed")
                    if resumed is not None:
                        return resumed
                    confirmed = self._confirm_reacquisition(
                        fused, "fused_reacquired_confirmed"
                    )
                    return confirmed if confirmed is not None else self._missing()
                return self._accept(fused, "fused", disagreement)

            # AI is authoritative when the detectors disagree. A nearby AI
            # candidate may continue tracking immediately; a distant one must
            # pass the same multi-frame confirmation as any AI reacquisition.
            if self.last_position is not None:
                ai_jump = self._distance(ai_position, self.last_position)
                if ai_jump <= self.max_reacquire_jump_px:
                    return self._accept(ai_position, "ai_disagreement", disagreement)
            confirmed = self._confirm_reacquisition(
                ai_position, "ai_reacquired_confirmed"
            )
            return confirmed if confirmed is not None else self._missing()

        if hsv_valid:
            hsv_position = np.asarray(hsv_position, dtype=np.float32)
            if not self.trust_hsv_alone:
                # Kept for an unmasked HSV source: blue board markers satisfy the
                # colour filter, so a bare candidate can be a dot on the rim.
                return self._missing()
            # With a masked source that ambiguity is gone geometrically -- the
            # dots sit 4-5 mm outside the maze edge and the search is clipped to
            # the maze -- so HSV alone is allowed to carry the frame. This is the
            # case the pairing exists for: the learned detector misses, most often
            # on a fast marble, and colour still has it.
            if self.last_position is None or self.missing_frames > 0:
                resumed = self._resume(hsv_position, "hsv_resumed")
                if resumed is not None:
                    return resumed
                confirmed = self._confirm_reacquisition(
                    hsv_position, "hsv_reacquired_confirmed"
                )
                return confirmed if confirmed is not None else self._missing()
            jump = self._distance(hsv_position, self.last_position)
            if jump <= self.max_reacquire_jump_px:
                return self._accept(hsv_position, "hsv_only")
            confirmed = self._confirm_reacquisition(
                hsv_position, "hsv_reacquired_confirmed"
            )
            return confirmed if confirmed is not None else self._missing()

        if ai_valid:
            ai_position = np.asarray(ai_position, dtype=np.float32)
            if self.last_position is None or self.missing_frames > 0:
                resumed = self._resume(ai_position, "ai_resumed")
                if resumed is not None:
                    return resumed
                confirmed = self._confirm_reacquisition(
                    ai_position, "ai_reacquired_confirmed"
                )
                return confirmed if confirmed is not None else self._missing()
            jump = self._distance(ai_position, self.last_position)
            if jump <= self.max_reacquire_jump_px:
                return self._accept(ai_position, "ai_reacquired")

            confirmed = self._confirm_reacquisition(
                ai_position, "ai_reacquired_confirmed"
            )
            if confirmed is not None:
                return confirmed

        return self._missing()
