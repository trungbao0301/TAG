import numpy as np
import cv2

from cyberrunner_state_estimation.core import gaussian_robust, masking
from cyberrunner_state_estimation.core.hybrid_ball import HybridBallTracker
from cyberrunner_state_estimation.core.hole_mask import (
    TimedHoleRejector,
    candidate_hole_index,
)
from cyberrunner_state_estimation.ai_marble_common import OnnxMarbleDetector

colors = [(255, 0, 255), (0, 255, 0), (0, 0, 255), (0, 255, 255)]
c_name = ["blue", "green", "red", "yellow"]


# from profileFIle import profile
class Detector:

    DEFAULT_HSV_CORNERS = (
        (43, 140),  # (minHue, maxHue)
        (125, 255),  # (minSat, maxSat)
        (9, 255),
    )  # (minVal, maxVal)
    DEFAULT_Q_CORNERS = 5  # gaussian detection param -> q-th quentile
    DEFAULT_TH_CORNERS = 0.002  # gaussian detection threshold

    DEFAULT_HSV_BALL = (
        (60, 116),  # (minHue, maxHue) — blue (locked-WB camera, saturation 40)
        (162, 255),  # (minSat, maxSat)
        (50, 243),
    )  # (minVal, maxVal)
    DEFAULT_Q_BALL = 6  # gaussian detection param -> q-th quentile
    DEFAULT_TH_BALL = 10 ** (-4)  # gaussian detection threshold

    DEFAULT_SIZE_CROP_CORNERS = 65 / 3
    DEFAULT_SIZE_CROP_BALL = 80.0
    BALL_MISS_THRESHOLD = 6  # consecutive missed frames before declaring ball lost
    HSV_CROP_RESET_MISSING_FRAMES = 6

    DEFAULT_INIT_BALL_POS = np.array([47, 330])  # np.array([55,485])

    def __init__(
        self,
        markers,
        hsv_params_corners: list = DEFAULT_HSV_CORNERS,
        q_corners: float = DEFAULT_Q_CORNERS,
        th_corners: float = DEFAULT_TH_CORNERS,
        hsv_params_ball: list = DEFAULT_HSV_BALL,
        q_ball: float = DEFAULT_Q_BALL,
        th_ball: float = DEFAULT_TH_BALL,
        ball_init_pos: np.ndarray = DEFAULT_INIT_BALL_POS,
        corner_subimage_half_size=25,
        show_subimages=False,
        acceleration_backend="cpu",
        ai_mode="off",
        ai_model_path=None,
        ai_confidence_threshold=0.90,
        ai_valid_roi=(0.25, 0.15, 0.72, 0.80),
        ai_check_every_n_frames=5,
        ai_agreement_radius_px=12.0,
        ai_max_reacquire_jump_px=25.0,
        ai_occlusion_grace_frames=90,
        ai_far_reacquire_confirm_frames=3,
        ai_hole_rejection_enabled=True,
        ai_hole_rejection_margin_m=0.0025,
        ai_hole_rejection_delay_sec=2.0,
        ai_roi_follows_corners=True,
        ai_roi_corner_inset_px=0.0,
        ai_corner_mask_radius_px=12.0,
    ):

        self.hsv_params_corners = hsv_params_corners
        self.q_corners = q_corners
        self.th_corners = th_corners
        self.hsv_params_ball = hsv_params_ball
        self.q_ball = q_ball
        self.th_ball = th_ball

        self.ball_pos = None
        self.corners = None
        self.show_subimages = show_subimages
        self.acceleration_backend = acceleration_backend
        self.ai_mode = str(ai_mode).lower()
        if self.ai_mode not in ("off", "shadow", "hybrid"):
            raise ValueError(f"Unsupported ai_mode: {ai_mode}")
        self.ai_check_every_n_frames = max(1, int(ai_check_every_n_frames))
        self.ai_hole_rejection_enabled = bool(ai_hole_rejection_enabled)
        self.ai_hole_rejection_margin_m = max(
            0.0, float(ai_hole_rejection_margin_m)
        )
        self.ai_hole_rejector = TimedHoleRejector(ai_hole_rejection_delay_sec)
        # The static ai_valid_roi is fixed in IMAGE space, but the board moves in
        # the image as the plate tilts. Measured margin between the board's corner
        # dots and that rectangle was +6 px at the top and +1 px at the bottom, so
        # past ~10 deg of tilt part of the board fell outside the ROI and a marble
        # there became undetectable (measured loss: 0% below 6 deg, 78% at 10-12
        # deg, 85% above 14 deg).
        #
        # So derive the ROI from the tracked corner dots each frame instead, and
        # keep it aligned to the dot bounding box rather than inset.
        #
        # An inset does NOT work here: a marble running along the top edge sits at
        # the same image ROWS as the top corner dots (observed marble at row 70,
        # dots at rows 66-71) because they are separated horizontally, not
        # vertically. No rectangle can include one and exclude the other. The blue
        # dots do look like the marble, so they are masked separately as discs --
        # the same protection detect_ball() already gets via mask_corner=True.
        self.ai_roi_follows_corners = bool(ai_roi_follows_corners)
        self.ai_roi_corner_inset_px = max(0.0, float(ai_roi_corner_inset_px))
        self.ai_corner_mask_radius_px = max(0.0, float(ai_corner_mask_radius_px))
        self.last_ai_roi = None
        self.ai_detector = None
        if self.ai_mode != "off":
            if not ai_model_path:
                raise ValueError("ai_model_path is required when ai_mode is not off")
            self.ai_detector = OnnxMarbleDetector(
                ai_model_path,
                confidence_threshold=ai_confidence_threshold,
                valid_roi=ai_valid_roi,
            )
        self.hybrid_tracker = HybridBallTracker(
            agreement_radius_px=ai_agreement_radius_px,
            max_reacquire_jump_px=ai_max_reacquire_jump_px,
            occlusion_grace_frames=ai_occlusion_grace_frames,
            far_reacquire_confirm_frames=ai_far_reacquire_confirm_frames,
        )
        self.frame_index = 0
        self.last_hsv_found = False
        self.last_ai_confidence = np.nan
        self.last_ai_checked = False
        self.last_ball_source = "hsv"
        self.last_detection_disagreement_px = np.nan
        self.last_ai_rejected_hole = -1
        self.last_ai_hole_candidate = -1
        self.last_ai_hole_elapsed_sec = 0.0
        self.hybrid_missing_frames = 0

        self.corners_missing = True
        self.corner_found = np.zeros(4, dtype=bool)

        self.fixed_corners = None
        self.is_ball_found = False
        self.consecutive_ball_misses = 0

        self.corner_subimage_half_size = corner_subimage_half_size
        corners = np.repeat(
            np.expand_dims(np.asarray(markers)[:, ::-1], axis=1), 2, axis=1
        )
        corners[:, 0] -= self.corner_subimage_half_size
        corners[:, 1] += self.corner_subimage_half_size
        self.default_coords_subimages_corners = corners.astype(int)
        self.default_coords_subimage_ball = (
            self.default_coords_subimages_corners[3, 0],
            self.default_coords_subimages_corners[1, 1],
        )

    # @profile(sort_by='cumulative', lines_to_print=10, strip_dirs=True)
    def process_frame(self, frame):
        """
        Process a frame to get raw coordinates of the corners and marble.

        Args :
            frame: np.ndarray, dim: (400, 640)
        Returns :
            corners: np.ndarray, dim: (4,2)
                     raw corner coordinates in (row, column) convention.
            ball: np.ndarray, dim: (2,)
                     raw marble coordinates in (row, column) convention.
        """

        ai_frame = frame.copy() if self.ai_detector is not None else None
        corners = self.detect_corners(frame)
        ball = self.detect_ball(frame, show_rectangle=True, mask_corner=True)
        if self.ai_detector is not None:
            ball = self._select_hybrid_ball(ai_frame, ball)
        else:
            if self.last_hsv_found:
                self.last_ball_source = "hsv"
            elif np.all(np.isfinite(ball)):
                self.last_ball_source = "hsv_hold"
            else:
                self.last_ball_source = "lost"
        self.frame_index += 1
        return corners, ball  # both in (x,y) conventions

    def _ai_roi_from_corners(self, frame_shape):
        """Normalised ROI inset inside the tracked corner dots, or None.

        None means "fall back to the configured static ROI" -- used when the
        corners are missing, so a lost-corner frame cannot silently widen the
        searchable region out to the blue dots or the robot arm.
        """
        if not self.ai_roi_follows_corners:
            return None
        if self.corners is None or self.corners_missing:
            return None
        corners = np.asarray(self.corners, dtype=np.float32)
        if corners.shape != (4, 2) or not np.all(np.isfinite(corners)):
            return None
        height, width = frame_shape[:2]
        rows, cols = corners[:, 0], corners[:, 1]
        inset = self.ai_roi_corner_inset_px
        x_min = (float(cols.min()) + inset) / width
        x_max = (float(cols.max()) - inset) / width
        y_min = (float(rows.min()) + inset) / height
        y_max = (float(rows.max()) - inset) / height
        x_min, y_min = max(0.0, x_min), max(0.0, y_min)
        x_max, y_max = min(1.0, x_max), min(1.0, y_max)
        # A degenerate box would mask the whole heatmap and report "not visible"
        # on every frame, which looks exactly like a detector failure.
        if not (x_min < x_max and y_min < y_max):
            return None
        return (x_min, y_min, x_max, y_max)

    def _select_hybrid_ball(self, frame, hsv_ball):
        hsv_position = self.ball_pos.copy() if self.last_hsv_found else None
        # Hybrid mode requires an AI decision for every selected measurement.
        # Shadow mode may still sample AI less often to reduce diagnostic cost.
        should_check_ai = self.ai_mode == "hybrid" or (
            not self.last_hsv_found
            or self.frame_index % self.ai_check_every_n_frames == 0
        )
        ai_position = None
        self.last_ai_checked = should_check_ai
        self.last_ai_confidence = np.nan
        self.last_ai_rejected_hole = -1
        self.last_ai_hole_candidate = -1
        self.last_ai_hole_elapsed_sec = 0.0
        hole_timer_updated = False
        if should_check_ai:
            roi = self._ai_roi_from_corners(frame.shape)
            self.last_ai_roi = (
                roi
                if roi is not None
                else getattr(self.ai_detector, "valid_roi", None)
            )
            # Mask the blue corner dots: same colour family as the marble, and a
            # rectangle cannot exclude them without also excluding an edge marble.
            dots = None
            if (
                self.ai_corner_mask_radius_px > 0.0
                and self.corners is not None
                and not self.corners_missing
            ):
                corners = np.asarray(self.corners, dtype=np.float32)
                if corners.shape == (4, 2) and np.all(np.isfinite(corners)):
                    dots = corners[:, ::-1]  # (row,col) -> (x_px, y_px)
            try:
                ai_detection = self.ai_detector.detect(
                    frame,
                    valid_roi=roi,
                    exclude_centers_px=dots,
                    exclude_radius_px=(
                        self.ai_corner_mask_radius_px if dots is not None else 0.0
                    ),
                )
            except TypeError:
                # Keep small diagnostic/test detector implementations compatible;
                # the production ONNX detector accepts the frame-local masks.
                ai_detection = self.ai_detector.detect(frame)
            self.last_ai_confidence = ai_detection.confidence
            if ai_detection.visible:
                # The classical detector uses [row, column], while the AI
                # detector reports conventional image [x, y] coordinates.
                ai_position = np.array(
                    [ai_detection.y_px, ai_detection.x_px], dtype=np.float32
                )
                if self.ai_hole_rejection_enabled and not self.corners_missing:
                    hole_index = candidate_hole_index(
                        ai_position,
                        self.corners,
                        margin_m=self.ai_hole_rejection_margin_m,
                    )
                    if hole_index is not None:
                        self.last_ai_hole_candidate = hole_index
                        reject_hole, elapsed_sec = self.ai_hole_rejector.update(
                            hole_index
                        )
                        hole_timer_updated = True
                        self.last_ai_hole_elapsed_sec = elapsed_sec
                        if reject_hole:
                            self.last_ai_rejected_hole = hole_index
                            ai_position = None
        if not hole_timer_updated:
            self.ai_hole_rejector.update(None)

        if self.ai_mode == "shadow":
            if self.last_hsv_found:
                self.last_ball_source = "shadow_hsv"
            elif np.all(np.isfinite(hsv_ball)):
                self.last_ball_source = "shadow_hsv_hold"
            else:
                self.last_ball_source = "shadow_lost"
            if hsv_position is not None and ai_position is not None:
                self.last_detection_disagreement_px = float(
                    np.linalg.norm(hsv_position - ai_position)
                )
            else:
                self.last_detection_disagreement_px = np.nan
            return hsv_ball

        result = self.hybrid_tracker.update(hsv_position, ai_position)
        self.last_ball_source = (
            f"ai_hole_rejected_{self.last_ai_rejected_hole + 1}"
            if self.last_ai_rejected_hole >= 0
            else (
                f"ai_hole_pending_{self.last_ai_hole_candidate + 1}"
                if self.last_ai_hole_candidate >= 0
                else result.source
            )
        )
        self.last_detection_disagreement_px = result.disagreement_px
        self.hybrid_missing_frames = result.missing_frames
        if np.all(np.isfinite(result.measurement)):
            self.ball_pos = result.measurement.copy()
            self.is_ball_found = True
            self.consecutive_ball_misses = 0
            return result.measurement

        if result.source == "kalman_occlusion":
            # Retain the crop center internally, but give the Kalman filter a
            # missing measurement so it performs a dynamics-only prediction.
            if (
                result.missing_frames
                < Detector.HSV_CROP_RESET_MISSING_FRAMES
            ):
                self.ball_pos = self.hybrid_tracker.last_position.copy()
                self.is_ball_found = True
            else:
                # Keep the Kalman track, but force HSV back to a full-board
                # search so a stale predictive crop cannot remain locked.
                self.ball_pos = None
                self.is_ball_found = False
        else:
            self.ball_pos = None
            self.is_ball_found = False
        return np.array([np.nan, np.nan], dtype=np.float32)

    def reset_ball_tracking(self):
        """Forget the last ball crop so the next frame searches the full board."""
        self.is_ball_found = False
        self.ball_pos = None
        self.consecutive_ball_misses = 0
        self.hybrid_tracker.reset()
        self.ai_hole_rejector.reset()

    def restore_ball_tracking(self, position):
        """Restore a known-valid crop center after rejecting an outside candidate."""
        self.ball_pos = np.asarray(position, dtype=np.float32).copy()
        self.is_ball_found = True
        self.hybrid_tracker.last_position = self.ball_pos.copy()
        self.hybrid_tracker.missing_frames = 0
        self.ai_hole_rejector.reset()

    def get_cropped(self, im: np.ndarray, pos: np.ndarray, h_p: float, w_p: float):
        """
        Return a crop and its upper-left and lower-right image coordinates.
        Args :
            im: np.ndarray
                image
            pos: np.ndarray
                 position of the center of the subimage
            h_p: float
                 height of the subimage
            w_p: float
                 width of the subimage
        Returns :
            im_cropped: np.ndarray
            ul: np.ndarray, dim: (2,)
                top-left corner coordinates in the given image.
            dr: np.ndarray, dim: (2,)
                down-right corner coordinates in the given image.
        """
        h, w = im.shape[:2]
        ul_x = min(h - 1, max(0, int(pos[0] - h_p / 2)))
        ul_y = min(w - 1, max(0, int(pos[1] - w_p / 2)))
        dr_x = min(h - 1, max(0, int(pos[0] + h_p / 2)))
        dr_y = min(w - 1, max(0, int(pos[1] + w_p / 2)))
        im_cropped = im[ul_x:dr_x, ul_y:dr_y]
        ul = np.array([ul_x, ul_y])
        dr = np.array([dr_x, dr_y])
        return im_cropped, ul, dr

    def predictive_cropping_corners(self, im: np.ndarray):
        h, w = im.shape[:2]
        h_p, w_p = (
            Detector.DEFAULT_SIZE_CROP_CORNERS,
            Detector.DEFAULT_SIZE_CROP_CORNERS,
        )
        subimgs = []
        subcoords = []
        for i in range(4):
            subimg, ul, dr = self.get_cropped(im, self.corners[i, :], h_p, w_p)
            subimgs.append(subimg)
            subcoords.append((ul, dr))
        return subimgs, subcoords

    def predictive_cropping_ball(self, im: np.ndarray, draw: bool = False):
        h_p, w_p = Detector.DEFAULT_SIZE_CROP_BALL, Detector.DEFAULT_SIZE_CROP_BALL
        im_cropped, ul, dr = self.get_cropped(im, self.ball_pos, h_p, w_p)
        if draw:
            im = cv2.rectangle(
                im, tuple(ul[::-1]), tuple(dr[::-1]), (0, 255, 0), 1
            )  # need to do im =.. ? or just remove the im = ??
        return im_cropped, ul

    def is_ball_in_corner(self):  # ball pos in in (x,y)
        if self.ball_pos[0] < 100 and self.ball_pos[1] < 200:
            return 3
        if self.ball_pos[0] < 100 and self.ball_pos[1] > 450:
            return 2
        if self.ball_pos[0] > 300 and self.ball_pos[1] > 450:
            return 1
        if self.ball_pos[0] > 300 and self.ball_pos[1] < 200:
            return 0
        return None

    def get_default_subimages_corners(self, im: np.ndarray, show: bool = False):
        h, w = im.shape[:2]
        if show:
            for c in self.default_coords_subimages_corners:
                cv2.rectangle(im, c[0][::-1], c[1][::-1], (0, 0, 255), 1)
        # TODO use get_cropped
        subimages = [
            im[cs[0][0] : cs[1][0], cs[0][1] : cs[1][1]]
            for cs in self.default_coords_subimages_corners
        ]
        return subimages, self.default_coords_subimages_corners

    def detect_corners(self, frame):
        corners = np.zeros((4, 2), dtype="float32")

        if self.corners is None or self.corners_missing:
            (
                cropped_corners_imgs,
                subcoords_corners_imgs,
            ) = self.get_default_subimages_corners(frame)
        else:
            (
                cropped_corners_imgs,
                subcoords_corners_imgs,
            ) = self.predictive_cropping_corners(frame)

        missing = False
        found_mask = np.zeros(4, dtype=bool)
        for i, sub_im in enumerate(cropped_corners_imgs):
            corners[i, :], found = self.detect_corner(
                sub_im, i, subcoords_corners_imgs[i][0]
            )
            found_mask[i] = bool(found)
            missing = missing or not found
        self.corner_found = found_mask
        self.corners_missing = missing

        self.corners = corners
        return corners

    def detect_corner(self, sub_im: np.ndarray, i: int, coords_ul_sub_im: np.ndarray):
        sub_masked, mask = masking.mask_hsv(
            sub_im, self.hsv_params_corners, self.acceleration_backend
        )
        c_local, found = gaussian_robust.detect_gaussian(
            mask, i, self.q_corners, self.th_corners, show_sub=self.show_subimages
        )
        c = (coords_ul_sub_im + c_local).astype("float32")
        return c, found

    def detect_ball(
        self,
        im: np.ndarray,
        show_rectangle: bool = False,
        mask_corner=False,
        mask_initial=True,
    ):

        # These blue reference markers move in image space as the board tilts.
        # Mask their freshly detected positions on every frame, including when
        # the marble tracker is using a predictive crop. The AI copy was made
        # before this masking, so a real marble near a marker can still be
        # accepted by the AI-authoritative hybrid path.
        if mask_corner or mask_initial:
            for marker_group in (self.corners, self.fixed_corners):
                if marker_group is None:
                    continue
                for marker in marker_group:
                    if np.all(np.isfinite(marker)):
                        cv2.circle(
                            im,
                            tuple(np.round(marker).astype(int)[::-1]),
                            18,
                            (0, 0, 255),
                            -1,
                        )

        if self.is_ball_found:
            if mask_corner:
                corner_ball = self.is_ball_in_corner()
                if (
                    corner_ball is not None
                ):  # masking the corner that is in vicinity of the ball
                    cv2.circle(
                        im,
                        tuple(self.corners[corner_ball, :].astype(int)[::-1]),
                        16,
                        (0, 0, 255),
                        -1,
                    )
            cropped_ball_im, coords_ul_cropped_img = self.predictive_cropping_ball(
                im, draw=show_rectangle
            )
        else:
            # if self.ball_pos is not None:
            # print("no ball in the frame")
            if mask_initial:
                for i in range(4):
                    cv2.circle(
                        im,
                        tuple(np.round(self.corners[i, :]).astype(int)[::-1]),
                        16,
                        (0, 0, 255),
                        -1,
                    )
                    cv2.circle(
                        im,
                        tuple(np.round(self.fixed_corners[i, :]).astype(int)[::-1]),
                        16,
                        (0, 0, 255),
                        -1,
                    )

            ul, dr = self.default_coords_subimage_ball
            cropped_ball_im, coords_ul_cropped_img = (
                im[ul[0] : dr[0], ul[1] : dr[1], :],
                ul,
            )
        sub_masked, mask = masking.mask_hsv(
            cropped_ball_im, self.hsv_params_ball, self.acceleration_backend
        )
        c_local, found = gaussian_robust.detect_gaussian(
            mask, 4, self.q_ball, self.th_ball, show_sub=self.show_subimages
        )
        if found:
            self.consecutive_ball_misses = 0
            self.is_ball_found = True
            self.last_hsv_found = True
            c = (coords_ul_cropped_img + c_local).astype("float32")  # (x,y)
            self.ball_pos = c
            return c

        self.last_hsv_found = False
        self.consecutive_ball_misses += 1
        if self.consecutive_ball_misses >= Detector.BALL_MISS_THRESHOLD:
            self.is_ball_found = False
            return np.array([np.nan, np.nan])
        # Within hysteresis window: keep searching from last known position
        return self.ball_pos if self.ball_pos is not None else np.array([np.nan, np.nan])

    def draw_corners(self, frame: np.ndarray):
        for i in range(self.corners.shape[0]):
            cv2.drawMarker(
                frame,
                (round(self.corners[i, 1]), round(self.corners[i, 0])),
                colors[i],
                cv2.MARKER_TILTED_CROSS,
                5,
                1,
            )  # (u,v)
        return

    def draw_ball(self, frame: np.ndarray):
        cv2.drawMarker(
            frame,
            tuple((np.round(self.ball_pos).astype(int))[::-1]),
            (0, 0, 255),
            cv2.MARKER_TILTED_CROSS,
            5,
            1,
        )  # (u,v)

    def reset(self, ball_pos_init: np.ndarray = DEFAULT_INIT_BALL_POS):
        self.corners = None
        self.ball_pos = ball_pos_init
        self.hybrid_tracker.reset()
        self.ai_hole_rejector.reset()


# TODO remove
class DetectorFixedPts(Detector):
    def __init__(
        self, markers, show_subimages: bool = False, acceleration_backend="cpu"
    ):
        hsv_corners = (
            (43, 140),  # (minHue, maxHue)
            (125, 255),  # (minSat, maxSat)
            (40, 255),  # (minVal, maxVal)
        )
        super().__init__(
            markers,
            hsv_params_corners=hsv_corners,
            corner_subimage_half_size=12,
            show_subimages=show_subimages,
            acceleration_backend=acceleration_backend,
        )

    def detect_corner(self, sub_im: np.ndarray, i: int, coords_ul_sub_im: np.ndarray):
        sub_masked, mask = masking.mask_hsv(
            sub_im, self.hsv_params_corners, self.acceleration_backend
        )
        c_local, blob_found = gaussian_robust.detect_gaussian(
            mask, i, self.q_corners, self.th_corners, show_sub=self.show_subimages
        )
        c = (coords_ul_sub_im + c_local).astype("float32")

        # TODO use ROS logger
        if not blob_found:
            print("Unable to find corner {}".format(i + 1))
            exit()

        return c, blob_found
