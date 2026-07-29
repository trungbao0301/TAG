import json
import os
import cv2
import numpy as np
from ament_index_python.packages import get_package_share_directory


colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]
c_name = ["blue", "green", "red", "yellow"]


class PlatePoseEstimator:

    # constants
    L_EXT_INT_X = 0.305  # measured outer frame span along x
    L_EXT_INT_Y = 0.274
    # Fitted from 40 tilted views / 745 hole observations by
    # tools/fit_marker_geometry.py, NOT the ETH original's 0.269 x 0.237. Those
    # were inherited nominal values for different hardware; this board runs a
    # custom maze. Profiling the dot-plane height h against the known DXF hole
    # positions gives a clear minimum at h = 10 mm (median residual 2.13 mm at
    # h=0, 1.49 mm at h=10, 1.92 mm at h=20), independently reproducing a 1 cm
    # ruler measurement, and at that optimum the spacing is 249.2 x 222.3 mm.
    # Median hole residual improves 5.67 -> 1.49 mm versus the old constants.
    C2C_X = 0.2492  # moving-marker center spacing along x
    C2C_Y = 0.2223  # moving-marker center spacing along y

    r = 0.008 / 2
    R_BALL = 0.012 / 2

    # Inset of the four fixed frame dots from the top/bottom of L_EXT_INT_Y.
    # Solved from markers.csv: the undistorted dots have an x/y span ratio of
    # 1.6357, so an x span of L_EXT_INT_X + 2r = 313.0 mm implies a y span of
    # 191.4 mm, i.e. an inset of (274 - 191.4) / 2. The previous 0.05 guess
    # made the span ratio 1.868, which drove camera_localization() into a wrong
    # PnP minimum (camera at x=0.015 m instead of 0.151 m) and biased the
    # resting plate angles by about -10 deg in beta.
    FIXED_DOT_INSET_Y = 0.0413

    MODEL_POINTS_FIXED_CORNERS = np.array(
        [
            (-r, FIXED_DOT_INSET_Y, 0),  # Corner 1
            (L_EXT_INT_X + r, FIXED_DOT_INSET_Y, 0),  # Corner 2
            (L_EXT_INT_X + r, L_EXT_INT_Y - FIXED_DOT_INSET_Y, 0),  # Corner 3
            (-r, L_EXT_INT_Y - FIXED_DOT_INSET_Y, 0),  # Corner 4
        ],
        dtype=np.float32,
    )

    # Fallback pinhole intrinsics, used only when pinhole_calib.json is absent.
    #
    # This class used to run every point through an OcamModel built from
    # calib_results_cyberrunner.txt. That file describes a 1920x1200 capture
    # while the pipeline runs 1280x720 downscaled to 640x400, so the polynomial
    # did not transfer. Measured on markers.csv, the ocam pass changed the dot
    # spacing by 1.93x and left both aspect ratios identical to four decimal
    # places -- no shape correction at all, only a wrong magnification. And
    # because get_pose_T__C_P solvePnPs with the same f that the reprojection
    # used, f cancelled and the effective focal length became the polynomial's
    # on-axis 673.2 px instead of the true ~298, putting the camera at 0.556 m
    # rather than a ruler-measured 0.290 m and scaling every derived angle by
    # 2.26x. No rescale of that file fixes it: OcamModel.scale is
    # angle-preserving, and scaling its coefficients is algebraically just a
    # change of f. The model and its calibration file are therefore gone.
    #
    # FALLBACK_PRINCIPAL_POINT is the frame centre. The retired calibration's
    # own centre, scaled by 3, was row 199.6 / col 319.6 -- within 0.4 px --
    # because the 16:10 -> 16:9 crop drops 60 rows per side, which is 20 rows
    # after /3, exactly the border fast_camera_publisher_v2.py adds back.
    FALLBACK_FOCAL_PX = 300.0
    FALLBACK_PRINCIPAL_POINT = (320.0, 200.0)  # (column, row) for a 640x400 frame

    def __init__(self, print_details: bool = False):
        self.print_details = print_details

        cx, cy = PlatePoseEstimator.FALLBACK_PRINCIPAL_POINT
        self.f = PlatePoseEstimator.FALLBACK_FOCAL_PX
        self.K = np.array([[self.f, 0, cx], [0, self.f, cy], [0, 0, 1]])
        self.dist = None
        self.calib_path = None

        # pinhole_calib.json, written by tools/calibrate_camera_holes.py, is the
        # real calibration and supersedes the fallback above.
        share = get_package_share_directory("cyberrunner_state_estimation")
        calib_path = os.path.join(share, "pinhole_calib.json")
        if os.path.isfile(calib_path):
            with open(calib_path) as handle:
                calib = json.load(handle)
            self.K = np.array(
                [
                    [calib["fx"], 0.0, calib["cx"]],
                    [0.0, calib["fy"], calib["cy"]],
                    [0.0, 0.0, 1.0],
                ]
            )
            dist = np.asarray(calib.get("dist", []), dtype=np.float64).reshape(1, -1)
            self.dist = dist if dist.size and np.any(dist) else None
            self.f = float(calib["fx"])
            self.calib_path = calib_path

        self.T__W_M = None
        self.T__W_C = None
        self.img_points_corners_undist = None
        self.img_points_fixed_corners_undist = None

    def get_pose_T__C_P(
        self, model_points: np.ndarray, img_points: np.ndarray, print_=False
    ):
        """
        Compute the pose of the frame {p} in which model points are expressed wrt to the camera frame {c}.

        Args :
            model_points: np.ndarray, dim: (4,3)
                           3d coordinates of the points in their frame {p}.
            img_points: np.ndarray, dim: (4,2)
                        undistorted image coordinates of the maze corners dots in (x,y) = (line, column) convention.

        Returns :
            T__C_P: np.ndarray, dim: (4,4)
                  pose in SE(3) of the frame {p} in which model points are expressed wrt to the camera frame {c}.
            R__C_P: np.ndarray, dim: (3,3)
                  rotation matrix of T__C_P.
            P__C: np.ndarray, dim: (3,)
                  translation vector of T__C_P.

        """
        img_points = np.flip(
            img_points, axis=1
        )  # conversion to opencv convention: (u,v) = (column, line)
        _, rotation_vec, translation_vec = cv2.solvePnP(
            model_points, img_points, self.K, None, flags=cv2.SOLVEPNP_ITERATIVE
        )
        rotation_mat, _ = cv2.Rodrigues(rotation_vec)

        if self.print_details:
            print("rot vec [deg]:")
            print(180 / np.pi * rotation_vec)
            print("tr vec:")
            print(translation_vec)

        R__C_P = rotation_mat
        P__C = translation_vec
        T__C_P = np.hstack((R__C_P, P__C))  # $$$
        T__C_P = np.vstack((T__C_P, np.array([0, 0, 0, 1])))
        return T__C_P, R__C_P, P__C

    def invert_pose(self, T):
        """
        Compute the inverse of the matrix T [4x4] in SE(3).

        Args :
            T: np.ndarray, dim: (4,4)
               pose matrix in SE(3).
        Returns :
           T_inv: np.ndarray, dim: (4,4)
                  inverse of the matrix T in SE(3).
        """
        R = T[:3, :3]
        t = np.expand_dims(T[:3, -1], axis=1)
        T_inv = np.hstack((R.T, -R.T @ t))
        T_inv = np.vstack((T_inv, np.array([0, 0, 0, 1])))
        return T_inv

    def getXYAnglesFrom_R__W_M(self, R, deg=False):
        """
        Compute the angles (Euler YXZ) that describe the orientation of the given rotation matrix R__W_M.

        Args :
            R: np.ndarray, dim: (3,3)
               rotation matrix that describe the orientation of the maze {m} wrt the world frame {w}.
        Returns :
            alpha: float
                    angle around +X axis
            beta: float
                    angle around +Y axis
        """
        beta = np.arctan(R[0, 2] / R[2, 2])  # around +y
        alpha = np.arcsin(-R[1, 2])  # around +x
        if deg:
            alpha = alpha * 180 / np.pi
            beta = beta * 180 / np.pi
        return alpha, beta

    def undistort_points(self, img_points_raw: np.ndarray):  # (x,y)
        """
        Remove lens distortion, staying in the same K so solvePnP is dist-free.

        Args :
            img_points_raw:    np.ndarray, dim: (N,2)
                               image coordinates of the raw points in (x,y) = (line, column) convention.
        Returns :
            img_point_undist:  np.ndarray, dim: (N,2)
                               image coordinates of the undistorted points in (x,y) = (line, column) convention.

        """
        points = np.asarray(img_points_raw, dtype=np.float64).reshape(-1, 2)
        if self.dist is None:
            # No measured distortion (no pinhole_calib.json): the raw pixels
            # already are the undistorted pixels and self.K carries the
            # principal point, so this is deliberately a no-op.
            return points.copy()
        undistorted = cv2.undistortPoints(
            np.flip(points, axis=1).reshape(-1, 1, 2), self.K, self.dist, P=self.K
        ).reshape(-1, 2)
        return np.flip(undistorted, axis=1)  # back to (row, column)
