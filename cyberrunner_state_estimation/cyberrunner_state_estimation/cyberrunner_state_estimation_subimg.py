#!usr/bin/env python3

import time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
from scipy.spatial.transform import Rotation
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
from cyberrunner_state_estimation.core.estimation_pipeline import EstimationPipeline
from cyberrunner_state_estimation.core.opencv_acceleration import (
    configure_opencv_acceleration,
)
from cyberrunner_interfaces.msg import StateEstimate, StateEstimateSub
from std_msgs.msg import Float32, String


class ImageSubscriber(Node):
    def __init__(self, skip=1):
        super().__init__("cyberrunner_state_estimation")
        self.declare_parameter("process_every_n", skip)
        self.declare_parameter("pipeline_fps", 55.0)
        self.declare_parameter("print_measurements", False)
        self.declare_parameter("show_image", False)
        self.declare_parameter("use_gpu", False)
        self.declare_parameter("gpu_backend", "auto")
        self.declare_parameter("gpu_device_id", 0)
        self.declare_parameter("require_gpu", False)
        self.declare_parameter("playable_width", 0.259)
        self.declare_parameter("playable_height", 0.229)
        # Tolerance ADDED beyond the nominal board edge. The marble legitimately
        # reaches the board edge (~half the playable size), and the board tilts
        # during play, which shifts the estimated position by a few mm. A small
        # negative/zero tolerance rejected valid edge marbles as "outside" and
        # published NaN, so the marble looked lost near the top/bottom. Expand
        # the accepted region instead of shrinking it.
        #
        # The tolerance is per-axis because the estimator's error is not
        # symmetric. Measured over 52k detected samples of
        # hardware_recordings/passive_20260724_1330/state.csv:
        #   x  p0.5 span 237.6 mm vs 242.0 mm reachable -> healthy, 0.05% of
        #      samples land beyond the board half-width.
        #   y  p0.5 reaches +120.5 mm where the marble centre can only reach
        #      +106.0 mm -> positions near the TOP of the board read ~14 mm too
        #      high, and 40% of samples exceed the reachable limit.
        # So y needs the loose tolerance to avoid rejecting real marbles at the
        # top edge, while x can stay tight and keep rejecting outliers. Once the
        # +y bias is calibrated out, y should come back down to match x.
        #
        # MITIGATION, not a fix: after the ROI was made to follow the board, the
        # detector keeps seeing the marble at tilt, but a marble running along the
        # TOP edge reports y = +0.126..+0.129 -- 12-15 mm above a board whose half
        # height is 0.1145, which is physically impossible. With the tolerance at
        # 15 mm the gate sat at 0.1295 and clipped those by 0.1-3.8 mm, turning
        # every top-edge pass into `lost_outside`. 20 mm clears them. The real
        # cure is calibrating the +y bias out; until then this trades a known
        # ~13 mm optimistic position for NaN, which is the better trade for
        # training since NaN ends the episode.
        self.declare_parameter("playable_edge_tolerance", 0.005)
        self.declare_parameter("playable_edge_tolerance_y", 0.020)
        self.declare_parameter("ai_mode", "off")
        self.declare_parameter("ai_model_path", "")
        self.declare_parameter("ai_confidence_threshold", 0.90)
        self.declare_parameter("ai_check_every_n_frames", 5)
        self.declare_parameter("ai_agreement_radius_px", 12.0)
        self.declare_parameter("ai_max_reacquire_jump_px", 25.0)
        self.declare_parameter("ai_occlusion_grace_frames", 90)
        self.declare_parameter("ai_far_reacquire_confirm_frames", 3)
        self.declare_parameter("ai_hole_rejection_enabled", True)
        self.declare_parameter("ai_hole_rejection_margin_m", 0.0025)
        self.declare_parameter("ai_hole_rejection_delay_sec", 2.0)
        self.declare_parameter("ai_roi_x_min", 0.25)
        self.declare_parameter("ai_roi_y_min", 0.15)
        self.declare_parameter("ai_roi_x_max", 0.72)
        self.declare_parameter("ai_roi_y_max", 0.80)
        self.declare_parameter("ai_max_prediction_std_m", 0.03)
        # Track the board instead of using ai_roi_* fixed in image space. The
        # ROI is inset inside the corner dots so the blue dots stay OUT of the
        # searchable region. Set ai_roi_follows_corners=False to go back to the
        # static rectangle.
        self.declare_parameter("ai_roi_follows_corners", True)
        self.declare_parameter("ai_roi_corner_inset_px", 0.0)
        # Radius (px) of the disc masked around each blue corner dot, so the AI
        # cannot lock onto a marker. Cannot be done with the ROI rectangle: an
        # edge-travelling marble shares image rows with the top/bottom dots.
        self.declare_parameter("ai_corner_mask_radius_px", 12.0)

        self.skip = max(1, int(self.get_parameter("process_every_n").value))
        self.pipeline_fps = float(self.get_parameter("pipeline_fps").value)
        self.print_measurements = bool(self.get_parameter("print_measurements").value)
        self.show_image = bool(self.get_parameter("show_image").value)
        self.use_gpu = bool(self.get_parameter("use_gpu").value)
        self.gpu_backend = str(self.get_parameter("gpu_backend").value)
        self.gpu_device_id = int(self.get_parameter("gpu_device_id").value)
        self.require_gpu = bool(self.get_parameter("require_gpu").value)
        self.ai_mode = str(self.get_parameter("ai_mode").value).lower()
        self.ai_model_path = str(self.get_parameter("ai_model_path").value)
        self.ai_max_prediction_std_m = float(
            self.get_parameter("ai_max_prediction_std_m").value
        )
        playable_width = float(self.get_parameter("playable_width").value)
        playable_height = float(self.get_parameter("playable_height").value)
        edge_tolerance = max(
            0.0, float(self.get_parameter("playable_edge_tolerance").value)
        )
        edge_tolerance_y = max(
            0.0, float(self.get_parameter("playable_edge_tolerance_y").value)
        )
        self.playable_half_x = playable_width / 2.0 + edge_tolerance
        self.playable_half_y = playable_height / 2.0 + edge_tolerance_y

        self.acceleration_backend, acceleration_msg = configure_opencv_acceleration(
            self.use_gpu,
            self.gpu_backend,
            self.gpu_device_id,
            self.require_gpu,
        )
        self.get_logger().info(acceleration_msg)

        self.subscription = self.create_subscription(
            Image, "cyberrunner_camera/image", self.listener_callback, 1
        )
        self.publisher_ = self.create_publisher(
            StateEstimateSub, "cyberrunner_state_estimation/estimate_subimg", 1
        )
        self.state_publisher_ = self.create_publisher(
            StateEstimate, "cyberrunner_state_estimation/estimate", 1
        )
        self.ball_source_publisher = self.create_publisher(
            String, "cyberrunner_state_estimation/ball_source", 10
        )
        self.ai_confidence_publisher = self.create_publisher(
            Float32, "cyberrunner_state_estimation/ai_confidence", 10
        )
        self.detection_disagreement_publisher = self.create_publisher(
            Float32, "cyberrunner_state_estimation/detection_disagreement_px", 10
        )
        self.tf_static_broadcaster = StaticTransformBroadcaster(self)
        self.tf_broadcaster = TransformBroadcaster(self)

        self.get_logger().info("Image subscriber has been initialized.")
        ai_config = {
            "ai_mode": self.ai_mode,
            "ai_model_path": self.ai_model_path,
            "ai_confidence_threshold": float(
                self.get_parameter("ai_confidence_threshold").value
            ),
            "ai_valid_roi": (
                float(self.get_parameter("ai_roi_x_min").value),
                float(self.get_parameter("ai_roi_y_min").value),
                float(self.get_parameter("ai_roi_x_max").value),
                float(self.get_parameter("ai_roi_y_max").value),
            ),
            "ai_check_every_n_frames": int(
                self.get_parameter("ai_check_every_n_frames").value
            ),
            "ai_agreement_radius_px": float(
                self.get_parameter("ai_agreement_radius_px").value
            ),
            "ai_max_reacquire_jump_px": float(
                self.get_parameter("ai_max_reacquire_jump_px").value
            ),
            "ai_occlusion_grace_frames": int(
                self.get_parameter("ai_occlusion_grace_frames").value
            ),
            "ai_far_reacquire_confirm_frames": int(
                self.get_parameter("ai_far_reacquire_confirm_frames").value
            ),
            "ai_hole_rejection_enabled": bool(
                self.get_parameter("ai_hole_rejection_enabled").value
            ),
            "ai_hole_rejection_margin_m": float(
                self.get_parameter("ai_hole_rejection_margin_m").value
            ),
            "ai_hole_rejection_delay_sec": float(
                self.get_parameter("ai_hole_rejection_delay_sec").value
            ),
            "ai_roi_follows_corners": bool(
                self.get_parameter("ai_roi_follows_corners").value
            ),
            "ai_roi_corner_inset_px": float(
                self.get_parameter("ai_roi_corner_inset_px").value
            ),
            "ai_corner_mask_radius_px": float(
                self.get_parameter("ai_corner_mask_radius_px").value
            ),
        }
        self.br = CvBridge()
        self.estimation_pipeline = EstimationPipeline(
            fps=self.pipeline_fps,
            estimator="KF",  #  "FiniteDiff",  "KF", "KFBias"
            print_measurements=self.print_measurements,
            show_image=self.show_image,
            do_anim_3d=False,
            viewpoint="top",  # 'top', 'side', 'topandside'
            show_subimages_detector=False,
            acceleration_backend=self.acceleration_backend,
            ai_config=ai_config,
        )
        self.get_logger().info(
            f"AI marble mode={self.ai_mode}; model={self.ai_model_path or 'none'}"
        )
        self.last_valid_ball_pixel = None
        self.outside_candidate_active = False
        self.outside_candidate_count = 0
        self.outside_warning_frames = 5

        self.count = 0
        # self.prev_a = self.prev_b = 0.0
        self.a = np.zeros(15, dtype=float)
        self.b = np.zeros(15, dtype=float)

        # --- lightweight frame-rate profiler ---------------------------------
        # Separates the two possible bottlenecks: camera/USB delivery rate
        # (inter-arrival between callbacks) vs. estimate() compute cost.
        self.declare_parameter("profile_timing", True)
        self.profile_timing = bool(self.get_parameter("profile_timing").value)
        self.profile_window = 60          # frames per printed summary
        self._prof_last_recv = None
        self._prof_arrival = []           # s between consecutive frames
        self._prof_compute = []           # s spent inside estimate()

    def _profile_report(self):
        import statistics
        arr = self._prof_arrival
        cmp = self._prof_compute
        if not arr or not cmp:
            return
        arr_mean = statistics.mean(arr)
        cmp_mean = statistics.mean(cmp)
        cam_fps = 1.0 / arr_mean if arr_mean > 0 else 0.0
        cmp_fps = 1.0 / cmp_mean if cmp_mean > 0 else 0.0
        print(
            "[PROFILE] "
            f"camera_arrival: {arr_mean*1000:.1f} ms avg / {max(arr)*1000:.1f} ms max "
            f"(~{cam_fps:.1f} fps)  |  "
            f"estimate(): {cmp_mean*1000:.1f} ms avg / {max(cmp)*1000:.1f} ms max "
            f"(~{cmp_fps:.1f} fps cap)  ->  "
            f"bottleneck={'COMPUTE' if cmp_mean >= arr_mean * 0.9 else 'CAMERA/USB'}"
        )
        self._prof_arrival.clear()
        self._prof_compute.clear()

    def listener_callback(self, data):
        # self.get_logger().info('Receiving image frame')
        t_recv = time.perf_counter()
        if self.profile_timing:
            if self._prof_last_recv is not None:
                self._prof_arrival.append(t_recv - self._prof_last_recv)
            self._prof_last_recv = t_recv

        frame = self.br.imgmsg_to_cv2(data)

        # cv2.imshow("before", frame)
        b, g, r = np.mean(np.mean(frame, axis=0), axis=0)
        # print(b,g,r)
        if g > 100 and b < 40 and r < 40:
            print("SKIP THIS FRAME")
            # cv2.waitKey(1)
            return
        # cv2.imshow("before", frame)
        t_est0 = time.perf_counter()
        x_hat, P, angles, subimg, xb, yb = self.estimation_pipeline.estimate(
            frame, return_ball_subimg=True
        )
        detector = self.estimation_pipeline.measurements.detector
        ball_source = detector.last_ball_source
        if ball_source == "kalman_occlusion":
            prediction_std = float(np.sqrt(np.max(np.diag(P)[:2])))
            if (
                np.all(np.isfinite(x_hat[:2]))
                and np.isfinite(prediction_std)
                and prediction_std <= self.ai_max_prediction_std_m
            ):
                xb, yb = map(float, x_hat[:2])
            else:
                ball_source = "lost_uncertain"
        if self.profile_timing:
            self._prof_compute.append(time.perf_counter() - t_est0)
            if len(self._prof_compute) >= self.profile_window:
                self._profile_report()
        if np.isfinite(xb) and np.isfinite(yb) and (
            abs(float(xb)) > self.playable_half_x
            or abs(float(yb)) > self.playable_half_y
        ):
            if self.last_valid_ball_pixel is not None:
                detector.restore_ball_tracking(self.last_valid_ball_pixel)
            else:
                detector.reset_ball_tracking()
            self.outside_candidate_count += 1
            if (
                not self.outside_candidate_active
                and self.outside_candidate_count >= self.outside_warning_frames
            ):
                self.get_logger().warn(
                    "Repeated marble candidates outside playable map; "
                    "retaining last valid tracking crop "
                    f"(x={float(xb):.4f}, y={float(yb):.4f})"
                )
                self.outside_candidate_active = True
            xb = np.nan
            yb = np.nan
            ball_source = "lost_outside"
        elif np.isfinite(xb) and np.isfinite(yb):
            if detector.ball_pos is not None:
                self.last_valid_ball_pixel = detector.ball_pos.copy()
            if self.outside_candidate_active:
                self.get_logger().info("Marble tracking recovered inside playable map.")
            self.outside_candidate_active = False
            self.outside_candidate_count = 0
        if self.count % self.skip == 0:
            msg = StateEstimateSub()
            msg.state.x_b = xb
            msg.state.y_b = yb
            msg.state.x_b_dot = x_hat[2]
            msg.state.y_b_dot = x_hat[3]
            msg.state.alpha = -angles[1]
            msg.state.beta = angles[0]
            msg.subimg = self.br.cv2_to_imgmsg(subimg)
            self.publisher_.publish(msg)

            state_msg = StateEstimate()
            state_msg.x_b = msg.state.x_b
            state_msg.y_b = msg.state.y_b
            state_msg.x_b_dot = msg.state.x_b_dot
            state_msg.y_b_dot = msg.state.y_b_dot
            state_msg.alpha = msg.state.alpha
            state_msg.beta = msg.state.beta
            self.state_publisher_.publish(state_msg)
            source_msg = String()
            source_msg.data = ball_source
            self.ball_source_publisher.publish(source_msg)
            confidence_msg = Float32()
            confidence_msg.data = float(detector.last_ai_confidence)
            self.ai_confidence_publisher.publish(confidence_msg)
            disagreement_msg = Float32()
            disagreement_msg.data = float(
                detector.last_detection_disagreement_px
            )
            self.detection_disagreement_publisher.publish(disagreement_msg)
            # self.get_logger().info(f"Publishing: {x_hat}")

        # Broadcast transforms
        if self.count == 0:
            t = self.get_tf_msg(
                self.estimation_pipeline.measurements.plate_pose.T__W_C,
                'camera',
                'world',
            )
            self.tf_static_broadcaster.sendTransform(t)
        t_maze = self.get_tf_msg(
            self.estimation_pipeline.measurements.plate_pose.T__W_M,
            'maze',
            'world',
        )
        T__B_M = np.eye(4)
        if np.isfinite(xb) and np.isfinite(yb):
            T__B_M[:3, -1] = [
                xb,
                yb,
                self.estimation_pipeline.measurements.plate_pose.R_BALL,
            ]
        else:
            T__B_M[:3, -1] = (
                self.estimation_pipeline.measurements.get_ball_position_in_maze()
            )
        t_ball = self.get_tf_msg(
            T__B_M,
            'maze',
            'ball'
        )
        self.tf_broadcaster.sendTransform([t_maze, t_ball])

        # self.a[:-1] = self.a[1:]
        # self.a[-1] = msg.state.alpha
        # self.b[:-1] = self.b[1:]
        # self.b[-1] = msg.state.beta
        # print("a_dot: {:.4f}, b_dot: {:.4f}".format((self.a[-1] - self.a[0]) * 55.0 / 14.0, (self.b[-1] - self.b[0]) * 55.0 / 14.0))
        # #self.prev_a = msg.state.alpha
        # self.prev_b = msg.state.beta
        # cv2.imshow("sub", subimg)
        # cv2.waitKey(1)
        self.count += 1

    def get_tf_msg(self, se3, frame_id, child_frame_id):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = frame_id
        t.child_frame_id = child_frame_id
        t.transform.translation.x = se3[0, 3]
        t.transform.translation.y = se3[1, 3]
        t.transform.translation.z = se3[2, 3]
        q = Rotation.from_matrix(se3[:3, :3]).as_quat()
        t.transform.rotation.x = q[0]
        t.transform.rotation.y = q[1]
        t.transform.rotation.z = q[2]
        t.transform.rotation.w = q[3]
        return t


def main(args=None):
    rclpy.init(args=args)
    image_subscriber = ImageSubscriber()
    rclpy.spin(image_subscriber)
    rclpy.shutdown()
