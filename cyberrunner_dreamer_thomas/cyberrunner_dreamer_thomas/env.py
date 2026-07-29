import sys

from cyberrunner_dreamer_thomas import cyberrunner_layout_custom
from cyberrunner_dreamer_thomas.path import LinearPath

from cyberrunner_interfaces.msg import DynamixelVel, StateEstimateSub
from cyberrunner_interfaces.srv import DynamixelReset

import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
import gym
import numpy as np
import time
import cv2
import os
from ament_index_python.packages import get_package_share_directory


class CyberrunnerGym(gym.Env):
    def __init__(
        self,
        repeat=1,
        layout=cyberrunner_layout_custom.cyberrunner_dxf_layout,
        num_rel_path=5,
        num_wait_steps=30,
        reward_on_fail=0.0,
        reward_on_goal=0.5,
    ):
        super().__init__()
        if not rclpy.ok():
            rclpy.init()

        self.future = None

        self.cheat = False
        self.observation_space = gym.spaces.Dict(
            image=gym.spaces.Box(0, 255, (64, 64, 1), np.uint8),
            states=gym.spaces.Box(-np.inf, np.inf, (4,), np.float32),
            goal=gym.spaces.Box(-np.inf, np.inf, (num_rel_path * 2,), np.float32),
            progress=gym.spaces.Box(-np.inf, np.inf, (1,), np.float32),
            log_reward=gym.spaces.Box(-np.inf, np.inf, (1,), np.float32),
            log_elapsed_sec=gym.spaces.Box(0.0, np.inf, (1,), np.float32),
            log_success=gym.spaces.Box(0.0, 1.0, (1,), np.float32),
        )
        self.action_space = gym.spaces.Box(-1.0, 1.0, (2,))
        self.obs = dict(self.observation_space.sample())
        self.num_rel_path = num_rel_path
        # self.norm_min = np.array(
        #     [0.0, 0.0, -10 * np.pi / 180.0, -10 * np.pi / 180.0] +
        #     [-0.01 * (k + 1) for k in range(self.num_rel_path) for _ in
        #      range(2)])
        board_size = np.array(
            [layout["board_width"], layout["board_height"]], dtype=np.float32
        )
        self.norm_max = np.array(
            [10 * np.pi / 180.0, 10 * np.pi / 180.0, *board_size]
        )
        self.goal_norm_max = np.array(
            [0.0002 * 60 * k for k in range(1, self.num_rel_path + 1) for _ in range(2)]
        )

        self.node = Node("cyberrunner_gym")
        self.publisher = self.node.create_publisher(
            DynamixelVel,
            "cyberrunner_dynamixel/cmd",
            1,
        )
        self.subscription = self.node.create_subscription(
            StateEstimateSub,
            "cyberrunner_state_estimation/estimate_subimg",
            self._msg_to_obs,
            1,
        )
        self.client = self.node.create_client(
            DynamixelReset, "cyberrunner_dynamixel/reset"
        )

        self.br = CvBridge()

        self.repeat = repeat

        self.offset = board_size / 2.0
        self.holes = np.asarray(layout["holes"], dtype=np.float32)
        self.danger_zones = layout.get("danger_zones", [])
        self.danger_lines = layout.get("danger_lines", [])
        self.segment_guards = layout.get("segment_guards", [])
        self.hole_detect_radius = float(
            os.environ.get("CYBERRUNNER_HOLE_DETECT_RADIUS_M", "0.005")
        )
        self.hole_detect_frames = max(
            1, int(os.environ.get("CYBERRUNNER_HOLE_DETECT_FRAMES", "3"))
        )
        self.hole_detect_count = 0
        self.ball_in_hole = False
        self.ball_hole_index = -1
        self.ball_hole_distance = np.inf
        shared = get_package_share_directory("cyberrunner_dreamer_thomas")
        self.p = LinearPath.load(os.path.join(shared, "path_custom.pkl"))
        # if not self.cheat:
        #     self.p = LinearPath.load("path_0002_hard.pkl")
        # else:
        #     self.p = LinearPath(np.array(layout["waypoints"]))
        # self.p = LinearPath(
        #     np.array(layout["waypoints"]),
        #      walls_h=np.array(layout["walls_h"]),
        #      walls_v=np.array(layout["walls_v"]),
        #      holes=np.empty((0, 3)),
        #  )
        # self.p.save("/home/thomas/path_0002_hard.pkl")
        cheat_threshold_m = float(os.environ.get("CYBERRUNNER_CHEAT_THRESHOLD_M", "0.10"))
        self.cheat_threshold = int(cheat_threshold_m / self.p.distance)
        self.cheat_threshold_target_dt = 1.0 / float(
            os.environ.get("CYBERRUNNER_CHEAT_THRESHOLD_TARGET_FPS", "35.0")
        )
        self.cheat_threshold_max_scale = float(
            os.environ.get("CYBERRUNNER_CHEAT_THRESHOLD_MAX_SCALE", "2.0")
        )
        self.cheat_threshold_scale = 1.0
        self.progress_spike_count = 0
        self.max_progress_spikes = int(os.environ.get("CYBERRUNNER_MAX_PROGRESS_SPIKES", "8"))
        self.local_path_back = int(
            float(os.environ.get("CYBERRUNNER_LOCAL_PATH_BACK_M", "0.04"))
            / self.p.distance
        )
        self.local_path_forward = int(
            float(os.environ.get("CYBERRUNNER_LOCAL_PATH_FORWARD_M", "0.16"))
            / self.p.distance
        )
        self.enforce_checkpoint_order = (
            os.environ.get("CYBERRUNNER_ENFORCE_CHECKPOINT_ORDER", "true").lower()
            not in ("0", "false", "no")
        )
        self.checkpoint_radius = float(os.environ.get("CYBERRUNNER_CHECKPOINT_RADIUS_M", "0.010"))
        self.checkpoint_skip_tolerance = int(
            float(os.environ.get("CYBERRUNNER_CHECKPOINT_SKIP_TOL_M", "0.02"))
            / self.p.distance
        )
        self.enforce_path_corridor = (
            os.environ.get("CYBERRUNNER_ENFORCE_PATH_CORRIDOR", "true").lower()
            not in ("0", "false", "no")
        )
        self.path_corridor_radius = float(
            os.environ.get("CYBERRUNNER_PATH_CORRIDOR_RADIUS_M", "0.018")
        )
        self.waypoint_path_indices = self._make_waypoint_path_indices()
        print(
            f"[ROS ENV] cheat_threshold={self.cheat_threshold} "
            f"path points ({cheat_threshold_m:.3f} m), "
            f"dt_scale_max={self.cheat_threshold_max_scale:.2f}, "
            f"max_progress_spikes={self.max_progress_spikes}, "
            f"local_path_window=[-{self.local_path_back}, +{self.local_path_forward}], "
            f"checkpoint_order={self.enforce_checkpoint_order}, "
            f"checkpoint_radius={self.checkpoint_radius:.3f} m, "
            f"path_corridor={self.enforce_path_corridor}, "
            f"path_corridor_radius={self.path_corridor_radius:.3f} m, "
            f"hole_detect_radius={self.hole_detect_radius:.3f} m, "
            f"hole_detect_frames={self.hole_detect_frames}"
        )
        self.prev_pos_path = 0
        self.next_checkpoint = 1
        self.last_step_next_checkpoint = 1
        self.checkpoint_skip = False
        self.path_shortcut = False
        self.path_shortcut_distance = 0.0
        self.num_wait_steps = num_wait_steps
        self.reward_on_fail = float(os.environ.get("CYBERRUNNER_REWARD_ON_FAIL", "-0.10"))
        self.reward_on_hole = float(os.environ.get("CYBERRUNNER_REWARD_ON_HOLE", "-0.20"))
        self.reward_on_shortcut = float(
            os.environ.get("CYBERRUNNER_REWARD_ON_SHORTCUT", "-0.20")
        )
        self.reward_on_danger = float(
            os.environ.get("CYBERRUNNER_REWARD_ON_DANGER", "-0.40")
        )
        self.reward_on_checkpoint = float(
            os.environ.get("CYBERRUNNER_REWARD_ON_CHECKPOINT", "0.03")
        )
        self.reward_on_goal = float(os.environ.get("CYBERRUNNER_REWARD_ON_GOAL", str(reward_on_goal)))
        print(
            f"[ROS ENV] rewards: fail={self.reward_on_fail:.3f}, "
            f"hole={self.reward_on_hole:.3f}, "
            f"shortcut={self.reward_on_shortcut:.3f}, "
            f"danger={self.reward_on_danger:.3f}, "
            f"checkpoint={self.reward_on_checkpoint:.3f}, "
            f"goal={self.reward_on_goal:.3f}"
        )
        print(
            f"[ROS ENV] segment_guards={len(self.segment_guards)}, "
            f"danger_zones={len(self.danger_zones)}, "
            f"danger_lines={len(self.danger_lines)}"
        )

        self.ball_detected = False

        # Dynamixel
        self.max_angle_vel = float(os.environ.get("CYBERRUNNER_MAX_ANGLE_VEL", "180"))
        self.alpha_fac = -1.0  # -6.0 / 0.051
        self.beta_fac = -1.0  # -12 / 0.088

        self.last_time = 0
        self.episode_start_time = time.monotonic()
        self.progress = 0
        self.accum_reward = 0.0
        self.steps = 0
        self.off_path = False

        self.episodes = 0
        self.in_danger_zone = False
        self.danger_zone_name = ""
        self.danger_zone_distance = np.inf
        self.last_ball_pos = None
        self.segment_guard_hit = False
        self.segment_guard_name = ""

        self.new_obs = False

    def step(self, action):

        self.steps += 1

        # Send action to dynamixel
        if self.cheat and (self.p.num_points - self.prev_pos_path <= 200):
            action[1] = 1.0
        self._send_action(action)

        # Get observation
        obs = self._get_obs()

        now = time.time()
        step_dt = now - self.last_time
        self.last_time = now
        if step_dt > 0.0:
            self.cheat_threshold_scale = min(
                max(step_dt / self.cheat_threshold_target_dt, 1.0),
                self.cheat_threshold_max_scale,
            )
        else:
            self.cheat_threshold_scale = 1.0

        # Compute reward
        reward = self._get_reward(obs)

        # Get done
        done = self._get_done(obs)
        if done and (not self.success):
            if self.ball_in_hole:
                reward = self.reward_on_hole
            elif self.in_danger_zone or self.segment_guard_hit:
                reward = self.reward_on_danger
            elif self.checkpoint_skip or self.path_shortcut:
                reward = self.reward_on_shortcut
            else:
                reward = self.reward_on_fail
        if self.success:
            reward += self.reward_on_goal

        if done or self.steps == 3000:
            # action = np.zeros((2,))
            # self._send_action(action)
            print("Reset board")
            self._reset_board()

        if self.success:
            info = {"is_terminal": False}
        else:
            info = {}

        if step_dt > (1.0 / 35.0):
            step_fps = 1.0 / step_dt if step_dt > 0.0 else 0.0
            print(f"Slower than 35fps: step_dt={step_dt * 1000.0:.1f} ms, fps={step_fps:.1f}")
        # print(self.prev_pos_path)
        self.accum_reward += reward if not done else 0
        elapsed_sec = time.monotonic() - self.episode_start_time
        obs["states"] = (obs["states"] / self.norm_max).astype(np.float32)
        obs["goal"] = (obs["goal"] / self.goal_norm_max).astype(np.float32)
        obs["progress"] = np.asarray([1 + self.prev_pos_path], dtype=np.float32)
        obs["log_reward"] = np.asarray([reward if not done else 0], dtype=np.float32)
        obs["log_elapsed_sec"] = np.asarray([elapsed_sec], dtype=np.float32)
        obs["log_success"] = np.asarray([float(self.success)], dtype=np.float32)
        return obs, reward, done, info

    def reset(self):
        print("Resetting ...")
        self.episodes += 1
        print("Previous reward: {}".format(self.accum_reward))
        print("Previous episode length: {}".format(self.steps / 60.0))
        print("Episodes: {}".format(self.episodes))
        self.accum_reward = 0.0
        self.steps = 0
        self.success = False
        self.ball_detected = False
        self.hole_detect_count = 0
        self.ball_in_hole = False
        self.ball_hole_index = -1
        self.ball_hole_distance = np.inf
        self.checkpoint_skip = False
        self.path_shortcut = False
        self.path_shortcut_distance = 0.0
        self.last_step_next_checkpoint = 1
        self.in_danger_zone = False
        self.danger_zone_name = ""
        self.danger_zone_distance = np.inf
        self.last_ball_pos = None
        self.segment_guard_hit = False
        self.segment_guard_name = ""
        self.episode_start_time = time.monotonic()

        # Wait for board reset to be done
        if self.future is not None:
            rclpy.spin_until_future_complete(self.node, self.future, timeout_sec=10)

        # Set velocities to 0  TODO: use action/service to set board to 0 state
        action = np.zeros((2,))
        self._send_action(action)

        kb_reset = False
        if kb_reset:
            # Wait for keyboard press
            cv2.imshow("reset", np.zeros((200, 200, 3)))
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            print("Done resetting ...")

            # Get observation
            for _ in range(2):
                obs = self._get_obs()
        else:
            count = 0
            obs = self._get_obs()
            while count < self.num_wait_steps:
                obs = self._get_obs()
                count = count + 1 if self.ball_detected else 0

        self.prev_pos_path = self.p.closest_point(obs["states"][2:4])[0]
        self.last_ball_pos = obs["states"][2:4].copy()
        self.next_checkpoint = self._next_checkpoint_from_path(self.prev_pos_path)
        self.last_step_next_checkpoint = self.next_checkpoint
        self.progress = 0
        self.progress_spike_count = 0
        self.last_time = time.time()
        obs["states"] = (obs["states"] / self.norm_max).astype(np.float32)
        obs["goal"] = (obs["goal"] / self.goal_norm_max).astype(np.float32)
        obs["progress"] = np.asarray([1 + self.prev_pos_path], dtype=np.float32)
        obs["log_reward"] = np.asarray([0], dtype=np.float32)
        obs["log_elapsed_sec"] = np.asarray([0.0], dtype=np.float32)
        obs["log_success"] = np.asarray([0.0], dtype=np.float32)
        return obs

    def render(self, mode="human"):
        pass

    def _send_action(self, action):
        # Scale action
        action = action.copy()
        action *= self.max_angle_vel
        vel_1 = self.alpha_fac * action[0]  # TODO define these as parameters
        vel_2 = self.beta_fac * action[1]

        # To message and publish
        msg = DynamixelVel()
        msg.vel_1 = vel_1
        msg.vel_2 = vel_2
        self.publisher.publish(msg)

    def _get_obs(self):
        # Spin repeat times to get the next observation
        while not self.new_obs:
            for _ in range(self.repeat):
                rclpy.spin_once(self.node)
        self.new_obs = False

        return self.obs.copy()

    def _get_reward(self, obs):
        # If no ball is detected, reward is 0
        if not self.ball_detected:
            reward = 0.0
        else:
            curr_pos_path, p = self._closest_point(obs["states"][2:4])
            self.off_path = curr_pos_path == -1
            self.path_shortcut = False
            self.path_shortcut_distance = 0.0
            if curr_pos_path != -1:
                self.path_shortcut_distance = float(np.linalg.norm(obs["states"][2:4] - p))
                self.path_shortcut = (
                    self.enforce_path_corridor
                    and (not self.cheat)
                    and self.path_shortcut_distance > self.path_corridor_radius
                )
                if self.path_shortcut:
                    print(
                        "[Shortcut]: outside path corridor "
                        f"dist={self.path_shortcut_distance:.4f} "
                        f"threshold={self.path_corridor_radius:.4f} "
                        f"path={curr_pos_path}"
                    )
            if self.off_path and self.cheat:
                curr_pos_path = self.prev_pos_path

            raw_progress = curr_pos_path - self.prev_pos_path
            if self.path_shortcut:
                self.progress = 0
                return 0.0

            scaled_cheat_threshold = int(self.cheat_threshold * self.cheat_threshold_scale)
            if (not self.cheat) and raw_progress > scaled_cheat_threshold:
                self.progress = 0
                self.progress_spike_count += 1
                reward = 0.0
                print(
                    "[Spike]: Ignoring too high progress "
                    f"(progress={raw_progress}, threshold={scaled_cheat_threshold}, "
                    f"scale={self.cheat_threshold_scale:.2f}, "
                    f"count={self.progress_spike_count}/{self.max_progress_spikes})"
                )
                return reward

            checkpoint_before = self.next_checkpoint
            if self._update_checkpoint_order(obs["states"][2:4], curr_pos_path):
                self.progress = 0
                reward = 0.0
                return reward

            self.progress_spike_count = 0
            self.progress = raw_progress
            checkpoints_passed = self.next_checkpoint - checkpoint_before
            reward = (
                float(raw_progress) * 0.004 / 16.0
                + checkpoints_passed * self.reward_on_checkpoint
            )
            # reward = np.clip(reward, -5.0, 5.0)
            self.prev_pos_path = curr_pos_path
            # self.curr_dist = np.linalg.norm(p - obs["states"][:2])
        return reward

    def _get_done(self, obs):
        # Done if ball is not detected
        done = not self.ball_detected
        if done:
            print("[Done]: BALL LOST")

        if self.ball_detected:
            self._update_ball_in_hole(obs["states"][2:4])
            self._update_danger_zone_state(obs["states"][2:4])
            self._update_segment_guard_state(obs["states"][2:4])
            if self.ball_in_hole:
                done = True
                print(
                    "[Done]: BALL IN HOLE "
                    f"index={self.ball_hole_index} "
                    f"dist={self.ball_hole_distance:.4f}"
                )
            elif self.in_danger_zone:
                done = True
                print(
                    "[Done]: DANGER ZONE "
                    f"name={self.danger_zone_name} "
                    f"dist={self.danger_zone_distance:.4f}"
                )
            elif self.segment_guard_hit:
                done = True
                print(f"[Done]: SEGMENT GUARD name={self.segment_guard_name}")

        # Done if off path and not cheating
        if (not self.cheat) and self.off_path:
            done = True
            print("[Done]: OFFPATH")

        if (not self.cheat) and self.path_shortcut:
            done = True
            print(
                "[Done]: PATH SHORTCUT "
                f"dist={self.path_shortcut_distance:.4f} "
                f"threshold={self.path_corridor_radius:.4f}"
            )

        # Done if reached goal
        if self.p.num_points - self.prev_pos_path <= 1:
            self.success = True
            done = True
            print("[Done]: SUCCESS")
            # self._send_action(np.array([0.0, 0.75]))
            # time.sleep(0.05)
            self._send_action(np.array([0.0, 0.0]))

        if (not self.cheat) and self.progress_spike_count >= self.max_progress_spikes:
            done = True
            print(
                "[Done]: Repeated too high progress "
                f"(count={self.progress_spike_count}/{self.max_progress_spikes})"
            )

        if (not self.cheat) and self.checkpoint_skip:
            done = True
            print(f"[Done]: CHECKPOINT SKIP next={self.next_checkpoint}")

        # if self.curr_dist > 0.02:
        #     done = True

        return done

    def _update_ball_in_hole(self, ball_pos):
        distances = np.linalg.norm(self.holes - ball_pos, axis=1)
        hole_index = int(np.argmin(distances))
        hole_distance = float(distances[hole_index])

        self.ball_hole_index = hole_index
        self.ball_hole_distance = hole_distance
        if hole_distance <= self.hole_detect_radius:
            self.hole_detect_count += 1
        else:
            self.hole_detect_count = 0

        self.ball_in_hole = self.hole_detect_count >= self.hole_detect_frames

    def _checkpoint_in_range(self, checkpoint, start_checkpoint, end_checkpoint):
        return start_checkpoint <= checkpoint <= end_checkpoint

    def _segment_distance_to_point(self, start, end, point):
        segment = end - start
        denom = float(np.dot(segment, segment))
        if denom <= 1e-12:
            return float(np.linalg.norm(point - end))
        t = float(np.dot(point - start, segment) / denom)
        t = min(1.0, max(0.0, t))
        closest = start + t * segment
        return float(np.linalg.norm(point - closest))

    def _cross2d(self, a, b):
        return float(a[0] * b[1] - a[1] * b[0])

    def _segments_intersect(self, a, b, c, d):
        ab = b - a
        cd = d - c
        o1 = self._cross2d(ab, c - a)
        o2 = self._cross2d(ab, d - a)
        o3 = self._cross2d(cd, a - c)
        o4 = self._cross2d(cd, b - c)
        return (o1 * o2 < 0.0) and (o3 * o4 < 0.0)

    def _point_line_side(self, point, line_start, line_end):
        return self._cross2d(line_end - line_start, point - line_start)

    def _danger_line_crossed(self, previous, current, line_start, line_end, width):
        if self._segments_intersect(previous, current, line_start, line_end):
            return True, 0.0
        distance = self._segment_distance_to_segment(previous, current, line_start, line_end)
        return distance <= width, distance

    def _segment_distance_to_segment(self, a, b, c, d):
        if self._segments_intersect(a, b, c, d):
            return 0.0
        return min(
            self._segment_distance_to_point(a, b, c),
            self._segment_distance_to_point(a, b, d),
            self._segment_distance_to_point(c, d, a),
            self._segment_distance_to_point(c, d, b),
        )

    def _update_danger_zone_state(self, ball_pos):
        self.in_danger_zone = False
        self.danger_zone_name = ""
        self.danger_zone_distance = np.inf
        current = np.asarray(ball_pos, dtype=np.float32)
        previous = None if self.last_ball_pos is None else np.asarray(self.last_ball_pos, dtype=np.float32)
        if previous is not None:
            for line in self.danger_lines:
                if not self._checkpoint_in_range(
                    self.next_checkpoint,
                    int(line["start_checkpoint"]),
                    int(line["end_checkpoint"]),
                ):
                    continue
                p1 = np.asarray(line["p1"], dtype=np.float32)
                p2 = np.asarray(line["p2"], dtype=np.float32)
                width = float(line.get("width", 0.0))
                crossed, distance = self._danger_line_crossed(previous, current, p1, p2, width)
                if crossed:
                    self.in_danger_zone = True
                    self.danger_zone_name = line.get("name", "danger_line")
                    self.danger_zone_distance = distance
                    print(
                        "[DangerLine]: crossed "
                        f"name={self.danger_zone_name} "
                        f"dist={distance:.4f} width={width:.4f}"
                    )
                    self.last_ball_pos = current
                    return
        for zone in self.danger_zones:
            if not self._checkpoint_in_range(
                self.next_checkpoint,
                int(zone["start_checkpoint"]),
                int(zone["end_checkpoint"]),
            ):
                continue
            center = np.asarray(zone["center"], dtype=np.float32)
            radius = float(zone["radius"])
            point_distance = float(np.linalg.norm(current - center))
            previous_distance = None if previous is None else float(np.linalg.norm(previous - center))
            distance = point_distance
            crossed_through = False
            if previous is not None:
                distance = self._segment_distance_to_point(previous, current, center)
                crossed_through = (
                    previous_distance is not None
                    and previous_distance > radius
                    and point_distance > radius
                    and distance <= radius
                )
            if crossed_through:
                self.in_danger_zone = True
                self.danger_zone_name = zone.get("name", "danger_zone")
                self.danger_zone_distance = distance
                self.last_ball_pos = current
                return
        self.last_ball_pos = current

    def _update_segment_guard_state(self, ball_pos):
        self.segment_guard_hit = False
        self.segment_guard_name = ""
        x, y = float(ball_pos[0]), float(ball_pos[1])
        for guard in self.segment_guards:
            if not self._checkpoint_in_range(
                self.next_checkpoint,
                int(guard["start_checkpoint"]),
                int(guard["end_checkpoint"]),
            ):
                continue
            y_min = float(guard.get("y_min", -np.inf))
            y_max = float(guard.get("y_max", np.inf))
            if not (y_min <= y <= y_max):
                continue
            x_min = guard.get("x_min")
            x_max = guard.get("x_max")
            if x_min is not None and x < float(x_min):
                self.segment_guard_hit = True
            if x_max is not None and x > float(x_max):
                self.segment_guard_hit = True
            if self.segment_guard_hit:
                self.segment_guard_name = guard.get("name", "segment_guard")
                return

    def _make_waypoint_path_indices(self):
        indices = [0]
        total = 0
        for i in range(1, self.p.orig_waypoints.shape[0]):
            segment = self.p.orig_waypoints[i] - self.p.orig_waypoints[i - 1]
            total += int(np.floor(np.linalg.norm(segment) / self.p.distance)) + 1
            indices.append(min(total, self.p.num_points - 1))
        return np.asarray(indices, dtype=np.int32)

    def _next_checkpoint_from_path(self, path_idx):
        if path_idx < 0:
            return 1
        passed = int(np.searchsorted(self.waypoint_path_indices, path_idx, side="right") - 1)
        return min(max(passed + 1, 1), len(self.waypoint_path_indices) - 1)

    def _update_checkpoint_order(self, ball_pos, curr_pos_path):
        if (not self.enforce_checkpoint_order) or self.next_checkpoint >= len(self.waypoint_path_indices):
            return False

        while self.next_checkpoint < len(self.waypoint_path_indices):
            waypoint = self.p.orig_waypoints[self.next_checkpoint]
            dist = float(np.linalg.norm(ball_pos - waypoint))
            target_idx = int(self.waypoint_path_indices[self.next_checkpoint])

            if curr_pos_path >= target_idx:
                print(
                    f"[Checkpoint]: passed {self.next_checkpoint} "
                    f"path={curr_pos_path} target={target_idx} dist={dist:.4f}"
                )
                self.next_checkpoint += 1
                continue

            return False

        return False

    def _closest_point(self, point):
        if self.steps <= 0 or self.prev_pos_path < 0:
            return self.p.closest_point(point)

        start = max(0, self.prev_pos_path - self.local_path_back)
        end = min(self.p.num_points, self.prev_pos_path + self.local_path_forward + 1)
        if end <= start:
            return self.p.closest_point(point)

        local_points = self.p.points[start:end]
        local_idx = int(np.argmin(np.linalg.norm(local_points - point, axis=1)))
        idx = start + local_idx
        return idx, self.p.points[idx]

    def _get_rel_path(self, point):
        idx = self._closest_point(point)[0]
        if idx == -1:
            return np.zeros((self.num_rel_path * 2,), dtype=np.float32)
        indices = np.clip(
            np.arange(idx, idx + self.num_rel_path * 60, 60),
            0,
            self.p.num_points - 1,
        )
        return (self.p.points[indices] - point).flatten().astype(np.float32)

    def _msg_to_obs(self, msg):
        # ROS message to gym observation
        if np.isnan(msg.state.x_b):
            self.ball_detected = False
        else:
            self.ball_detected = True
            states = np.array(
                [msg.state.alpha, msg.state.beta, msg.state.x_b, msg.state.y_b]
            )
            states[2:] += self.offset
            rel_path = self._get_rel_path(states[2:])
            # states = np.concatenate((states, rel_path), axis=0)
            img = self.br.imgmsg_to_cv2(msg.subimg).mean(axis=-1, keepdims=True)
            self.obs = {"states": states, "goal": rel_path, "image": img}

        self.new_obs = True

    def _normalize_states(self, states):
        return states / self.norm_max

    def _reset_board(self):
        # Reset torque
        req = DynamixelReset.Request()
        req.max_temp = 256
        self.future = self.client.call_async(req)
